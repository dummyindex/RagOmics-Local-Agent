#!/usr/bin/env python3
"""Verify that all test outputs comply with the tree structure specifications."""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def check_structure_compliance(output_dir: Path) -> Tuple[bool, List[str]]:
    """Check if an output directory complies with the structure specifications.
    
    Returns:
        Tuple of (is_compliant, list_of_issues)
    """
    issues = []
    
    # Check 1: analysis_tree.json at base level
    analysis_tree_path = output_dir / "analysis_tree.json"
    if not analysis_tree_path.exists():
        issues.append("❌ analysis_tree.json not found at base level")
    else:
        # Load tree to get ID
        try:
            with open(analysis_tree_path) as f:
                tree_data = json.load(f)
                tree_id = tree_data.get('id')
                
                if tree_id:
                    # Check 2: Tree directory with just UUID (no tree_ prefix)
                    tree_dir = output_dir / tree_id
                    if not tree_dir.exists():
                        issues.append(f"❌ Tree directory '{tree_id}' not found")
                        # Check if there's a tree_ prefixed version (old format)
                        old_tree_dir = output_dir / f"tree_{tree_id}"
                        if old_tree_dir.exists():
                            issues.append(f"  ⚠️  Found old format: tree_{tree_id}")
                    else:
                        # Check 3: nodes/ directory inside tree directory
                        nodes_dir = tree_dir / "nodes"
                        if not nodes_dir.exists():
                            issues.append("❌ nodes/ directory not found in tree directory")
                        else:
                            # Check node structures
                            node_dirs = [d for d in nodes_dir.iterdir() if d.is_dir()]
                            if node_dirs:
                                sample_node = node_dirs[0]
                                
                                # Check node subdirectories
                                required_dirs = ["function_block", "jobs", "outputs", "agent_tasks"]
                                for req_dir in required_dirs:
                                    if not (sample_node / req_dir).exists():
                                        issues.append(f"❌ {req_dir}/ not found in node directory")
                                
                                # Check node_info.json
                                if not (sample_node / "node_info.json").exists():
                                    issues.append("❌ node_info.json not found in node directory")
                                
                                # Check jobs structure
                                jobs_dir = sample_node / "jobs"
                                if jobs_dir.exists():
                                    job_dirs = [d for d in jobs_dir.iterdir() if d.is_dir() and d.name.startswith("job_")]
                                    if job_dirs:
                                        sample_job = job_dirs[0]
                                        
                                        # Check job subdirectories
                                        job_required = ["input", "logs", "output"]
                                        for req_dir in job_required:
                                            if not (sample_job / req_dir).exists():
                                                issues.append(f"❌ {req_dir}/ not found in job directory")
                                        
                                        # Check output subdirectories
                                        output_dir = sample_job / "output"
                                        if output_dir.exists():
                                            if not (output_dir / "figures").exists():
                                                issues.append("❌ figures/ not found in job output")
                                            if not (output_dir / "past_jobs").exists():
                                                issues.append("❌ past_jobs/ not found in job output")
                                    
                                    # Check latest symlink
                                    latest_link = jobs_dir / "latest"
                                    if not latest_link.exists() and not latest_link.is_symlink():
                                        issues.append("❌ latest symlink not found in jobs directory")
                                
                                # Check outputs directory
                                outputs_dir = sample_node / "outputs"
                                if outputs_dir.exists():
                                    # Check for _node_anndata.h5ad
                                    if not (outputs_dir / "_node_anndata.h5ad").exists():
                                        issues.append("⚠️  _node_anndata.h5ad not found in outputs (may be expected)")
                            else:
                                issues.append("⚠️  No node directories found (tree might be empty)")
        except Exception as e:
            issues.append(f"❌ Error reading analysis_tree.json: {e}")
    
    # Check for incorrect node directories at root level
    root_node_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("node_")]
    if root_node_dirs:
        issues.append(f"❌ Found {len(root_node_dirs)} node directories at root level (should be under tree_id/nodes/)")
        for node_dir in root_node_dirs[:3]:  # Show first 3
            issues.append(f"  - {node_dir.name}")
    
    return len(issues) == 0, issues


def main():
    """Check all test outputs for compliance."""
    print("=" * 80)
    print("VERIFYING TEST OUTPUT STRUCTURES")
    print("=" * 80)
    
    # Use consistent test output directory
    from test_utils import get_test_output_dir
    test_outputs_dir = get_test_output_dir()
    
    if not test_outputs_dir.exists():
        print("❌ No test_outputs directory found")
        return
    
    # Find all test output directories
    test_dirs = [d for d in test_outputs_dir.iterdir() if d.is_dir()]
    
    if not test_dirs:
        print("❌ No test output directories found")
        return
    
    print(f"\nFound {len(test_dirs)} test output directories to verify:\n")
    
    all_compliant = True
    results = []
    
    for test_dir in sorted(test_dirs):
        print(f"📁 Checking: {test_dir.name}")
        
        compliant, issues = check_structure_compliance(test_dir)
        results.append((test_dir.name, compliant, issues))
        
        if compliant:
            print("  ✅ COMPLIANT with structure specifications")
        else:
            print(f"  ❌ NOT COMPLIANT - {len(issues)} issues found")
            all_compliant = False
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    compliant_count = sum(1 for _, compliant, _ in results if compliant)
    print(f"\n✅ Compliant: {compliant_count}/{len(results)}")
    print(f"❌ Non-compliant: {len(results) - compliant_count}/{len(results)}")
    
    if not all_compliant:
        print("\n⚠️  ISSUES FOUND:\n")
        for dir_name, compliant, issues in results:
            if not compliant:
                print(f"📁 {dir_name}:")
                for issue in issues:
                    print(f"   {issue}")
                print()
    else:
        print("\n🎉 All test outputs are compliant with the structure specifications!")
    
    print("=" * 80)
    
    return all_compliant


if __name__ == "__main__":
    import sys
    compliant = main()
    sys.exit(0 if compliant else 1)