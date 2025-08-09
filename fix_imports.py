#!/usr/bin/env python3
"""Fix relative imports in the codebase."""

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Files with relative imports to fix
FILES_TO_FIX = [
    "agents/base_agent.py",
    "agents/claude_code_sdk_bug_fixer.py", 
    "agents/function_creator_agent.py",
    "agents/orchestrator_agent.py",
    "agents/schemas.py",
    "agents/task_manager.py",
    "llm_service/mock_service.py",
    "llm_service/prompt_builder.py",
    "utils/function_block_loader.py"
]

def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    shutil.copy2(file_path, backup_path)
    return backup_path

def convert_imports(file_path, dry_run=True):
    """Convert relative imports to absolute imports in a file."""
    with open(file_path, 'r') as f:
        original_content = f.read()
    
    content = original_content
    changes = []
    replacements_made = set()  # Track what we've already replaced
    
    # Pattern 1: from ..module import -> from module import
    pattern1 = re.compile(r'from \.\.([\w.]+) import (.+)')
    for match in pattern1.finditer(original_content):
        old_import = match.group(0)
        if old_import not in replacements_made:
            module = match.group(1).lstrip('.')  # Remove any leading dots
            items = match.group(2)
            new_import = f'from {module} import {items}'
            content = content.replace(old_import, new_import, 1)
            changes.append((old_import, new_import))
            replacements_made.add(old_import)
    
    # Pattern 2: from .module import -> from current_package.module import
    pattern2 = re.compile(r'from \.([\w.]+) import (.+)')
    
    # Determine current package based on file location
    parts = file_path.parts
    if 'agents' in parts:
        current_package = 'agents'
    elif 'llm_service' in parts:
        current_package = 'llm_service'
    elif 'utils' in parts:
        current_package = 'utils'
    else:
        current_package = None
    
    if current_package:
        for match in pattern2.finditer(original_content):
            old_import = match.group(0)
            if old_import not in replacements_made:
                module = match.group(1)
                items = match.group(2)
                new_import = f'from {current_package}.{module} import {items}'
                content = content.replace(old_import, new_import, 1)
                changes.append((old_import, new_import))
                replacements_made.add(old_import)
    
    return content, changes

def fix_imports(dry_run=True):
    """Fix imports in all files."""
    project_root = Path(__file__).parent
    
    print(f"{'DRY RUN' if dry_run else 'FIXING'} imports in {len(FILES_TO_FIX)} files...")
    print("=" * 60)
    
    all_changes = {}
    
    for file_rel_path in FILES_TO_FIX:
        file_path = project_root / file_rel_path
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
        
        try:
            new_content, changes = convert_imports(file_path, dry_run)
            
            if changes:
                all_changes[file_rel_path] = changes
                print(f"\n📄 {file_rel_path}")
                for old, new in changes:
                    print(f"  - {old}")
                    print(f"  + {new}")
                
                if not dry_run:
                    # Backup original file
                    backup_path = backup_file(file_path)
                    print(f"  ✓ Backed up to {backup_path.name}")
                    
                    # Write new content
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    print(f"  ✓ Updated {file_path.name}")
            else:
                print(f"\n📄 {file_rel_path} - No changes needed")
                
        except Exception as e:
            print(f"\n❌ Error processing {file_rel_path}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Summary: {len(all_changes)} files with changes")
    
    if dry_run:
        print("\nThis was a DRY RUN. To apply changes, run with --fix flag")
    else:
        print(f"\n✓ Import fixes applied to {len(all_changes)} files")
        print("✓ Backup files created with .backup extension")
    
    return all_changes

def verify_imports():
    """Verify that imports work after fixes."""
    print("\nVerifying imports...")
    print("=" * 60)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    success_count = 0
    fail_count = 0
    
    # Try importing key modules
    test_imports = [
        "from agents.base_agent import BaseAgent",
        "from agents.main_agent import MainAgent",
        "from llm_service.prompt_builder import PromptBuilder",
        "from utils.logger import get_logger",
        "from models import AnalysisTree"
    ]
    
    for import_statement in test_imports:
        try:
            exec(import_statement)
            print(f"✓ {import_statement}")
            success_count += 1
        except Exception as e:
            print(f"❌ {import_statement} - {e}")
            fail_count += 1
    
    print(f"\nVerification: {success_count} passed, {fail_count} failed")
    return fail_count == 0

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix relative imports in the codebase")
    parser.add_argument('--fix', action='store_true', help='Apply fixes (default is dry run)')
    parser.add_argument('--verify', action='store_true', help='Verify imports after fix')
    args = parser.parse_args()
    
    print(f"Import Fixer - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run fixes
    changes = fix_imports(dry_run=not args.fix)
    
    # Verify if requested
    if args.verify and args.fix:
        verify_imports()
    elif args.verify and not args.fix:
        print("\nNote: --verify only works with --fix")

if __name__ == "__main__":
    main()