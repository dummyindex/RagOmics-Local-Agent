#!/usr/bin/env python3
"""Tests for parallel job execution with job pool."""

import time
import threading
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys
import uuid
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.job_executors.job_pool import (
    JobPool, Job, JobResult, JobStatus, JobPriority
)
from ragomics_agent_local.agents.orchestrator_agent import OrchestratorAgent
from ragomics_agent_local.models import (
    AnalysisTree, AnalysisNode, NodeState, 
    NewFunctionBlock, FunctionBlockType, StaticConfig
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager


class TestJobPool(unittest.TestCase):
    """Test job pool functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.job_pool = JobPool(max_parallel_jobs=2)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'job_pool'):
            self.job_pool.shutdown(wait=False)
    
    def test_submit_and_execute_simple_job(self):
        """Test submitting and executing a simple job."""
        def simple_task(x, y):
            return x + y
        
        job_id = self.job_pool.submit_job(
            node_id="test_node",
            tree_id="test_tree",
            execute_fn=simple_task,
            args=(2, 3)
        )
        
        self.assertIsNotNone(job_id)
        
        # Wait for completion
        result = self.job_pool.wait_for_job(job_id, timeout=5)
        
        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertEqual(result.result, 5)
    
    def test_parallel_execution(self):
        """Test that jobs run in parallel."""
        execution_times = []
        lock = threading.Lock()
        
        def slow_task(task_id):
            start = time.time()
            time.sleep(0.5)  # Simulate work
            with lock:
                execution_times.append((task_id, start, time.time()))
            return f"Task {task_id} completed"
        
        # Submit 4 jobs with max_parallel=2
        job_ids = []
        for i in range(4):
            job_id = self.job_pool.submit_job(
                node_id=f"node_{i}",
                tree_id="test_tree",
                execute_fn=slow_task,
                args=(i,)
            )
            job_ids.append(job_id)
        
        # Wait for all to complete
        results = self.job_pool.wait_for_all(timeout=10)
        
        # Check all completed
        self.assertEqual(len(results), 4)
        for job_id in job_ids:
            self.assertIn(job_id, results)
            self.assertEqual(results[job_id].status, JobStatus.COMPLETED)
        
        # Check parallelism - at least 2 should overlap
        overlaps = 0
        for i in range(len(execution_times)):
            for j in range(i + 1, len(execution_times)):
                _, start1, end1 = execution_times[i]
                _, start2, end2 = execution_times[j]
                # Check if they overlapped
                if start1 < end2 and start2 < end1:
                    overlaps += 1
        
        self.assertGreaterEqual(overlaps, 1, "Jobs should run in parallel")
    
    def test_job_dependencies(self):
        """Test job dependency handling."""
        results = []
        
        def task_with_result(task_id, expected_deps):
            # Check that dependencies completed
            for dep in expected_deps:
                self.assertIn(dep, results)
            results.append(task_id)
            return task_id
        
        # Submit job A (no dependencies)
        job_a = self.job_pool.submit_job(
            node_id="node_a",
            tree_id="test_tree",
            execute_fn=task_with_result,
            args=("A", [])
        )
        
        # Submit job B (depends on A)
        job_b = self.job_pool.submit_job(
            node_id="node_b",
            tree_id="test_tree",
            execute_fn=task_with_result,
            args=("B", ["A"]),
            dependencies={job_a}
        )
        
        # Submit job C (depends on B)
        job_c = self.job_pool.submit_job(
            node_id="node_c",
            tree_id="test_tree",
            execute_fn=task_with_result,
            args=("C", ["A", "B"]),
            dependencies={job_b}
        )
        
        # Wait for all
        all_results = self.job_pool.wait_for_all(timeout=10)
        
        # Check execution order
        self.assertEqual(results, ["A", "B", "C"])
        
        # Check all completed
        for job_id in [job_a, job_b, job_c]:
            self.assertEqual(all_results[job_id].status, JobStatus.COMPLETED)
    
    def test_job_failure_handling(self):
        """Test handling of failed jobs."""
        def failing_task():
            raise ValueError("Test error")
        
        job_id = self.job_pool.submit_job(
            node_id="failing_node",
            tree_id="test_tree",
            execute_fn=failing_task
        )
        
        result = self.job_pool.wait_for_job(job_id, timeout=5)
        
        self.assertEqual(result.status, JobStatus.FAILED)
        self.assertIn("Test error", result.error)
    
    def test_job_callbacks(self):
        """Test job completion callbacks."""
        callback_results = []
        
        def callback(result: JobResult):
            callback_results.append(result)
        
        def simple_task():
            return "success"
        
        job_id = self.job_pool.submit_job(
            node_id="callback_node",
            tree_id="test_tree",
            execute_fn=simple_task,
            callback=callback
        )
        
        # Wait for completion
        self.job_pool.wait_for_job(job_id, timeout=5)
        
        # Check callback was called
        self.assertEqual(len(callback_results), 1)
        self.assertEqual(callback_results[0].job_id, job_id)
        self.assertEqual(callback_results[0].status, JobStatus.COMPLETED)
        self.assertEqual(callback_results[0].result, "success")
    
    def test_job_priority(self):
        """Test job priority handling."""
        execution_order = []
        lock = threading.Lock()
        
        # Create a pool with 1 worker to ensure sequential execution
        pool = JobPool(max_parallel_jobs=1)
        
        def task(task_id):
            with lock:
                execution_order.append(task_id)
            return task_id
        
        # Submit jobs with different priorities
        # High priority should execute first
        jobs = [
            ("low", JobPriority.LOW),
            ("high", JobPriority.HIGH),
            ("normal", JobPriority.NORMAL),
            ("critical", JobPriority.CRITICAL)
        ]
        
        job_ids = []
        for task_id, priority in jobs:
            job_id = pool.submit_job(
                node_id=f"node_{task_id}",
                tree_id="test_tree",
                execute_fn=task,
                args=(task_id,),
                priority=priority
            )
            job_ids.append(job_id)
        
        # Wait for all
        pool.wait_for_all(timeout=10)
        
        # Critical and high priority should execute before low
        critical_idx = execution_order.index("critical")
        high_idx = execution_order.index("high")
        low_idx = execution_order.index("low")
        
        self.assertLess(critical_idx, low_idx)
        self.assertLess(high_idx, low_idx)
        
        pool.shutdown(wait=False)
    
    def test_job_cancellation(self):
        """Test job cancellation."""
        # Create a pool and submit jobs that will block
        pool = JobPool(max_parallel_jobs=1)  # One worker
        
        # Submit a blocking job first
        blocking_job = pool.submit_job(
            node_id="blocking_node",
            tree_id="test_tree",
            execute_fn=lambda: time.sleep(1)
        )
        
        # Submit another job that will be queued
        job_id = pool.submit_job(
            node_id="cancel_node",
            tree_id="test_tree",
            execute_fn=lambda: "should not run"
        )
        
        # Give it a moment to queue
        time.sleep(0.1)
        
        # Cancel the queued job
        cancelled = pool.cancel_job(job_id)
        self.assertTrue(cancelled)
        
        # Check status
        self.assertIn(job_id, pool.results)
        self.assertEqual(pool.results[job_id].status, JobStatus.CANCELLED)
        
        pool.shutdown(wait=False)
    
    def test_pool_status(self):
        """Test getting pool status."""
        status = self.job_pool.get_status()
        
        self.assertEqual(status["max_parallel_jobs"], 2)
        self.assertEqual(status["total_jobs"], 0)
        self.assertEqual(status["running"], 0)
        self.assertEqual(status["completed"], 0)
        self.assertEqual(status["failed"], 0)
        
        # Submit a job
        self.job_pool.submit_job(
            node_id="status_node",
            tree_id="test_tree",
            execute_fn=lambda: time.sleep(0.1)
        )
        
        # Check status again
        status = self.job_pool.get_status()
        self.assertGreaterEqual(status["total_jobs"], 1)


class TestOrchestratorParallelExecution(unittest.TestCase):
    """Test orchestrator agent with parallel execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tree_manager = Mock(spec=AnalysisTreeManager)
        self.function_creator = Mock()
        self.bug_fixer = Mock()
        
        self.orchestrator = OrchestratorAgent(
            tree_manager=self.tree_manager,
            function_creator=self.function_creator,
            bug_fixer=self.bug_fixer,
            max_parallel_jobs=2
        )
        
        # Create test tree
        self.tree = AnalysisTree(
            id=str(uuid.uuid4()),
            root_node_id="root_node",
            user_request="Test analysis",
            input_data_path="/test/data.h5ad",
            max_nodes=10,
            max_children_per_node=3,
            max_debug_trials=3
        )
        
        # Create test nodes
        self.root_node = AnalysisNode(
            id="root_node",
            analysis_id=self.tree.id,
            function_block=NewFunctionBlock(
                name="root_analysis",
                type=FunctionBlockType.PYTHON,
                description="Root analysis",
                code="""def run(path_dict, params):
    import os
    # Follow framework convention
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    if adata is None:
        # Load from standard input location
        import scanpy as sc
        adata = sc.read_h5ad('/workspace/input/_node_anndata.h5ad')
    # Process (minimal for test)
    # Save to standard output location
    os.makedirs('/workspace/output', exist_ok=True)
    adata.write(os.path.join(path_dict["output_dir"], "_node_anndata.h5ad"))
    return adata""",
                requirements="",
                static_config=StaticConfig(
                    args=[],
                    description="Root analysis",
                    tag="test"
                )
            ),
            state=NodeState.PENDING,
            level=0
        )
        
        self.child_node1 = AnalysisNode(
            id="child_1",
            parent_id="root_node",
            analysis_id=self.tree.id,
            function_block=NewFunctionBlock(
                name="child_analysis_1",
                type=FunctionBlockType.PYTHON,
                description="Child analysis 1",
                code="""def run(path_dict, params):
    import os
    # Follow framework convention
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    if adata is None:
        # Load from standard input location
        import scanpy as sc
        adata = sc.read_h5ad('/workspace/input/_node_anndata.h5ad')
    # Process (minimal for test)
    # Save to standard output location
    os.makedirs('/workspace/output', exist_ok=True)
    adata.write(os.path.join(path_dict["output_dir"], "_node_anndata.h5ad"))
    return adata""",
                requirements="",
                static_config=StaticConfig(
                    args=[],
                    description="Child analysis 1",
                    tag="test"
                )
            ),
            state=NodeState.PENDING,
            level=1
        )
        
        self.child_node2 = AnalysisNode(
            id="child_2",
            parent_id="root_node",
            analysis_id=self.tree.id,
            function_block=NewFunctionBlock(
                name="child_analysis_2",
                type=FunctionBlockType.PYTHON,
                description="Child analysis 2",
                code="""def run(path_dict, params):
    import os
    # Follow framework convention
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    if adata is None:
        # Load from standard input location
        import scanpy as sc
        adata = sc.read_h5ad('/workspace/input/_node_anndata.h5ad')
    # Process (minimal for test)
    # Save to standard output location
    os.makedirs('/workspace/output', exist_ok=True)
    adata.write(os.path.join(path_dict["output_dir"], "_node_anndata.h5ad"))
    return adata""",
                requirements="",
                static_config=StaticConfig(
                    args=[],
                    description="Child analysis 2",
                    tag="test"
                )
            ),
            state=NodeState.PENDING,
            level=1
        )
        
        self.tree.nodes = {
            "root_node": self.root_node,
            "child_1": self.child_node1,
            "child_2": self.child_node2
        }
        
        self.root_node.children = ["child_1", "child_2"]
    
    def test_parallel_node_execution(self):
        """Test that child nodes execute in parallel."""
        # Mock node executor
        node_executor = Mock()
        execution_times = []
        
        def mock_execute(node, tree, input_path, output_base_dir):
            start = time.time()
            time.sleep(0.2)  # Simulate work
            execution_times.append((node.id, start, time.time()))
            return NodeState.COMPLETED, f"/output/{node.id}"
        
        node_executor.execute_node.side_effect = mock_execute
        
        # Configure tree manager
        self.tree_manager.can_continue_expansion.return_value = False
        
        # Run orchestrator
        context = {
            "tree": self.tree,
            "user_request": "Test analysis",
            "max_iterations": 1,
            "node_executor": node_executor,
            "output_dir": Path("/test/output")
        }
        
        result = self.orchestrator.process(context)
        
        # Check that nodes were executed
        self.assertEqual(node_executor.execute_node.call_count, 3)
        
        # Check parallelism - child nodes should overlap
        root_times = next(t for t in execution_times if t[0] == "root_node")
        child1_times = next(t for t in execution_times if t[0] == "child_1")
        child2_times = next(t for t in execution_times if t[0] == "child_2")
        
        # Children should start after root completes
        self.assertGreater(child1_times[1], root_times[2])
        self.assertGreater(child2_times[1], root_times[2])
        
        # Children should overlap (parallel execution)
        self.assertTrue(
            child1_times[1] < child2_times[2] and child2_times[1] < child1_times[2],
            "Child nodes should execute in parallel"
        )
    
    def test_reactive_node_expansion(self):
        """Test reactive expansion on node completion."""
        # Mock node executor
        node_executor = Mock()
        node_executor.execute_node.return_value = (NodeState.COMPLETED, "/output/node")
        
        # Mock tree manager to allow expansion
        self.tree_manager.can_continue_expansion.return_value = True
        self.tree_manager.add_child_nodes.return_value = []
        
        # Mock function selector to suggest new nodes
        self.function_creator.process_selection_or_creation.return_value = {
            "satisfied": False,
            "function_blocks": [
                NewFunctionBlock(
                    name="new_analysis",
                    type=FunctionBlockType.PYTHON,
                    description="New analysis",
                    code="def run(path_dict, params): return None",
                    requirements="",
                    parameters={},
                    static_config=StaticConfig(args=[], description="Test", tag="test")
                )
            ]
        }
        
        # Run orchestrator
        context = {
            "tree": self.tree,
            "user_request": "Test analysis",
            "max_iterations": 2,
            "node_executor": node_executor,
            "output_dir": Path("/test/output")
        }
        
        result = self.orchestrator.process(context)
        
        # Check that expansion decisions were made
        self.assertGreater(len(self.orchestrator.expansion_decisions), 0)
        
        # Check that nodes were marked for expansion
        for node_id in self.orchestrator.completed_nodes:
            if node_id in self.orchestrator.expansion_decisions:
                decision = self.orchestrator.expansion_decisions[node_id]
                self.assertIn("should_expand", decision)
    
    def test_dependency_handling(self):
        """Test that child nodes wait for parent completion."""
        # Track execution order
        execution_order = []
        
        def mock_execute(node, tree, input_path, output_base_dir):
            execution_order.append(node.id)
            time.sleep(0.1)
            return NodeState.COMPLETED, f"/output/{node.id}"
        
        node_executor = Mock()
        node_executor.execute_node.side_effect = mock_execute
        
        # Configure tree manager
        self.tree_manager.can_continue_expansion.return_value = False
        
        # Run orchestrator
        context = {
            "tree": self.tree,
            "user_request": "Test analysis",
            "max_iterations": 1,
            "node_executor": node_executor,
            "output_dir": Path("/test/output")
        }
        
        result = self.orchestrator.process(context)
        
        # Check execution order - root should complete before children
        root_idx = execution_order.index("root_node")
        child1_idx = execution_order.index("child_1")
        child2_idx = execution_order.index("child_2")
        
        self.assertLess(root_idx, child1_idx)
        self.assertLess(root_idx, child2_idx)
    
    def test_failure_handling(self):
        """Test handling of node execution failures."""
        # Mock node executor to fail for one child
        def mock_execute(node, tree, input_path, output_base_dir):
            if node.id == "child_1":
                raise ValueError("Test failure")
            return NodeState.COMPLETED, f"/output/{node.id}"
        
        node_executor = Mock()
        node_executor.execute_node.side_effect = mock_execute
        
        # Configure tree manager
        self.tree_manager.can_continue_expansion.return_value = False
        
        # Run orchestrator
        context = {
            "tree": self.tree,
            "user_request": "Test analysis",
            "max_iterations": 1,
            "node_executor": node_executor,
            "output_dir": Path("/test/output")
        }
        
        result = self.orchestrator.process(context)
        
        # Check that failure was recorded
        self.assertIn("child_1", result["failed_nodes"])
        self.assertIn("root_node", result["completed_nodes"])
        self.assertIn("child_2", result["completed_nodes"])
    
    def test_max_parallel_jobs_limit(self):
        """Test that max parallel jobs limit is respected."""
        # Create more nodes than max parallel
        for i in range(3, 6):
            node = AnalysisNode(
                id=f"child_{i}",
                parent_id="root_node",
                analysis_id=self.tree.id,
                function_block=NewFunctionBlock(
                    name=f"child_analysis_{i}",
                    type=FunctionBlockType.PYTHON,
                    description=f"Child analysis {i}",
                    code="def run(path_dict, params): return None",
                    requirements="",
                    parameters={},
                    static_config=StaticConfig(args=[], description="Test", tag="test")
                ),
                state=NodeState.PENDING,
                level=1
            )
            self.tree.nodes[f"child_{i}"] = node
            self.root_node.children.append(f"child_{i}")
        
        # Track concurrent executions
        concurrent_count = []
        lock = threading.Lock()
        current_running = 0
        
        def mock_execute(node, tree, input_path, output_base_dir):
            nonlocal current_running
            with lock:
                current_running += 1
                concurrent_count.append(current_running)
            
            time.sleep(0.1)  # Simulate work
            
            with lock:
                current_running -= 1
            
            return NodeState.COMPLETED, f"/output/{node.id}"
        
        node_executor = Mock()
        node_executor.execute_node.side_effect = mock_execute
        
        # Configure tree manager
        self.tree_manager.can_continue_expansion.return_value = False
        
        # Run orchestrator with max_parallel_jobs=2
        self.orchestrator.max_parallel_jobs = 2
        
        context = {
            "tree": self.tree,
            "user_request": "Test analysis",
            "max_iterations": 1,
            "node_executor": node_executor,
            "output_dir": Path("/test/output")
        }
        
        result = self.orchestrator.process(context)
        
        # Check that we never exceeded max parallel jobs
        # Root runs alone first, then children with max 2 at a time
        max_concurrent = max(concurrent_count)
        self.assertLessEqual(max_concurrent, 2, 
                           f"Max concurrent executions {max_concurrent} exceeded limit of 2")


class TestIntegration(unittest.TestCase):
    """Integration tests for parallel execution system."""
    
    def test_end_to_end_parallel_execution(self):
        """Test complete workflow with parallel execution."""
        # This would be a more comprehensive integration test
        # using real components rather than mocks
        pass


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()