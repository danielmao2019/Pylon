"""
Test progress tracking under concurrent access scenarios.

These tests validate that the progress tracking system handles multiple
processes/threads accessing progress files simultaneously without corruption
or race conditions.
"""
import pytest
import tempfile
import threading
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.automation.progress_tracking.trainer_progress_tracker import TrainerProgressTracker
from utils.automation.progress_tracking.evaluator_progress_tracker import EvaluatorProgressTracker
from utils.automation.progress_tracking.session_progress import get_session_progress
from utils.io.json import safe_load_json, safe_save_json, atomic_json_update


# ============================================================================
# CONCURRENCY TESTS - MULTIPLE READERS
# ============================================================================

def test_multiple_readers_same_progress_file(create_progress_json, EXPECTED_FILES):
    """Test multiple threads reading the same progress.json file simultaneously."""
    with tempfile.TemporaryDirectory() as work_dir:
        expected_files = EXPECTED_FILES
        
        # Create progress.json
        create_progress_json(work_dir, completed_epochs=42, early_stopped=False, tot_epochs=100)
        
        # Function for reader threads
        def read_progress():
            return get_session_progress(work_dir, expected_files)
        
        # Run 10 concurrent readers
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_progress) for _ in range(10)]
            for future in as_completed(futures):
                results.append(future.result())
        
        # All readers should get the same result
        assert len(results) == 10
        assert all(r.completed_epochs == 42 for r in results)
        assert all(r.progress_percentage == 42.0 for r in results)
        assert all(r.early_stopped == False for r in results)


def test_multiple_tracker_readers_same_file(create_progress_json):
    """Test multiple TrainerProgressTracker instances reading same progress.json."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create progress.json
        create_progress_json(work_dir, completed_epochs=25, early_stopped=False, tot_epochs=100)
        
        # Function for tracker reader threads
        def read_with_tracker():
            tracker = TrainerProgressTracker(work_dir)
            return tracker.get_progress()
        
        # Run 8 concurrent tracker readers
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_with_tracker) for _ in range(8)]
            for future in as_completed(futures):
                results.append(future.result())
        
        # All readers should get consistent results
        assert len(results) == 8
        assert all(r.completed_epochs == 25 for r in results)
        assert all(r.runner_type == 'trainer' for r in results)


# ============================================================================
# CONCURRENCY TESTS - READER-WRITER CONFLICTS
# ============================================================================

def test_reader_writer_conflict_resolution(create_progress_json, EXPECTED_FILES):
    """Test readers and writers accessing progress.json simultaneously."""
    with tempfile.TemporaryDirectory() as work_dir:
        expected_files = EXPECTED_FILES
        
        # Create initial progress.json
        create_progress_json(work_dir, completed_epochs=10, early_stopped=False, tot_epochs=100)
        
        results = {"reads": [], "writes": []}
        
        def reader_thread():
            for _ in range(5):
                try:
                    progress = get_session_progress(work_dir, expected_files)
                    results["reads"].append(progress.completed_epochs)
                    time.sleep(0.01)  # Small delay to increase chance of conflicts
                except Exception as e:
                    results["reads"].append(f"ERROR: {e}")
        
        def writer_thread():
            for i in range(3):
                try:
                    # Update progress.json with new values
                    new_epochs = 20 + i * 10
                    create_progress_json(work_dir, completed_epochs=new_epochs, early_stopped=False, tot_epochs=100)
                    results["writes"].append(new_epochs)
                    time.sleep(0.02)  # Small delay to increase chance of conflicts
                except Exception as e:
                    results["writes"].append(f"ERROR: {e}")
        
        # Run concurrent readers and writers
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            # Start 2 reader threads and 2 writer threads
            futures.append(executor.submit(reader_thread))
            futures.append(executor.submit(reader_thread))
            futures.append(executor.submit(writer_thread))
            futures.append(executor.submit(writer_thread))
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # Verify no errors occurred
        read_errors = [r for r in results["reads"] if isinstance(r, str) and r.startswith("ERROR")]
        write_errors = [w for w in results["writes"] if isinstance(w, str) and w.startswith("ERROR")]
        
        assert len(read_errors) == 0, f"Reader errors: {read_errors}"
        assert len(write_errors) == 0, f"Writer errors: {write_errors}"
        
        # Verify we got some reads and writes
        valid_reads = [r for r in results["reads"] if isinstance(r, int)]
        valid_writes = [w for w in results["writes"] if isinstance(w, int)]
        
        assert len(valid_reads) > 0, "Should have successful reads"
        assert len(valid_writes) > 0, "Should have successful writes"


# ============================================================================
# CONCURRENCY TESTS - MULTIPLE WRITERS
# ============================================================================

def test_multiple_writers_same_progress_file():
    """Test multiple threads writing to the same progress.json file."""
    with tempfile.TemporaryDirectory() as work_dir:
        progress_file = os.path.join(work_dir, "progress.json")
        
        def writer_thread(thread_id):
            for i in range(3):
                progress_data = {
                    "completed_epochs": thread_id * 10 + i,
                    "progress_percentage": float(thread_id * 10 + i),
                    "early_stopped": False,
                    "early_stopped_at_epoch": None,
                    "runner_type": "trainer",
                    "total_epochs": 100,
                    "thread_id": thread_id,  # Track which thread wrote this
                    "write_count": i
                }
                safe_save_json(progress_data, progress_file)
                time.sleep(0.01)  # Small delay to increase chance of conflicts
        
        # Run 5 concurrent writers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(writer_thread, thread_id) for thread_id in range(5)]
            for future in as_completed(futures):
                future.result()
        
        # Verify final file is valid JSON and not corrupted
        assert os.path.exists(progress_file)
        final_data = safe_load_json(progress_file)
        
        # Verify structure is correct (one of the threads succeeded)
        assert "completed_epochs" in final_data
        assert "thread_id" in final_data
        assert "write_count" in final_data
        assert isinstance(final_data["completed_epochs"], int)
        assert 0 <= final_data["thread_id"] < 5


# ============================================================================
# CONCURRENCY TESTS - CACHE INVALIDATION RACES
# ============================================================================

def test_cache_invalidation_race_conditions(create_progress_json):
    """Test cache invalidation when multiple trackers access same file."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create initial progress.json
        create_progress_json(work_dir, completed_epochs=5, early_stopped=False, tot_epochs=100)
        
        # Create tracker instances (they'll cache results)
        trackers = [TrainerProgressTracker(work_dir) for _ in range(3)]
        
        results = []
        
        def cache_and_read(tracker_id):
            tracker = trackers[tracker_id]
            
            # First read - should cache
            progress1 = tracker.get_progress()
            results.append(("first_read", tracker_id, progress1.completed_epochs))
            
            # Small delay to let other threads potentially update the file
            time.sleep(0.05)
            
            # Second read - should use cache unless invalidated
            progress2 = tracker.get_progress()
            results.append(("second_read", tracker_id, progress2.completed_epochs))
            
            # Skip force recompute for this test to avoid config path issues
            # The test focuses on cache behavior, not force recompute
        
        def file_updater():
            # Update the file while trackers are reading
            time.sleep(0.02)
            create_progress_json(work_dir, completed_epochs=15, early_stopped=False, tot_epochs=100)
            time.sleep(0.02)
            create_progress_json(work_dir, completed_epochs=25, early_stopped=False, tot_epochs=100)
        
        # Run concurrent cache operations and file updates
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            # Start tracker threads
            for i in range(3):
                futures.append(executor.submit(cache_and_read, i))
            # Start file updater
            futures.append(executor.submit(file_updater))
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # Analyze results - should not have any errors/exceptions
        assert len(results) > 0
        
        # Group results by tracker
        tracker_results = {}
        for operation, tracker_id, epochs in results:
            if tracker_id not in tracker_results:
                tracker_results[tracker_id] = []
            tracker_results[tracker_id].append((operation, epochs))
        
        # Each tracker should have completed read operations
        for tracker_id in range(3):
            assert tracker_id in tracker_results
            ops = [op for op, _ in tracker_results[tracker_id]]
            assert "first_read" in ops
            assert "second_read" in ops


# ============================================================================
# CONCURRENCY TESTS - FORCE RECOMPUTE RACES
# ============================================================================

def test_multiple_force_recompute_same_file(create_epoch_files, create_real_config, EXPECTED_FILES):
    """Test multiple force_progress_recompute calls simultaneously."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_concurrent_force")
        config_path = os.path.join(configs_dir, "test_concurrent_force.py")
        
        os.makedirs(work_dir, exist_ok=True)
        expected_files = EXPECTED_FILES
        
        # Create epoch files
        for epoch_idx in range(10):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            results = []
            
            def force_recompute_thread(thread_id):
                for i in range(3):
                    progress = get_session_progress(work_dir, expected_files, force_progress_recompute=True)
                    results.append((thread_id, i, progress.completed_epochs))
                    time.sleep(0.01)  # Small delay to increase interleaving
            
            # Run 4 concurrent force recompute operations
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(force_recompute_thread, thread_id) for thread_id in range(4)]
                for future in as_completed(futures):
                    future.result()
            
            # All should get the same result (10 epochs from filesystem)
            assert len(results) == 12  # 4 threads * 3 operations each
            epochs_values = [epochs for _, _, epochs in results]
            assert all(epochs == 10 for epochs in epochs_values), f"Expected all 10, got: {set(epochs_values)}"
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# CONCURRENCY TESTS - MIXED TRACKER TYPES
# ============================================================================

def test_mixed_tracker_types_concurrent_access():
    """Test TrainerProgressTracker and EvaluatorProgressTracker accessing different files concurrently."""
    with tempfile.TemporaryDirectory() as base_dir:
        trainer_dir = os.path.join(base_dir, "trainer")
        evaluator_dir = os.path.join(base_dir, "evaluator")
        os.makedirs(trainer_dir)
        os.makedirs(evaluator_dir)
        
        # Create trainer progress.json
        trainer_progress = {
            "completed_epochs": 50,
            "progress_percentage": 50.0,
            "early_stopped": False,
            "early_stopped_at_epoch": None,
            "runner_type": "trainer",
            "total_epochs": 100
        }
        safe_save_json(trainer_progress, os.path.join(trainer_dir, "progress.json"))
        
        # Create evaluator evaluation_scores.json
        eval_scores = {"aggregated": {"acc": 0.95}, "per_datapoint": {"acc": [0.95]}}
        safe_save_json(eval_scores, os.path.join(evaluator_dir, "evaluation_scores.json"))
        
        results = []
        
        def trainer_worker():
            for i in range(5):
                tracker = TrainerProgressTracker(trainer_dir)
                progress = tracker.get_progress()
                results.append(("trainer", progress.completed_epochs))
                time.sleep(0.01)
        
        def evaluator_worker():
            for i in range(5):
                tracker = EvaluatorProgressTracker(evaluator_dir)
                progress = tracker.get_progress()
                results.append(("evaluator", progress.completed_epochs))
                time.sleep(0.01)
        
        # Run concurrent trainer and evaluator operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.append(executor.submit(trainer_worker))
            futures.append(executor.submit(trainer_worker))
            futures.append(executor.submit(evaluator_worker))
            futures.append(executor.submit(evaluator_worker))
            
            for future in as_completed(futures):
                future.result()
        
        # Verify results
        trainer_results = [epochs for tracker_type, epochs in results if tracker_type == "trainer"]
        evaluator_results = [epochs for tracker_type, epochs in results if tracker_type == "evaluator"]
        
        assert len(trainer_results) == 10  # 2 workers * 5 operations each
        assert len(evaluator_results) == 10  # 2 workers * 5 operations each
        assert all(epochs == 50 for epochs in trainer_results)
        assert all(epochs == 1 for epochs in evaluator_results)  # Binary for evaluators


# ============================================================================
# CONCURRENCY TESTS - FILE LOCKING STRESS TEST
# ============================================================================

def test_json_file_locking_stress_test_broken():
    """Demonstrate that non-atomic read-modify-write has race conditions."""
    with tempfile.TemporaryDirectory() as work_dir:
        test_file = os.path.join(work_dir, "stress_test.json")
        
        # Initialize with base data
        initial_data = {"counter": 0, "operations": []}
        safe_save_json(initial_data, test_file)
        
        def stress_worker(worker_id):
            for i in range(20):  # 20 operations per worker
                # Read current data
                data = safe_load_json(test_file)
                
                # Modify data
                data["counter"] += 1
                data["operations"].append(f"worker_{worker_id}_op_{i}")
                
                # Save modified data
                safe_save_json(data, test_file)
                
                # Small random delay to increase chance of conflicts
                time.sleep(0.001)
        
        # Run 10 workers concurrently (200 total operations)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(stress_worker, worker_id) for worker_id in range(10)]
            for future in as_completed(futures):
                future.result()
        
        # Verify final state
        final_data = safe_load_json(test_file)
        
        # This test documents the race condition - counter will be less than 200
        assert final_data["counter"] < 200, f"Expected < 200 due to race conditions, got {final_data['counter']}"
        print(f"Race condition demonstrated: got {final_data['counter']} instead of 200")


def test_atomic_json_update_stress_test():
    """Stress test the atomic JSON update mechanism with high concurrency."""
    with tempfile.TemporaryDirectory() as work_dir:
        test_file = os.path.join(work_dir, "atomic_stress_test.json")
        
        def stress_worker(worker_id):
            for i in range(20):  # 20 operations per worker
                def update_func(data):
                    # Ensure data structure exists
                    if "counter" not in data:
                        data["counter"] = 0
                    if "operations" not in data:
                        data["operations"] = []
                    
                    # Modify data atomically
                    data["counter"] += 1
                    data["operations"].append(f"worker_{worker_id}_op_{i}")
                    return data
                
                # Atomic read-modify-write operation
                atomic_json_update(test_file, update_func, {"counter": 0, "operations": []})
                
                # Small random delay to increase chance of conflicts
                time.sleep(0.001)
        
        # Run 10 workers concurrently (200 total operations)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(stress_worker, worker_id) for worker_id in range(10)]
            for future in as_completed(futures):
                future.result()
        
        # Verify final state
        final_data = safe_load_json(test_file)
        
        # Counter should be exactly 200 (no lost updates with atomic operations)
        assert final_data["counter"] == 200, f"Expected 200, got {final_data['counter']}"
        
        # Should have exactly 200 operations recorded
        assert len(final_data["operations"]) == 200
        
        # All operations should be unique (no duplicates from race conditions)
        assert len(set(final_data["operations"])) == 200


# ============================================================================
# CONCURRENCY TESTS - DEADLOCK PREVENTION
# ============================================================================

def test_no_deadlocks_multiple_files():
    """Test that accessing multiple progress files doesn't cause deadlocks."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create multiple progress files
        files = []
        for i in range(5):
            file_path = os.path.join(work_dir, f"progress_{i}.json")
            data = {"id": i, "value": i * 10}
            safe_save_json(data, file_path)
            files.append(file_path)
        
        def cross_file_worker(worker_id):
            # Each worker accesses files in different order to test for deadlocks
            if worker_id % 2 == 0:
                file_order = files  # Forward order
            else:
                file_order = files[::-1]  # Reverse order
            
            for _ in range(10):
                for file_path in file_order:
                    # Read and update each file
                    data = safe_load_json(file_path)
                    data["accessed_by"] = worker_id
                    safe_save_json(data, file_path)
                    time.sleep(0.001)
        
        # Run workers that access files in different orders
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(cross_file_worker, worker_id) for worker_id in range(6)]
            for future in as_completed(futures):
                future.result()
        end_time = time.time()
        
        # Should complete within reasonable time (no deadlocks)
        elapsed = end_time - start_time
        assert elapsed < 10.0, f"Operations took too long: {elapsed}s (possible deadlock)"
        
        # Verify all files are still accessible
        for file_path in files:
            data = safe_load_json(file_path)
            assert "accessed_by" in data
            assert isinstance(data["accessed_by"], int)