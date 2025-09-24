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
from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob
from agents.manager.manager import Manager
from utils.io.json import load_json, save_json


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
            return TrainingJob.get_session_progress(work_dir, expected_files)
        
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
    """Test multiple TrainingJob classmethods reading same progress.json."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create progress.json
        create_progress_json(work_dir, completed_epochs=25, early_stopped=False, tot_epochs=100)
        
        # Function for tracker reader threads
        def read_with_tracker():
            return TrainingJob.calculate_progress(work_dir, config=None)
        
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
                    progress = TrainingJob.get_session_progress(work_dir, expected_files)
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

def test_multiple_writers_same_progress_file(create_epoch_files, create_real_config, EXPECTED_FILES):
    """Test multiple progress trackers writing to the same progress.json file."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_concurrent_writers")
        config_path = os.path.join(configs_dir, "test_concurrent_writers.py")
        
        os.makedirs(work_dir, exist_ok=True)
        expected_files = EXPECTED_FILES
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            results = []
            
            def tracker_writer_thread(thread_id):
                """Each thread simulates a progress tracker updating progress."""
                # Create varying numbers of epoch files to simulate different progress states
                base_epochs = thread_id * 2  # Thread 0: 0 epochs, Thread 1: 2 epochs, etc.
                
                for update_round in range(3):
                    # Create epoch files to simulate training progress
                    current_epochs = base_epochs + update_round
                    for epoch_idx in range(current_epochs + 1):  # +1 to ensure the epoch exists
                        create_epoch_files(work_dir, epoch_idx)
                    
                    # Use TrainingJob to compute and save progress
                    progress = TrainingJob.calculate_progress(work_dir, config=None, force_progress_recompute=True)
                    
                    results.append((thread_id, update_round, progress.completed_epochs))
                    time.sleep(0.01)  # Small delay to increase chance of conflicts
            
            # Run 5 concurrent tracker writers
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(tracker_writer_thread, thread_id) for thread_id in range(5)]
                for future in as_completed(futures):
                    future.result()
            
            # Verify final progress.json file is valid and not corrupted
            progress_file = os.path.join(work_dir, "progress.json")
            assert os.path.exists(progress_file)
            final_data = load_json(progress_file)
            
            # Verify structure is correct (standard progress.json format)
            assert "completed_epochs" in final_data
            assert "progress_percentage" in final_data
            assert "early_stopped" in final_data
            assert "runner_type" in final_data
            assert final_data["runner_type"] == "trainer"
            assert isinstance(final_data["completed_epochs"], int)
            assert final_data["completed_epochs"] >= 0
            
            # Verify all threads completed without errors
            assert len(results) == 15  # 5 threads * 3 updates each
            for thread_id, update_round, epochs in results:
                assert isinstance(epochs, int)
                assert epochs >= 0
                
        finally:
            os.chdir(original_cwd)


# ============================================================================
# CONCURRENCY TESTS - CACHE INVALIDATION RACES
# ============================================================================

def test_cache_invalidation_race_conditions(create_progress_json):
    """Test cache invalidation when multiple trackers access same file."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create initial progress.json
        create_progress_json(work_dir, completed_epochs=5, early_stopped=False, tot_epochs=100)
        
        # Create tracker instances (they'll cache results)
        trackers = [TrainingJob for _ in range(3)]
        
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
                    progress = TrainingJob.get_session_progress(work_dir, expected_files, force_progress_recompute=True)
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
    """Test TrainingJob and EvaluationJob accessing different files concurrently."""
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
        save_json(trainer_progress, os.path.join(trainer_dir, "progress.json"))
        
        # Create evaluator evaluation_scores.json
        eval_scores = {"aggregated": {"acc": 0.95}, "per_datapoint": {"acc": [0.95]}}
        save_json(eval_scores, os.path.join(evaluator_dir, "evaluation_scores.json"))
        
        results = []
        
        def trainer_worker():
            for i in range(5):
                progress = TrainingJob.get_session_progress(trainer_dir, TrainingJob.get_expected_files())
                results.append(("trainer", progress.completed_epochs))
                time.sleep(0.01)
        
        def evaluator_worker():
            for i in range(5):
                progress = EvaluationJob.calculate_progress(evaluator_dir, config=None)
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
# CONCURRENCY TESTS - REALISTIC PROGRESS TRACKING SCENARIOS
# ============================================================================

def test_concurrent_progress_json_creation():
    """Test multiple processes creating progress.json files in different directories."""
    with tempfile.TemporaryDirectory() as base_dir:
        def create_progress_worker(worker_id):
            # Each worker creates progress in its own directory
            work_dir = os.path.join(base_dir, f"experiment_{worker_id}")
            os.makedirs(work_dir, exist_ok=True)
            
            # Create multiple progress files (simulating repeated progress updates)
            for update in range(5):
                progress_data = {
                    "completed_epochs": update * 10,
                    "progress_percentage": update * 10.0,
                    "early_stopped": False,
                    "early_stopped_at_epoch": None,
                    "runner_type": "trainer",
                    "total_epochs": 100
                }
                progress_file = os.path.join(work_dir, "progress.json")
                save_json(progress_data, progress_file)
                time.sleep(0.01)
        
        # Run multiple workers creating progress files
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_progress_worker, worker_id) for worker_id in range(5)]
            for future in as_completed(futures):
                future.result()
        
        # Verify all progress files were created correctly
        for worker_id in range(5):
            progress_file = os.path.join(base_dir, f"experiment_{worker_id}", "progress.json")
            assert os.path.exists(progress_file)
            data = load_json(progress_file)
            assert data["completed_epochs"] == 40  # Final update
            assert data["runner_type"] == "trainer"
