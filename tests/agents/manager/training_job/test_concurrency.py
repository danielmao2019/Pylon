"""
Test progress tracking under concurrent access scenarios.

These tests validate that the progress system handles multiple
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
from agents.manager.progress_info import ProgressInfo
from utils.io.json import load_json, save_json


# ============================================================================
# CONCURRENCY TESTS - MULTIPLE READERS
# ============================================================================

def test_multiple_readers_same_progress_file(create_progress_json):
    """Test multiple threads reading the same progress.json file simultaneously."""
    with tempfile.TemporaryDirectory() as root:
        logs_dir = os.path.join(root, 'logs')
        configs_dir = os.path.join(root, 'configs')
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        work_dir = os.path.join(logs_dir, 'exp_readers')
        os.makedirs(work_dir, exist_ok=True)
        create_progress_json(work_dir, completed_epochs=42, early_stopped=False, tot_epochs=100)

        config_path = os.path.join(configs_dir, 'exp_readers.py')
        with open(config_path, 'w') as f:
            f.write('config = {"epochs": 100, "work_dir": "' + work_dir.replace('\\', '/') + '"}\n')

        def read_progress():
            cwd = os.getcwd()
            os.chdir(root)
            try:
                job = TrainingJob(config_path)
                return job.get_progress()
            finally:
                os.chdir(cwd)

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_progress) for _ in range(10)]
            for future in as_completed(futures):
                results.append(future.result())

        assert len(results) == 10
        assert all(r.completed_epochs == 42 for r in results)
        assert all(r.progress_percentage == 42.0 for r in results)
        assert all(r.early_stopped is False for r in results)


def test_multiple_readers_via_get_progress(create_progress_json):
    """Test multiple TrainingJob instances reading same progress.json."""
    with tempfile.TemporaryDirectory() as root:
        logs_dir = os.path.join(root, 'logs')
        configs_dir = os.path.join(root, 'configs')
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        work_dir = os.path.join(logs_dir, 'exp_calc')
        os.makedirs(work_dir, exist_ok=True)
        create_progress_json(work_dir, completed_epochs=25, early_stopped=False, tot_epochs=100)

        config_path = os.path.join(configs_dir, 'exp_calc.py')
        with open(config_path, 'w') as f:
            f.write('config = {"epochs": 100, "work_dir": "' + work_dir.replace('\\', '/') + '"}\n')

        def read_with_calc():
            cwd = os.getcwd()
            os.chdir(root)
            try:
                job = TrainingJob(config_path)
                return job.get_progress()
            finally:
                os.chdir(cwd)

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_with_calc) for _ in range(8)]
            for future in as_completed(futures):
                results.append(future.result())

        assert len(results) == 8
        assert all(r.completed_epochs == 25 for r in results)
        assert all(r.runner_type == 'trainer' for r in results)


# ============================================================================
# CONCURRENCY TESTS - READER-WRITER CONFLICTS
# ============================================================================

def test_reader_writer_conflict_resolution(create_progress_json):
    """Test readers and writers accessing progress.json simultaneously."""
    with tempfile.TemporaryDirectory() as root:
        logs_dir = os.path.join(root, 'logs')
        configs_dir = os.path.join(root, 'configs')
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        work_dir = os.path.join(logs_dir, 'exp_rw')
        os.makedirs(work_dir, exist_ok=True)

        create_progress_json(work_dir, completed_epochs=10, early_stopped=False, tot_epochs=100)

        config_path = os.path.join(configs_dir, 'exp_rw.py')
        with open(config_path, 'w') as f:
            f.write('config = {"epochs": 100, "work_dir": "' + work_dir.replace('\\', '/') + '"}\n')

        results = {"reads": [], "writes": []}

        def compute_progress() -> ProgressInfo:
            cwd = os.getcwd()
            os.chdir(root)
            try:
                job = TrainingJob(config_path)
                return job.get_progress()
            finally:
                os.chdir(cwd)

        def reader_thread():
            for _ in range(5):
                try:
                    progress = compute_progress()
                    results["reads"].append(progress.completed_epochs)
                    time.sleep(0.01)
                except Exception as e:
                    results["reads"].append(f"ERROR: {e}")

        def writer_thread():
            for i in range(3):
                try:
                    new_epochs = 20 + i * 10
                    create_progress_json(work_dir, completed_epochs=new_epochs, early_stopped=False, tot_epochs=100)
                    results["writes"].append(new_epochs)
                    time.sleep(0.02)
                except Exception as e:
                    results["writes"].append(f"ERROR: {e}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(reader_thread),
                executor.submit(reader_thread),
                executor.submit(writer_thread),
                executor.submit(writer_thread),
            ]
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
            rel_config = "./configs/test_concurrent_writers.py"
            
            def tracker_writer_thread(thread_id):
                """Each thread simulates a progress updater writing progress.json."""
                # Create varying numbers of epoch files to simulate different progress states
                base_epochs = thread_id * 2  # Thread 0: 0 epochs, Thread 1: 2 epochs, etc.
                
                for update_round in range(3):
                    # Create epoch files to simulate training progress
                    current_epochs = base_epochs + update_round
                    for epoch_idx in range(current_epochs + 1):  # +1 to ensure the epoch exists
                        create_epoch_files(work_dir, epoch_idx)
                    
                    # Use TrainingJob to compute and save progress
                    job = TrainingJob(rel_config)
                    progress = job.get_progress(force_progress_recompute=True)
                    
                    results.append((thread_id, update_round, progress.completed_epochs))
                    time.sleep(0.01)  # Small delay to increase chance of conflicts
            
            # Run 5 concurrent writers
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

def test_cache_invalidation_race_conditions(create_progress_json, create_real_config):
    """Test cache invalidation when multiple trackers access same file."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "cache_invalidation")
        config_path = os.path.join(configs_dir, "cache_invalidation.py")

        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        create_progress_json(work_dir, completed_epochs=5, early_stopped=False, tot_epochs=100)
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)

        results = []
        rel_config = "./configs/cache_invalidation.py"

        def cache_and_read(tracker_id):
            cwd = os.getcwd()
            os.chdir(temp_root)
            try:
                job = TrainingJob(rel_config)
                progress1 = job.get_progress()
                results.append(("first_read", tracker_id, progress1.completed_epochs))

                time.sleep(0.05)

                progress2 = job.get_progress()
                results.append(("second_read", tracker_id, progress2.completed_epochs))
            finally:
                os.chdir(cwd)

        def file_updater():
            time.sleep(0.02)
            create_progress_json(work_dir, completed_epochs=15, early_stopped=False, tot_epochs=100)
            time.sleep(0.02)
            create_progress_json(work_dir, completed_epochs=25, early_stopped=False, tot_epochs=100)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cache_and_read, i) for i in range(3)]
            futures.append(executor.submit(file_updater))
            for future in as_completed(futures):
                future.result()

        assert len(results) > 0

        tracker_results = {}
        for operation, tracker_id, epochs in results:
            tracker_results.setdefault(tracker_id, []).append((operation, epochs))

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
            rel_config = "./configs/test_concurrent_force.py"

            def force_recompute_thread(thread_id):
                cwd = os.getcwd()
                os.chdir(temp_root)
                try:
                    job = TrainingJob(rel_config)
                    for i in range(3):
                        progress = job.get_progress(force_progress_recompute=True)
                        results.append((thread_id, i, progress.completed_epochs))
                        time.sleep(0.01)  # Small delay to increase interleaving
                finally:
                    os.chdir(cwd)
            
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
# CONCURRENCY TESTS - MIXED RUNNER TYPES
# ============================================================================

def test_mixed_runner_types_concurrent_access():
    """Test TrainingJob and EvaluationJob accessing different files concurrently."""
    with tempfile.TemporaryDirectory() as base_dir:
        logs_dir = os.path.join(base_dir, "logs")
        configs_dir = os.path.join(base_dir, "configs")
        trainer_dir = os.path.join(logs_dir, "trainer")
        evaluator_dir = os.path.join(logs_dir, "evaluator")
        os.makedirs(trainer_dir)
        os.makedirs(evaluator_dir)
        os.makedirs(configs_dir)

        # Write minimal config files so BaseJob constructor succeeds
        trainer_config_path = os.path.join(configs_dir, "trainer.py")
        evaluator_config_path = os.path.join(configs_dir, "evaluator.py")
        with open(trainer_config_path, 'w', encoding='utf-8') as f:
            f.write("config = {'epochs': 100}\n")
        with open(evaluator_config_path, 'w', encoding='utf-8') as f:
            f.write("config = {}\n")

        # Create trainer progress.json
        trainer_progress = {
            "completed_epochs": 50,
            "progress_percentage": 50.0,
            "early_stopped": False,
            "early_stopped_at_epoch": None,
            "runner_type": "trainer",
            "total_epochs": 100,
        }
        save_json(trainer_progress, os.path.join(trainer_dir, "progress.json"))

        # Create evaluator evaluation_scores.json
        eval_scores = {"aggregated": {"acc": 0.95}, "per_datapoint": {"acc": [0.95]}}
        save_json(eval_scores, os.path.join(evaluator_dir, "evaluation_scores.json"))

        def trainer_worker():
            cwd = os.getcwd()
            os.chdir(base_dir)
            try:
                job = TrainingJob("./configs/trainer.py")
                for _ in range(5):
                    progress = job.get_progress()
                    results.append(("trainer", progress.completed_epochs))
                    time.sleep(0.01)
            finally:
                os.chdir(cwd)

        def evaluator_worker():
            cwd = os.getcwd()
            os.chdir(base_dir)
            try:
                job = EvaluationJob("./configs/evaluator.py")
                for _ in range(5):
                    progress = job.get_progress()
                    results.append(("evaluator", progress.completed_epochs))
                    time.sleep(0.01)
            finally:
                os.chdir(cwd)

        results = []

        # Run concurrent trainer and evaluator operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(trainer_worker),
                executor.submit(trainer_worker),
                executor.submit(evaluator_worker),
                executor.submit(evaluator_worker),
            ]
            for future in as_completed(futures):
                future.result()

        # Verify results
        trainer_results = [epochs for kind, epochs in results if kind == "trainer"]
        evaluator_results = [epochs for kind, epochs in results if kind == "evaluator"]

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
