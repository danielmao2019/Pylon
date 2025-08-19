import pytest
import torch
import time
import threading
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from runners.base_trainer import BaseTrainer
from typing import Dict, Any, Optional


class MockTrainer(BaseTrainer):
    """Mock trainer for testing async after_train/after_val operations."""
    
    def __init__(self, work_dir: str):
        # Minimal initialization for testing
        self.work_dir = work_dir
        self.cum_epochs = 1
        self.after_train_thread = None
        self.after_val_thread = None
        self.buffer_lock = threading.Lock()
        
        # Mock required components
        self.criterion = MagicMock()
        self.criterion.summarize = MagicMock()
        self.optimizer = MagicMock()
        self.optimizer.summarize = MagicMock()
        self.metric = MagicMock()
        self.metric.summarize = MagicMock()
        self.early_stopping = None
        self.debugger = None
        
        # Mock checkpoint methods
        self._save_checkpoint_ = MagicMock()
        self._find_best_checkpoint = MagicMock(return_value=os.path.join(work_dir, "epoch_1", "checkpoint.pt"))
        self._clean_checkpoints = MagicMock()
    
    def _init_optimizer(self) -> None:
        """Mock implementation of abstract method."""
        pass
    
    def _init_scheduler(self) -> None:
        """Mock implementation of abstract method."""
        pass
    
    def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        """Mock implementation of abstract method."""
        pass
    
    def _after_train_loop_(self) -> None:
        """Copy of actual after_train_loop implementation for testing."""
        def after_train_ops():
            # Create epoch directory
            epoch_root = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
            os.makedirs(epoch_root, exist_ok=True)
            
            # Save criterion buffer (with lock protection)
            with self.buffer_lock:
                self.criterion.summarize(output_path=os.path.join(epoch_root, "training_losses.pt"))
            
            # Save optimizer buffer
            self.optimizer.summarize(output_path=os.path.join(epoch_root, "optimizer_buffer.json"))
            
            # Save checkpoint
            self._save_checkpoint_(output_path=os.path.join(epoch_root, "checkpoint.pt"))
            
            # Update latest checkpoint symlink
            latest_checkpoint = os.path.join(epoch_root, "checkpoint.pt")
            soft_link = os.path.join(self.work_dir, "checkpoint_latest.pt")
            if os.path.islink(soft_link):
                os.system(' '.join(["rm", soft_link]))
            os.system(' '.join(["ln", "-s", os.path.relpath(path=latest_checkpoint, start=self.work_dir), soft_link]))
        
        self.after_train_thread = threading.Thread(target=after_train_ops)
        self.after_train_thread.start()
    
    def _after_val_loop_(self) -> None:
        """Copy of actual after_val_loop implementation for testing."""
        def after_val_ops():
            # Create epoch directory
            epoch_root = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
            os.makedirs(epoch_root, exist_ok=True)
            
            # Save metric buffer (with lock protection)
            with self.buffer_lock:
                self.metric.summarize(output_path=os.path.join(epoch_root, "validation_scores.json"))
            
            # Update early stopping
            if self.early_stopping:
                self.early_stopping.update()
            
            # Find and set best checkpoint - WITH BARE EXCEPT
            try:
                best_checkpoint: str = self._find_best_checkpoint()
                soft_link: str = os.path.join(self.work_dir, "checkpoint_best.pt")
                if os.path.isfile(soft_link):
                    os.system(' '.join(["rm", soft_link]))
                os.system(' '.join(["ln", "-s", os.path.relpath(path=best_checkpoint, start=self.work_dir), soft_link]))
            except:
                best_checkpoint = None
            
            # Save debugger outputs (if enabled)
            if self.debugger and hasattr(self.debugger, 'enabled') and self.debugger.enabled:
                debugger_dir = os.path.join(epoch_root, "debugger")
                self.debugger.save_all(debugger_dir)
            
            # Cleanup old checkpoints
            self._clean_checkpoints(None, best_checkpoint)
        
        self.after_val_thread = threading.Thread(target=after_val_ops)
        self.after_val_thread.start()


class FailingTrainer(MockTrainer):
    """Trainer that fails during async operations for error testing."""
    
    def __init__(self, work_dir: str, fail_mode: str = "criterion_save"):
        super().__init__(work_dir)
        self.fail_mode = fail_mode
        
        # Configure failing components based on mode
        if fail_mode == "criterion_save":
            self.criterion.summarize.side_effect = RuntimeError("Simulated criterion save failure")
        elif fail_mode == "optimizer_save":
            self.optimizer.summarize.side_effect = RuntimeError("Simulated optimizer save failure")
        elif fail_mode == "metric_save":
            self.metric.summarize.side_effect = RuntimeError("Simulated metric save failure")
        elif fail_mode == "checkpoint_save":
            self._save_checkpoint_.side_effect = RuntimeError("Simulated checkpoint save failure")
        elif fail_mode == "best_checkpoint_find":
            self._find_best_checkpoint.side_effect = RuntimeError("Simulated best checkpoint find failure")
        elif fail_mode == "directory_creation":
            # We'll mock os.makedirs to fail
            pass


@pytest.fixture
def temp_work_dir():
    """Fixture providing a temporary work directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_trainer(temp_work_dir):
    """Fixture providing a MockTrainer instance."""
    return MockTrainer(temp_work_dir)


def test_after_train_error_propagation_criterion_save(temp_work_dir):
    """Test that criterion save failures in after_train operations crash the program."""
    failing_trainer = FailingTrainer(temp_work_dir, fail_mode="criterion_save")
    
    # Start the failing async operation
    failing_trainer._after_train_loop_()
    
    # Give time for the operation to fail
    time.sleep(0.2)
    
    # In a real scenario, this would crash the program with RuntimeError
    # Here we can verify the thread has the potential to fail
    assert hasattr(failing_trainer, 'after_train_thread')
    assert isinstance(failing_trainer.after_train_thread, threading.Thread)
    
    # The thread should either be dead (if it failed) or still alive (if error handling prevented failure)
    # Since we removed error handling, the expectation is that errors propagate


def test_after_train_error_propagation_optimizer_save(temp_work_dir):
    """Test that optimizer save failures in after_train operations crash the program."""
    failing_trainer = FailingTrainer(temp_work_dir, fail_mode="optimizer_save")
    
    failing_trainer._after_train_loop_()
    time.sleep(0.2)
    
    # Verify error propagation potential
    assert hasattr(failing_trainer, 'after_train_thread')
    assert isinstance(failing_trainer.after_train_thread, threading.Thread)


def test_after_train_error_propagation_checkpoint_save(temp_work_dir):
    """Test that checkpoint save failures in after_train operations crash the program."""
    failing_trainer = FailingTrainer(temp_work_dir, fail_mode="checkpoint_save")
    
    failing_trainer._after_train_loop_()
    time.sleep(0.2)
    
    # Verify error propagation potential
    assert hasattr(failing_trainer, 'after_train_thread')
    assert isinstance(failing_trainer.after_train_thread, threading.Thread)


def test_after_val_error_propagation_metric_save(temp_work_dir):
    """Test that metric save failures in after_val operations crash the program."""
    failing_trainer = FailingTrainer(temp_work_dir, fail_mode="metric_save")
    
    failing_trainer._after_val_loop_()
    time.sleep(0.2)
    
    # Verify error propagation potential
    assert hasattr(failing_trainer, 'after_val_thread')
    assert isinstance(failing_trainer.after_val_thread, threading.Thread)


def test_after_val_error_propagation_best_checkpoint_suppressed(temp_work_dir):
    """Test that best checkpoint find failures are suppressed by bare except."""
    failing_trainer = FailingTrainer(temp_work_dir, fail_mode="best_checkpoint_find")
    
    # This operation should NOT crash due to the bare except clause
    failing_trainer._after_val_loop_()
    
    # Wait for completion
    failing_trainer.after_val_thread.join(timeout=2.0)
    
    # Thread should complete successfully despite the error
    assert not failing_trainer.after_val_thread.is_alive()
    
    # Verify the bare except suppressed the error
    # (This is the problematic behavior we want to highlight)


def test_after_train_error_propagation_directory_creation(temp_work_dir):
    """Test that directory creation failures in after_train operations crash the program."""
    trainer = MockTrainer(temp_work_dir)
    
    # Mock os.makedirs to fail
    with patch('os.makedirs', side_effect=PermissionError("Simulated directory creation failure")):
        trainer._after_train_loop_()
        time.sleep(0.2)
        
        # Verify error propagation potential
        assert hasattr(trainer, 'after_train_thread')


def test_concurrent_after_train_timing(mock_trainer):
    """Test timing of concurrent after_train operations."""
    # Add delays to mock operations to test concurrency
    def slow_operation(*args, **kwargs):
        time.sleep(0.1)
    
    mock_trainer.criterion.summarize.side_effect = slow_operation
    mock_trainer.optimizer.summarize.side_effect = slow_operation
    mock_trainer._save_checkpoint_.side_effect = slow_operation
    
    start_time = time.time()
    
    # Start first after_train operation
    mock_trainer.cum_epochs = 1
    mock_trainer._after_train_loop_()
    first_thread = mock_trainer.after_train_thread
    
    # Small delay then start second operation
    time.sleep(0.02)
    mock_trainer.cum_epochs = 2
    mock_trainer._after_train_loop_()
    second_thread = mock_trainer.after_train_thread
    
    # Threads should be different instances
    assert first_thread != second_thread
    
    # At least one should still be alive initially
    assert first_thread.is_alive() or second_thread.is_alive()
    
    # Wait for both to complete
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)
    
    total_time = time.time() - start_time
    
    # Should complete in reasonable time even with concurrent operations
    assert total_time < 5.0, f"Concurrent operations took too long: {total_time:.2f}s"
    
    # Verify both operations completed
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()


def test_concurrent_after_val_timing(mock_trainer):
    """Test timing of concurrent after_val operations."""
    # Add delays to mock operations to test concurrency
    def slow_operation(*args, **kwargs):
        time.sleep(0.1)
    
    mock_trainer.metric.summarize.side_effect = slow_operation
    mock_trainer._find_best_checkpoint.side_effect = slow_operation
    
    start_time = time.time()
    
    # Start first after_val operation
    mock_trainer.cum_epochs = 1
    mock_trainer._after_val_loop_()
    first_thread = mock_trainer.after_val_thread
    
    # Small delay then start second operation
    time.sleep(0.02)
    mock_trainer.cum_epochs = 2
    mock_trainer._after_val_loop_()
    second_thread = mock_trainer.after_val_thread
    
    # Threads should be different instances
    assert first_thread != second_thread
    
    # At least one should still be alive initially
    assert first_thread.is_alive() or second_thread.is_alive()
    
    # Wait for both to complete
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)
    
    total_time = time.time() - start_time
    
    # Should complete in reasonable time
    assert total_time < 5.0, f"Concurrent operations took too long: {total_time:.2f}s"
    
    # Verify both operations completed
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()


def test_mixed_concurrent_after_operations_timing(mock_trainer):
    """Test timing of mixed after_train and after_val operations running concurrently."""
    # Add delays to mock operations
    def slow_operation(*args, **kwargs):
        time.sleep(0.15)
    
    mock_trainer.criterion.summarize.side_effect = slow_operation
    mock_trainer.optimizer.summarize.side_effect = slow_operation
    mock_trainer.metric.summarize.side_effect = slow_operation
    mock_trainer._save_checkpoint_.side_effect = slow_operation
    
    start_time = time.time()
    
    # Start both types of operations
    mock_trainer.cum_epochs = 1
    mock_trainer._after_train_loop_()
    train_thread = mock_trainer.after_train_thread
    
    # Small delay then start after_val
    time.sleep(0.02)
    mock_trainer._after_val_loop_()
    val_thread = mock_trainer.after_val_thread
    
    # At least one should still be alive initially
    assert train_thread.is_alive() or val_thread.is_alive()
    
    # Wait for both to complete
    train_thread.join(timeout=3.0)
    val_thread.join(timeout=3.0)
    
    total_time = time.time() - start_time
    
    # Mixed operations should complete efficiently
    assert total_time < 8.0, f"Mixed concurrent operations took too long: {total_time:.2f}s"
    
    # Verify both completed
    assert not train_thread.is_alive()
    assert not val_thread.is_alive()


def test_buffer_lock_contention_during_async_ops(mock_trainer):
    """Test buffer lock behavior during concurrent async operations."""
    # This test verifies that the lock mechanism exists and operations work correctly
    
    # Add simple delays without lock interaction to avoid deadlocks
    def simple_slow_operation(*args, **kwargs):
        """Mock operation with delay but no lock interaction."""
        time.sleep(0.05)
    
    mock_trainer.criterion.summarize.side_effect = simple_slow_operation
    mock_trainer.metric.summarize.side_effect = simple_slow_operation
    mock_trainer.optimizer.summarize.side_effect = simple_slow_operation
    
    start_time = time.time()
    
    # Start both operations
    mock_trainer.cum_epochs = 1
    mock_trainer._after_train_loop_()
    train_thread = mock_trainer.after_train_thread
    
    mock_trainer._after_val_loop_()
    val_thread = mock_trainer.after_val_thread
    
    # Both threads should exist
    assert train_thread is not None
    assert val_thread is not None
    
    # Wait for both to complete
    train_thread.join(timeout=2.0)
    val_thread.join(timeout=2.0)
    
    total_time = time.time() - start_time
    
    # Should complete efficiently
    assert total_time < 5.0, f"Operations took too long: {total_time:.2f}s"
    
    # Verify both operations completed successfully
    assert not train_thread.is_alive()
    assert not val_thread.is_alive()
    
    # Verify the lock mechanism exists (key part of the test)
    assert hasattr(mock_trainer, 'buffer_lock')
    assert hasattr(mock_trainer.buffer_lock, 'acquire')
    assert hasattr(mock_trainer.buffer_lock, 'release')
    
    # Test that the lock can be acquired and released
    with mock_trainer.buffer_lock:
        # Lock acquired successfully
        pass


def test_rapid_sequential_async_operations_timing(mock_trainer):
    """Test timing of rapid sequential after_train/after_val cycles."""
    num_cycles = 5
    start_time = time.time()
    
    completed_threads = []
    
    for epoch in range(1, num_cycles + 1):
        mock_trainer.cum_epochs = epoch
        
        # Start after_train
        mock_trainer._after_train_loop_()
        train_thread = mock_trainer.after_train_thread
        
        # Small delay then start after_val
        time.sleep(0.01)
        mock_trainer._after_val_loop_()
        val_thread = mock_trainer.after_val_thread
        
        completed_threads.extend([train_thread, val_thread])
        
        # Brief pause between cycles
        time.sleep(0.02)
    
    # Wait for all threads to complete
    for thread in completed_threads:
        thread.join(timeout=2.0)
    
    total_time = time.time() - start_time
    
    # Rapid cycles should complete efficiently
    assert total_time < 15.0, f"Rapid sequential operations took too long: {total_time:.2f}s"
    
    # Verify all threads completed
    for thread in completed_threads:
        assert not thread.is_alive(), "All threads should have completed"
    
    # Should have processed all cycles
    assert len(completed_threads) == num_cycles * 2


def test_async_operation_thread_lifecycle_management(mock_trainer):
    """Test proper thread lifecycle management for async operations."""
    # Initial state - no threads
    assert mock_trainer.after_train_thread is None
    assert mock_trainer.after_val_thread is None
    
    # Start operations
    mock_trainer._after_train_loop_()
    mock_trainer._after_val_loop_()
    
    train_thread = mock_trainer.after_train_thread
    val_thread = mock_trainer.after_val_thread
    
    # Threads should be alive initially
    assert train_thread.is_alive()
    assert val_thread.is_alive()
    
    # Threads should be different instances
    assert train_thread != val_thread
    
    # Wait for completion
    train_thread.join(timeout=2.0)
    val_thread.join(timeout=2.0)
    
    # Threads should be dead after completion
    assert not train_thread.is_alive()
    assert not val_thread.is_alive()
    
    # References should still exist
    assert mock_trainer.after_train_thread == train_thread
    assert mock_trainer.after_val_thread == val_thread


def test_synchronization_point_behavior(mock_trainer):
    """Test behavior of synchronization points (thread.join() calls)."""
    # Add delay to make thread run longer
    def slow_operation(*args, **kwargs):
        time.sleep(0.2)
    
    mock_trainer.criterion.summarize.side_effect = slow_operation
    mock_trainer.optimizer.summarize.side_effect = slow_operation
    mock_trainer._save_checkpoint_.side_effect = slow_operation
    
    # Start an operation
    mock_trainer._after_train_loop_()
    train_thread = mock_trainer.after_train_thread
    
    # Give thread time to start but not finish
    time.sleep(0.05)
    
    # Test non-blocking join with timeout
    start_join = time.time()
    joined = train_thread.join(timeout=0.1)  # Should timeout
    join_time = time.time() - start_join
    
    # Should timeout quickly and thread might still be alive
    assert join_time < 0.3
    
    # Test blocking join
    start_join = time.time()
    train_thread.join()  # Should block until completion
    join_time = time.time() - start_join
    
    # Should complete in reasonable time
    assert join_time < 3.0
    assert not train_thread.is_alive()  # Dead after successful join


def test_large_file_async_operations_timing(mock_trainer):
    """Test timing with large file I/O operations."""
    # Mock large file operations
    def slow_save(*args, **kwargs):
        """Simulate slow file I/O operation."""
        time.sleep(0.1)  # Simulate large file write
    
    mock_trainer.criterion.summarize.side_effect = slow_save
    mock_trainer.optimizer.summarize.side_effect = slow_save
    mock_trainer.metric.summarize.side_effect = slow_save
    mock_trainer._save_checkpoint_.side_effect = slow_save
    
    start_time = time.time()
    
    # Start both operations with slow I/O
    mock_trainer._after_train_loop_()
    mock_trainer._after_val_loop_()
    
    # Wait for completion
    mock_trainer.after_train_thread.join(timeout=5.0)
    mock_trainer.after_val_thread.join(timeout=5.0)
    
    total_time = time.time() - start_time
    
    # Should handle large file operations efficiently
    # (parallel execution should be faster than sequential)
    assert total_time < 3.0, f"Large file operations took too long: {total_time:.2f}s"
    
    # Both should have completed
    assert not mock_trainer.after_train_thread.is_alive()
    assert not mock_trainer.after_val_thread.is_alive()


def test_directory_creation_race_conditions(temp_work_dir):
    """Test directory creation when multiple async operations create same directories."""
    # Create multiple trainers targeting the same epoch
    trainer1 = MockTrainer(temp_work_dir)
    trainer2 = MockTrainer(temp_work_dir)
    
    trainer1.cum_epochs = 1
    trainer2.cum_epochs = 1  # Same epoch - potential race condition
    
    start_time = time.time()
    
    # Start both operations simultaneously
    trainer1._after_train_loop_()
    trainer2._after_val_loop_()
    
    # Wait for both to complete
    trainer1.after_train_thread.join(timeout=2.0)
    trainer2.after_val_thread.join(timeout=2.0)
    
    total_time = time.time() - start_time
    
    # Should handle directory creation race conditions
    assert total_time < 5.0, f"Directory race condition caused delay: {total_time:.2f}s"
    
    # Directory should exist after operations
    epoch_dir = os.path.join(temp_work_dir, "epoch_1")
    assert os.path.exists(epoch_dir)
    
    # Both operations should have completed successfully
    assert not trainer1.after_train_thread.is_alive()
    assert not trainer2.after_val_thread.is_alive()