"""The purpose of this set of test cases is to compare training performance against other frameworks,
including the native PyTorch for loop.
"""
import os
import threading
import time
import json
import torch

from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
import utils
from utils.automation.run_status import check_epoch_finished
from utils.ops import buffer_allclose
from configs.examples.linear.config import config
config['work_dir'] = "./logs/tests/supervised_single_task_trainer/interrupt_and_resume"


def train_until_epoch(config: dict, start_epoch: int, end_epoch: int) -> None:
    """Helper function to train from start_epoch until end_epoch.
    
    Args:
        config: Training configuration
        start_epoch: Epoch to start training from
        end_epoch: Epoch to train until (exclusive)
    """
    trainer = SupervisedSingleTaskTrainer(config=config)
    trainer._init_components_()
    
    # Verify we're starting from the correct epoch
    assert trainer.cum_epochs == start_epoch, f"Expected to start from epoch {start_epoch}, but got {trainer.cum_epochs}"
    print(f"Starting training from epoch {trainer.cum_epochs}")

    # Create an event to signal the observer thread to stop
    stop_event = threading.Event()
    # Create a flag to signal training interruption
    interrupt_flag = threading.Event()

    def observer_thread():
        """Monitor training progress and interrupt after target epoch."""
        while not stop_event.is_set():
            # Check if target epoch is complete
            epoch_dir = os.path.join(config['work_dir'], f"epoch_{end_epoch-1}")  # 0-based indexing
            if os.path.exists(epoch_dir) and check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=trainer.expected_files,
            ):
                # Set interrupt flag immediately
                interrupt_flag.set()
                break
            time.sleep(0.1)  # Check every 100ms

    # Start observer thread
    observer = threading.Thread(target=observer_thread)
    observer.start()

    # Run training in main thread
    trainer.logger.page_break()
    # Run until interrupted
    for idx in range(start_epoch, trainer.tot_epochs):
        if interrupt_flag.is_set():
            break
        utils.determinism.set_seed(seed=trainer.train_seeds[idx])
        trainer._train_epoch_()
        trainer._val_epoch_()
        trainer.logger.page_break()
        trainer.cum_epochs = idx + 1
        time.sleep(3)  # allow some more time for interrupt_flag to be set

    # Signal observer thread to stop
    stop_event.set()
    observer.join()
    # Clean up trainer
    del trainer


def test_interrupt_and_resume() -> None:
    """Test that training can be interrupted and resumed from a checkpoint.

    This test:
    1. Starts training in main thread and runs until epoch 3
    2. Creates a new trainer to resume from epoch 3
    3. Runs until epoch 6
    4. Verifies files match against reference run
    """
    os.system(f"rm -rf {config['work_dir']}")

    # First run: train from epoch 0 to 3
    train_until_epoch(config, start_epoch=0, end_epoch=3)

    # Second run: train from epoch 3 to 6
    train_until_epoch(config, start_epoch=3, end_epoch=6)

    # Compare files against reference run
    reference_dir = "logs/examples/linear"
    for epoch in range(6):
        interrupted_epoch_dir = os.path.join(config['work_dir'], f"epoch_{epoch}")
        reference_epoch_dir = os.path.join(reference_dir, f"epoch_{epoch}")

        # Compare training losses
        interrupted_losses = torch.load(os.path.join(interrupted_epoch_dir, "training_losses.pt"))
        reference_losses = torch.load(os.path.join(reference_epoch_dir, "training_losses.pt"))
        assert torch.allclose(interrupted_losses, reference_losses, rtol=1e-01, atol=0), \
            f"Training losses mismatch at epoch {epoch}"

        # Compare validation scores
        with open(os.path.join(interrupted_epoch_dir, "validation_scores.json")) as f:
            interrupted_scores = json.load(f)
        with open(os.path.join(reference_epoch_dir, "validation_scores.json")) as f:
            reference_scores = json.load(f)
        assert buffer_allclose(interrupted_scores, reference_scores, rtol=1e-01, atol=0), \
            f"Validation scores mismatch at epoch {epoch}"

        print(f"Epoch {epoch} files match between interrupted and reference training")
