"""The purpose of this set of test cases is to compare training performance against other frameworks,
including the native PyTorch for loop.
"""
from typing import Tuple, List, Dict
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
import os
import threading
import time
import json
import torch
import torchvision

import utils
from utils.automation.run_status import check_epoch_finished
from utils.ops import buffer_allclose
from configs.examples.linear.config import config
config['work_dir'] = "./logs/tests/supervised_single_task_trainer/interrupt_and_resume"


def test_interrupt_and_resume() -> None:
    """Test that training can be interrupted and resumed from a checkpoint.

    This test:
    1. Starts training in main thread
    2. Uses observer thread to monitor epoch 3 completion
    3. Interrupts training when epoch 3 is done
    4. Creates a new trainer to resume
    5. Interrupts again at epoch 6
    6. Runs uninterrupted training up to epoch 6
    7. Verifies files match between interrupted and uninterrupted training
    """
    if os.path.isdir(config['work_dir']):
        os.system(' '.join(["rm", "-r", config['work_dir']]))

    # Create first trainer
    trainer1 = SupervisedSingleTaskTrainer(config=config)

    # Create an event to signal the observer thread to stop
    stop_event = threading.Event()
    # Create a flag to signal training interruption
    interrupt_flag = threading.Event()

    def observer_thread():
        """Monitor training progress and interrupt after epoch 3."""
        while not stop_event.is_set():
            # Check if epoch 3 is complete
            epoch_dir = os.path.join(config['work_dir'], "epoch_2")  # 0-based indexing
            if os.path.exists(epoch_dir) and check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=trainer1.expected_files,
            ):
                # Set interrupt flag immediately
                interrupt_flag.set()
                break
            time.sleep(0.1)  # Check every 100ms

    # Start observer thread
    observer = threading.Thread(target=observer_thread)
    observer.start()

    # Start training in main thread
    trainer1._init_components_()
    start_epoch = trainer1.cum_epochs
    trainer1.logger.page_break()
    # Run until interrupted
    for idx in range(start_epoch, trainer1.tot_epochs):
        if interrupt_flag.is_set():
            break
        utils.determinism.set_seed(seed=trainer1.train_seeds[idx])
        trainer1._train_epoch_()
        trainer1._val_epoch_()
        trainer1.logger.page_break()
        trainer1.cum_epochs = idx + 1
        time.sleep(3)  # allow some more time for interrupt_flag to be set

    # Signal observer thread to stop
    stop_event.set()
    observer.join()
    # Clean up trainer1
    del trainer1

    # Second run - should resume from epoch 3
    trainer2 = SupervisedSingleTaskTrainer(config=config)
    trainer2._init_components_()
    # Verify that trainer2 is resuming from epoch 3
    assert trainer2.cum_epochs == 3, f"Expected to resume from epoch 3, but got {trainer2.cum_epochs}"
    print(f"Successfully resumed training from epoch {trainer2.cum_epochs}")

    # Create new event for second interruption
    stop_event = threading.Event()
    interrupt_flag = threading.Event()

    def observer_thread_2():
        """Monitor training progress and interrupt after epoch 6."""
        while not stop_event.is_set():
            # Check if epoch 6 is complete
            epoch_dir = os.path.join(config['work_dir'], "epoch_5")  # 0-based indexing
            if os.path.exists(epoch_dir) and check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=trainer2.expected_files,
            ):
                # Set interrupt flag immediately
                interrupt_flag.set()
                break
            time.sleep(0.1)  # Check every 100ms

    # Start observer thread
    observer = threading.Thread(target=observer_thread_2)
    observer.start()

    # Continue training in main thread
    start_epoch = trainer2.cum_epochs
    trainer2.logger.page_break()
    # Run until interrupted
    for idx in range(start_epoch, trainer2.tot_epochs):
        if interrupt_flag.is_set():
            break
        utils.determinism.set_seed(seed=trainer2.train_seeds[idx])
        trainer2._train_epoch_()
        trainer2._val_epoch_()
        trainer2.logger.page_break()
        trainer2.cum_epochs = idx + 1
        time.sleep(3)  # allow some more time for interrupt_flag to be set

    # Signal observer thread to stop
    stop_event.set()
    observer.join()
    # Clean up trainer2
    del trainer2

    # Create a new work directory for uninterrupted training
    uninterrupted_dir = config['work_dir'] + "_uninterrupted"
    if os.path.isdir(uninterrupted_dir):
        os.system(' '.join(["rm", "-r", uninterrupted_dir]))
    os.makedirs(uninterrupted_dir)

    # Create config for uninterrupted training
    uninterrupted_config = config.copy()
    uninterrupted_config['work_dir'] = uninterrupted_dir

    # Run uninterrupted training
    trainer3 = SupervisedSingleTaskTrainer(config=uninterrupted_config)

    # Create new event for third interruption
    stop_event = threading.Event()
    interrupt_flag = threading.Event()

    def observer_thread_3():
        """Monitor training progress and interrupt after epoch 6."""
        while not stop_event.is_set():
            # Check if epoch 6 is complete
            epoch_dir = os.path.join(uninterrupted_dir, "epoch_5")  # 0-based indexing
            if os.path.exists(epoch_dir) and check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=trainer3.expected_files,
            ):
                # Set interrupt flag immediately
                interrupt_flag.set()
                break
            time.sleep(0.1)  # Check every 100ms

    # Start observer thread
    observer = threading.Thread(target=observer_thread_3)
    observer.start()

    # Run training in main thread
    trainer3._init_components_()
    start_epoch = trainer3.cum_epochs
    trainer3.logger.page_break()
    # Run until interrupted
    for idx in range(start_epoch, trainer3.tot_epochs):
        if interrupt_flag.is_set():
            break
        utils.determinism.set_seed(seed=trainer3.train_seeds[idx])
        trainer3._train_epoch_()
        trainer3._val_epoch_()
        trainer3.logger.page_break()
        trainer3.cum_epochs = idx + 1
        time.sleep(3)  # allow some more time for interrupt_flag to be set

    # Signal observer thread to stop
    stop_event.set()
    observer.join()
    # Clean up trainer3
    del trainer3

    # Compare files between interrupted and uninterrupted training
    test_interrupt_and_resume_compare()


def test_interrupt_and_resume_compare() -> None:
    # Compare files between interrupted and uninterrupted training
    for epoch in range(6):
        interrupted_epoch_dir = os.path.join(config['work_dir'], f"epoch_{epoch}")
        uninterrupted_epoch_dir = os.path.join(config['work_dir'] + "_uninterrupted", f"epoch_{epoch}")

        # Compare training losses
        interrupted_losses = torch.load(os.path.join(interrupted_epoch_dir, "training_losses.pt"))
        uninterrupted_losses = torch.load(os.path.join(uninterrupted_epoch_dir, "training_losses.pt"))
        assert torch.allclose(interrupted_losses, uninterrupted_losses, rtol=1e-01, atol=0), f"Training losses mismatch at epoch {epoch}"

        # Compare validation scores
        with open(os.path.join(interrupted_epoch_dir, "validation_scores.json")) as f:
            interrupted_scores = json.load(f)
        with open(os.path.join(uninterrupted_epoch_dir, "validation_scores.json")) as f:
            uninterrupted_scores = json.load(f)
        assert buffer_allclose(interrupted_scores, uninterrupted_scores, rtol=1e-01, atol=0), f"Validation scores mismatch at epoch {epoch}"

        print(f"Epoch {epoch} files match between interrupted and uninterrupted training")
