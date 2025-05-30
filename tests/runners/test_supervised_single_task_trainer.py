"""The purpose of this set of test cases is to compare training performance against other frameworks,
including the native PyTorch for loop.
"""
from typing import List, Dict
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
import os
import json
import torch
import torchvision
from data.datasets.random_datasets import BaseRandomDataset
from data.transforms import Compose
import criteria
import metrics
import optimizers
import schedulers
import utils
import threading
import time
from utils.automation.run_status import check_epoch_finished


torch.manual_seed(0)
gt = torch.rand(size=(10, 10), dtype=torch.float32)

dataset_config = {
    'class': BaseRandomDataset,
    'args': {
        'num_examples': 10,
        'initial_seed': 0,
        'gen_func_config': {
            'inputs': {
                'x': (
                    torch.rand,
                    {'size': (10,), 'dtype': torch.float32},
                ),
            },
            'labels': {
                'y': (
                    torch.randn,
                    {'size': (10,), 'mean': 0, 'std': 0.1, 'dtype': torch.float32},
                ),
            },
        },
        'transforms_cfg': {
            'class': Compose,
            'args': {
                'transforms': [
                    {
                        'op': lambda xy: gt @ xy[0] + xy[1],
                        'input_names': [('inputs', 'x'), ('labels', 'y')],
                        'output_names': [('labels', 'y')],
                    }
                ],
            },
        },
    },
}


class TestModel(torch.nn.Module):

    def __init__(self) -> None:
        super(TestModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=10, out_features=10)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(inputs, dict), f"{type(inputs)=}"
        assert inputs.keys() == {'x'}, f"{inputs.keys()=}"
        return self.linear(inputs['x'])


config = {
    'work_dir': "./logs/test_supervised_single_task_trainer",
    'init_seed': 0,
    'epochs': 100,
    'train_seeds': [0] * 100,
    # ==================================================
    # dataloaders
    # ==================================================
    'train_dataset': dataset_config,
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 8,
        },
    },
    'val_dataset': dataset_config,
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
        },
    },
    'test_dataset': dataset_config,
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
        },
    },
    # ==================================================
    # model
    # ==================================================
    'model': {
        'class': TestModel,
        'args': {},
    },
    'criterion': {
        'class': criteria.wrappers.PyTorchCriterionWrapper,
        'args': {
            'criterion': torch.nn.MSELoss(reduction='mean'),
        },
    },
    'metric': {
        'class': metrics.wrappers.PyTorchMetricWrapper,
        'args': {
            'metric': torch.nn.MSELoss(reduction='mean'),
        },
    },
    # ==================================================
    # optimizer
    # ==================================================
    'optimizer': {
        'class': optimizers.SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': torch.optim.SGD,
                'args': {
                    'lr': 1e-03,
                },
            },
        },
    },
    'scheduler': {
        'class': torch.optim.lr_scheduler.LambdaLR,
        'args': {
            'lr_lambda': {
                'class': schedulers.lr_lambdas.ConstantLambda,
                'args': {},
            },
        },
    },
}

def test_supervised_single_task_trainer() -> None:
    if os.path.isdir(config['work_dir']):
        os.system(' '.join(["rm", "-r", config['work_dir']]))
    trainer = SupervisedSingleTaskTrainer(config=config)
    trainer.train()
    acc: List[float] = []
    for idx in range(config['epochs']):
        epoch_dir = os.path.join(config['work_dir'], f"epoch_{idx}")
        with open(os.path.join(epoch_dir, "validation_scores.json")) as f:
            acc.append(json.load(f)['accuracy'])
    plt.figure()
    plt.plot(acc)
    plt.savefig(os.path.join(config['work_dir'], "acc_trainer.png"))


def test_pytorch() -> None:
    dataset = dataset_config['class'](**dataset_config['args'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=10)
    criterion = config['criterion']['class'](**config['criterion']['args'])
    metric = config['metric']['class'](**config['metric']['args'])
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-03)
    all_accuracies: List[float] = []
    for _ in range(config['epochs']):
        # train epoch
        model.train()
        for example in dataloader:
            outputs = model(example['inputs']['image'])
            loss = criterion(y_pred=outputs, y_true=example['labels']['target'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # val epoch
        model.eval()
        metric.reset_buffer()
        for example in dataloader:
            outputs = model(example['inputs']['image'])
            metric(y_pred=outputs, y_true=example['labels']['target'])
        all_accuracies.append(metric.summarize()['accuracy'])
    plt.figure()
    plt.plot(all_accuracies)
    plt.savefig(os.path.join(config['work_dir'], "acc_pytorch.png"))


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
    for epoch in range(6):
        interrupted_dir = os.path.join(config['work_dir'], f"epoch_{epoch}")
        uninterrupted_epoch_dir = os.path.join(uninterrupted_dir, f"epoch_{epoch}")

        # Compare training losses
        interrupted_losses = torch.load(os.path.join(interrupted_dir, "training_losses.pt"))
        uninterrupted_losses = torch.load(os.path.join(uninterrupted_epoch_dir, "training_losses.pt"))
        assert torch.allclose(interrupted_losses, uninterrupted_losses), f"Training losses mismatch at epoch {epoch}"

        # Compare validation scores
        with open(os.path.join(interrupted_dir, "validation_scores.json")) as f:
            interrupted_scores = json.load(f)
        with open(os.path.join(uninterrupted_epoch_dir, "validation_scores.json")) as f:
            uninterrupted_scores = json.load(f)
        assert interrupted_scores == uninterrupted_scores, f"Validation scores mismatch at epoch {epoch}"

        print(f"Epoch {epoch} files match between interrupted and uninterrupted training")
