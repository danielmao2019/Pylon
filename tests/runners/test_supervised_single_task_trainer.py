"""The purpose of this set of test cases is to compare training performance against other frameworks,
including the native PyTorch for loop.
"""
from typing import List, Dict
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
import os
import json
import torch
import torchvision
import data
import criteria
import metrics
import optimizers
import schedulers
import utils
import threading
import time
from utils.automation.run_status import check_epoch_finished


dataset_config = {
    'class': data.datasets.random_datasets.ClassificationRandomDataset,
    'args': {
        'num_examples': 10,
        'num_classes': 10,
        'image_res': (64, 64),
        'initial_seed': 0,
    },
}

class TestModel(torch.nn.Module):

    def __init__(self, model: torch.nn.Module) -> None:
        super(TestModel, self).__init__()
        self.model = model

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert type(inputs) == dict, f"{type(inputs)=}"
        assert len(inputs) == 1, f"{inputs.keys()=}"
        return self.model(list(inputs.values())[0])


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
        'args': {
            'model': torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=10),
        },
    },
    'criterion': {
        'class': criteria.wrappers.PyTorchCriterionWrapper,
        'args': {
            'criterion': torch.nn.CrossEntropyLoss(reduction='mean'),
        },
    },
    'metric': {
        'class': metrics.common.ConfusionMatrix,
        'args': {
            'num_classes': 10,
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
    
    def observer_thread():
        """Monitor training progress and interrupt after epoch 3."""
        while not stop_event.is_set():
            # Check if epoch 3 is complete
            epoch_dir = os.path.join(config['work_dir'], "epoch_2")  # 0-based indexing
            if os.path.exists(epoch_dir) and check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=trainer1.expected_files,
            ):
                # Wait a bit to ensure epoch 3 files are written
                time.sleep(1)
                # Delete trainer to interrupt training
                del trainer1
                break
            time.sleep(0.1)  # Check every 100ms
    
    # Start observer thread
    observer = threading.Thread(target=observer_thread)
    observer.start()
    
    try:
        # Start training in main thread
        trainer1.run()
    except:
        # Training will be interrupted when trainer1 is deleted
        pass
    
    # Signal observer thread to stop
    stop_event.set()
    observer.join()
    
    # Second run - should resume from epoch 3
    trainer2 = SupervisedSingleTaskTrainer(config=config)
    # Verify that trainer2 is resuming from epoch 3
    trainer2._init_components_()
    assert trainer2.cum_epochs == 3, f"Expected to resume from epoch 3, but got {trainer2.cum_epochs}"
    print(f"Successfully resumed training from epoch {trainer2.cum_epochs}")
    
    # Create new event for second interruption
    stop_event = threading.Event()
    
    def observer_thread_2():
        """Monitor training progress and interrupt after epoch 6."""
        while not stop_event.is_set():
            # Check if epoch 6 is complete
            epoch_dir = os.path.join(config['work_dir'], "epoch_5")  # 0-based indexing
            if os.path.exists(epoch_dir) and check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=trainer2.expected_files,
            ):
                # Wait a bit to ensure epoch 6 files are written
                time.sleep(1)
                # Delete trainer to interrupt training
                del trainer2
                break
            time.sleep(0.1)  # Check every 100ms
    
    # Start observer thread
    observer = threading.Thread(target=observer_thread_2)
    observer.start()
    
    try:
        # Continue training in main thread
        trainer2.run()
    except:
        # Training will be interrupted when trainer2 is deleted
        pass
    
    # Signal observer thread to stop
    stop_event.set()
    observer.join()
    
    # Create a new work directory for uninterrupted training
    uninterrupted_dir = config['work_dir'] + "_uninterrupted"
    if os.path.isdir(uninterrupted_dir):
        os.system(' '.join(["rm", "-r", uninterrupted_dir]))
    os.makedirs(uninterrupted_dir)
    
    # Create config for uninterrupted training
    uninterrupted_config = config.copy()
    uninterrupted_config['work_dir'] = uninterrupted_dir
    uninterrupted_config['epochs'] = 6  # Only run 6 epochs
    
    # Run uninterrupted training
    trainer3 = SupervisedSingleTaskTrainer(config=uninterrupted_config)
    trainer3.run()
    
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
