"""Test cases for MultiStageTrainer comparing with SupervisedSingleTaskTrainer."""
import os
import json
import torch
from runners.multi_stage_trainer import MultiStageTrainer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from metrics.wrappers.single_task_metric import SingleTaskMetric
import optimizers
from utils.ops.dict_as_tensor import buffer_allclose


class SimpleMetric(SingleTaskMetric):
    """A simple metric implementation for testing."""

    DIRECTION = -1  # Lower is better for MSE

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute MSE score."""
        score = torch.mean((y_pred - y_true) ** 2)
        return {"mse": score}

    def __call__(self, datapoint: dict) -> dict[str, torch.Tensor]:
        """Update the metric buffer with the current batch's score."""
        # Extract outputs and labels from datapoint
        y_pred = datapoint['outputs']
        y_true = datapoint['labels']
        
        score = self._compute_score(y_pred, y_true)
        # Add to buffer
        self.add_to_buffer(score, datapoint)
        return score



class SimpleDataset(torch.utils.data.Dataset):
    """A simple dataset for testing."""

    def __init__(self, size=100, device='cuda'):
        self.device = device
        self.data = torch.randn(size, 10, device=device)
        self.labels = torch.randn(size, 1, device=device)
        self.base_seed = 0  # Add base_seed attribute

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'inputs': self.data[idx], 
            'labels': self.labels[idx],
            'meta_info': {'idx': idx}  # Add meta_info for proper structure
        }
    
    def set_base_seed(self, seed: int) -> None:
        """Set the base seed for deterministic behavior."""
        self.base_seed = seed


class SimpleModel(torch.nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class SimpleCriterion(SingleTaskCriterion):
    """A simple criterion for testing."""
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        return self.mse(y_pred, y_true)


class SupervisedMultiStageTrainer(SupervisedSingleTaskTrainer, MultiStageTrainer):
    """A concrete trainer class that combines supervised training with multi-stage functionality."""
    def __init__(self, stage_configs: list[dict]):
        """Initialize the trainer with a list of stage configurations.

        Args:
            stage_configs: List of configuration dictionaries for each stage.
        """
        MultiStageTrainer.__init__(self, stage_configs=stage_configs)


def create_base_config(work_dir: str, epochs: int, model, dataset, metric) -> dict:
    """Create a base config that can be used for both trainers."""
    return {
        'work_dir': work_dir,
        'init_seed': 42,
        'epochs': epochs,
        'train_seeds': [42] * epochs,
        'val_seeds': [42] * epochs,  # Add missing val_seeds
        'test_seed': 42,  # Add missing test_seed
        'checkpoint_method': 'all',
        'train_dataset': {
            'class': dataset,
            'args': {'size': 100, 'device': 'cuda'}
        },
        'train_dataloader': {
            'class': torch.utils.data.DataLoader,
            'args': {
                'batch_size': 32,
                'shuffle': True
            }
        },
        'criterion': {
            'class': SimpleCriterion,
            'args': {}
        },
        'val_dataset': {
            'class': dataset,
            'args': {'size': 100, 'device': 'cuda'}
        },
        'val_dataloader': {
            'class': torch.utils.data.DataLoader,
            'args': {
                'batch_size': 1,  # Pylon evaluators expect batch_size=1
                'shuffle': False
            }
        },
        'metric': {
            'class': metric,
            'args': {}
        },
        'model': {
            'class': model,
            'args': {}
        },
        'optimizer': {
            'class': optimizers.SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'lr': 1.0e-03,
                    },
                },
            },
        },
        'scheduler': {
            'class': torch.optim.lr_scheduler.ConstantLR,
            'args': {
                'factor': 1.0
            }
        }
    }


def test_multi_stage_vs_single_stage(test_dir):
    """Compare SupervisedMultiStageTrainer with SupervisedSingleTaskTrainer."""
    # Create configs
    single_stage_config = create_base_config(
        os.path.join(test_dir, "single_stage"),
        epochs=10,
        model=SimpleModel,
        dataset=SimpleDataset,
        metric=SimpleMetric
    )

    # Create two identical stage configs for multi-stage
    stage1_config = create_base_config(
        os.path.join(test_dir, "multi_stage_stage1"),
        epochs=5,
        model=SimpleModel,
        dataset=SimpleDataset,
        metric=SimpleMetric
    )
    stage2_config = create_base_config(
        os.path.join(test_dir, "multi_stage_stage2"),
        epochs=5,
        model=SimpleModel,
        dataset=SimpleDataset,
        metric=SimpleMetric
    )

    # Initialize and run single stage trainer first
    single_trainer = SupervisedSingleTaskTrainer(config=single_stage_config)
    single_trainer.run()

    # Explicitly delete the first trainer to ensure its logger is cleaned up
    del single_trainer

    # Then initialize and run multi-stage trainer
    multi_trainer = SupervisedMultiStageTrainer(stage_configs=[stage1_config, stage2_config])
    multi_trainer.run()

    # Compare model parameters at each epoch
    for epoch in range(10):
        # Load checkpoints
        single_checkpoint = torch.load(os.path.join(single_stage_config['work_dir'], f"epoch_{epoch}", "checkpoint.pt"))
        multi_checkpoint = torch.load(os.path.join(stage1_config['work_dir'], f"epoch_{epoch}", "checkpoint.pt"))

        # Compare model states
        single_model_state = single_checkpoint['model_state_dict']
        multi_model_state = multi_checkpoint['model_state_dict']

        # Check that all parameters match exactly
        for (single_name, single_param), (multi_name, multi_param) in zip(
            single_model_state.items(), multi_model_state.items()
        ):
            assert single_name == multi_name, f"Parameter names don't match: {single_name} vs {multi_name}. {epoch=}"
            assert torch.allclose(single_param, multi_param), f"Parameters don't match for {single_name}. {epoch=}"

        # Compare optimizer states
        single_optim_state = single_checkpoint['optimizer_state_dict']
        multi_optim_state = multi_checkpoint['optimizer_state_dict']

        # Check that all optimizer states match exactly
        for (single_name, single_param), (multi_name, multi_param) in zip(
            single_optim_state.items(), multi_optim_state.items()
        ):
            assert single_name == multi_name, f"Optimizer state names don't match: {single_name} vs {multi_name}"
            if isinstance(single_param, torch.Tensor):
                assert torch.allclose(single_param, multi_param), f"Optimizer states don't match for {single_name}"
            else:
                assert single_param == multi_param, f"Optimizer states don't match for {single_name}"

        # Compare training losses
        single_losses = torch.load(os.path.join(single_stage_config['work_dir'], f"epoch_{epoch}", "training_losses.pt"))
        multi_losses = torch.load(os.path.join(stage1_config['work_dir'], f"epoch_{epoch}", "training_losses.pt"))
        assert torch.allclose(single_losses, multi_losses), f"Training losses don't match at epoch {epoch}"

        # Compare validation scores
        with open(os.path.join(single_stage_config['work_dir'], f"epoch_{epoch}", "validation_scores.json")) as f:
            single_scores = json.load(f)
        with open(os.path.join(stage1_config['work_dir'], f"epoch_{epoch}", "validation_scores.json")) as f:
            multi_scores = json.load(f)
        
        # Compare validation scores using buffer_allclose utility
        assert buffer_allclose(single_scores, multi_scores, rtol=1e-6, atol=1e-6), \
            f"Validation scores don't match at epoch {epoch}: {single_scores} vs {multi_scores}"
