"""Test cases for MultiStageTrainer comparing with SupervisedSingleTaskTrainer."""
import os
import json
import torch
from runners.multi_stage_trainer import MultiStageTrainer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
from metrics.base_metric import BaseMetric


class SimpleMetric(BaseMetric):
    """A simple metric implementation for testing."""

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute MSE score."""
        score = torch.mean((y_pred - y_true) ** 2)
        return {"mse": score}

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, torch.Tensor]:
        """Update the metric buffer with the current batch's score."""
        score = self._compute_score(y_pred, y_true)
        # Detach and move to CPU
        score = {k: v.detach().cpu() for k, v in score.items()}
        # Add to buffer
        self.add_to_buffer(score)
        return score

    def summarize(self, output_path=None) -> dict[str, float]:
        """Calculate average score from buffer and optionally save to file."""
        if not self.buffer:
            return {"mse": 0.0}

        # Calculate average score
        mse_scores = [score["mse"] for score in self.buffer]
        avg_score = sum(mse_scores) / len(mse_scores)
        result = {"mse": avg_score}

        # Save to file if path is provided
        if output_path:
            save_json(obj=result, filepath=output_path)

        return result


class SimpleDataset(torch.utils.data.Dataset):
    """A simple dataset for testing."""

    def __init__(self, size=100, device='cuda'):
        self.device = device
        self.data = torch.randn(size, 10, device=device)
        self.labels = torch.randn(size, 1, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'inputs': self.data[idx], 'labels': self.labels[idx]}


class SimpleModel(torch.nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class SimpleCriterion(torch.nn.Module):
    """A simple criterion for testing."""
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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
        'val_dataset': {
            'class': dataset,
            'args': {'size': 100, 'device': 'cuda'}
        },
        'val_dataloader': {
            'class': torch.utils.data.DataLoader,
            'args': {
                'batch_size': 32,
                'shuffle': False
            }
        },
        'model': {
            'class': model,
            'args': {}
        },
        'criterion': {
            'class': SimpleCriterion,
            'args': {}
        },
        'metric': {
            'class': metric,
            'args': {}
        },
        'optimizer': {
            'class': torch.optim.SGD,
            'args': {
                'lr': 0.01
            }
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

    # Initialize trainers
    single_trainer = SupervisedSingleTaskTrainer(config=single_stage_config)
    multi_trainer = SupervisedMultiStageTrainer(stage_configs=[stage1_config, stage2_config])

    # Run training
    single_trainer.run()
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
            assert single_name == multi_name, f"Parameter names don't match: {single_name} vs {multi_name}"
            assert torch.allclose(single_param, multi_param), f"Parameters don't match for {single_name}"

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

        # Compare scheduler states
        single_sched_state = single_checkpoint['scheduler_state_dict']
        multi_sched_state = multi_checkpoint['scheduler_state_dict']

        # Check that all scheduler states match exactly
        for (single_name, single_param), (multi_name, multi_param) in zip(
            single_sched_state.items(), multi_sched_state.items()
        ):
            assert single_name == multi_name, f"Scheduler state names don't match: {single_name} vs {multi_name}"
            if isinstance(single_param, torch.Tensor):
                assert torch.allclose(single_param, multi_param), f"Scheduler states don't match for {single_name}"
            else:
                assert single_param == multi_param, f"Scheduler states don't match for {single_name}"

        # Compare validation scores
        with open(os.path.join(single_stage_config['work_dir'], f"epoch_{epoch}", "validation_scores.json")) as f:
            single_scores = json.load(f)
        with open(os.path.join(stage1_config['work_dir'], f"epoch_{epoch}", "validation_scores.json")) as f:
            multi_scores = json.load(f)

        assert single_scores == multi_scores, f"Validation scores don't match at epoch {epoch}"
