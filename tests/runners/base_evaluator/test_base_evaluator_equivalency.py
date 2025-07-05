from typing import Dict, Any
import pytest
import tempfile
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics.base_metric import BaseMetric
from runners.base_evaluator import BaseEvaluator
from utils.logging.text_logger import TextLogger
from utils.ops import transpose_buffer
from utils.io import save_json
from data.collators.base_collator import BaseCollator


class SimpleTestDataset(torch.utils.data.Dataset):
    """Simple test dataset that doesn't use BaseDataset to avoid device issues."""

    def __init__(self, num_examples: int = 20, num_classes: int = 10, seed: int = 42):
        self.num_examples = num_examples
        self.num_classes = num_classes
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate fixed random data
        self.images = torch.randn(num_examples, 3, 32, 32, dtype=torch.float32)
        self.labels = torch.randint(0, num_classes, (num_examples,), dtype=torch.int64)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return {
            'inputs': {'image': self.images[idx]},
            'labels': {'target': self.labels[idx]},
            'meta_info': {'idx': idx},
        }


class SimpleTestLogger(TextLogger):
    """TextLogger with missing eval/train methods for testing."""

    def eval(self):
        """Switch to evaluation mode."""
        pass

    def train(self):
        """Switch to training mode."""
        pass


class SimpleTestEvaluator(BaseEvaluator):
    """Custom evaluator with TestTextLogger for testing."""

    def _init_logger(self) -> None:
        """Initialize logger with TestTextLogger instead of trying ScreenLogger."""
        session_idx = 0  # Simplified for testing

        # Create a simple git log (skip actual git commands for testing)
        git_log = os.path.join(self.work_dir, f"git_{session_idx}.log")
        with open(git_log, 'w') as f:
            f.write("# Git log placeholder for testing\n")

        # Use TestTextLogger directly
        log_filepath = os.path.join(self.work_dir, f"eval_{session_idx}.log")
        self.logger = SimpleTestLogger(filepath=log_filepath)

        # Save config (simplified)
        with open(os.path.join(self.work_dir, "config.json"), mode='w') as f:
            json.dump(self.config, f, indent=2, default=str)

        # Initialize mock system monitor for testing
        class MockSystemMonitor:
            def start(self): pass
            def stop(self): pass
            def log_stats(self, logger): pass

        self.system_monitor = MockSystemMonitor()


class SimpleTestModel(nn.Module):
    """Simple classification model for testing."""

    def __init__(self, num_classes: int = 10, input_size: int = 3*32*32):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        image = inputs['image']
        batch_size = image.shape[0] if len(image.shape) == 4 else 1
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Move to same device as model
        image = image.to(next(self.parameters()).device)

        x = self.flatten(image)
        logits = self.classifier(x)

        return {'logits': logits}


class ErrorTestDataset(torch.utils.data.Dataset):
    """Test dataset that triggers errors at specific indices."""

    def __init__(self, num_examples: int = 20, error_indices: list = None, seed: int = 42):
        self.num_examples = num_examples
        self.error_indices = error_indices or []
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate fixed random data
        self.images = torch.randn(num_examples, 3, 32, 32, dtype=torch.float32)
        self.labels = torch.randint(0, 10, (num_examples,), dtype=torch.int64)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Trigger error at specific indices
        if idx in self.error_indices:
            raise ValueError(f"Test error at dataset index {idx}")

        return {
            'inputs': {'image': self.images[idx]},
            'labels': {'target': self.labels[idx]},
            'meta_info': {'idx': idx},
        }


class SimpleTestMetric(BaseMetric):
    """Simple accuracy metric for testing."""

    DIRECTIONS = {"accuracy": 1, "num_samples": 1}  # Higher is better

    def __init__(self):
        super().__init__()

    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Calculate accuracy."""
        # Extract outputs and labels from datapoint
        y_pred = datapoint['outputs']
        y_true = datapoint['labels']

        logits = y_pred['logits']
        targets = y_true['target']

        if len(targets.shape) == 0:
            targets = targets.unsqueeze(0)

        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).float()
        accuracy = correct.mean()

        scores = {
            'accuracy': accuracy,
            'num_samples': torch.tensor(len(targets), dtype=torch.float32)
        }

        # Add to buffer for summarization
        self.add_to_buffer(scores, datapoint)

        return scores

    def summarize(self, output_path: str = None) -> Dict[str, Any]:
        """Summarize metrics across all batches following Pylon pattern."""
        # Wait for buffer to be processed
        self._buffer_queue.join()

        buffer = self.get_buffer()
        if not buffer:
            aggregated = {'accuracy': 0.0, 'reduced': 0.0}
            per_datapoint = {'accuracy': [], 'num_samples': []}
        else:
            # Compute aggregated metrics
            total_correct = sum(item['accuracy'] * item['num_samples'] for item in buffer)
            total_samples = sum(item['num_samples'] for item in buffer)
            final_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

            aggregated = {
                'accuracy': float(final_accuracy),
                'reduced': float(final_accuracy)  # For compatibility with BaseTrainer._find_best_checkpoint_
            }

            # Create per_datapoint structure by transposing buffer
            per_datapoint = transpose_buffer(buffer)

        # Create result following Pylon pattern
        result = {
            'aggregated': aggregated,
            'per_datapoint': per_datapoint
        }

        if output_path:
            save_json(obj=result, filepath=output_path)

        return result


def create_test_config(work_dir: str, eval_n_jobs: int, seed: int = 42) -> Dict[str, Any]:
    """Create a test configuration for the evaluator."""
    return {
        'work_dir': work_dir,
        'seed': seed,
        'eval_n_jobs': eval_n_jobs,
        'eval_dataset': {
            'class': SimpleTestDataset,
            'args': {
                'num_classes': 10,
                'num_examples': 20,  # Small dataset for fast testing
                'seed': seed
            }
        },
        'eval_dataloader': {
            'class': DataLoader,
            'args': {
                'batch_size': 1,  # Process one sample at a time for clear comparison
                'shuffle': False,
                'num_workers': 0,
                'drop_last': False,
                'collate_fn': BaseCollator()
            }
        },
        'model': {
            'class': SimpleTestModel,
            'args': {
                'num_classes': 10,
                'input_size': 3*32*32
            }
        },
        'metric': {
            'class': SimpleTestMetric,
            'args': {}
        }
    }


def run_evaluator(config: Dict[str, Any]) -> Dict[str, float]:
    """Run evaluator and return results."""
    # Force CPU device for testing
    evaluator = SimpleTestEvaluator(config, device=torch.device('cpu'))
    evaluator.run()

    # Load the evaluation scores
    scores_path = os.path.join(config['work_dir'], 'evaluation_scores.json')
    assert os.path.exists(scores_path), f"Evaluation scores file not found at {scores_path}"

    with open(scores_path, 'r') as f:
        scores = json.load(f)

    return scores


def test_base_evaluator_single_vs_multi_worker():
    """
    Integration test comparing eval_n_jobs=1 (sequential) vs multi-worker execution.
    Verifies that both produce identical evaluation_scores.json files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create separate directories for single and multi-worker runs
        single_worker_dir = os.path.join(temp_dir, 'single_worker')
        multi_worker_dir = os.path.join(temp_dir, 'multi_worker')
        os.makedirs(single_worker_dir, exist_ok=True)
        os.makedirs(multi_worker_dir, exist_ok=True)

        # Create configurations
        single_worker_config = create_test_config(single_worker_dir, eval_n_jobs=1, seed=123)
        multi_worker_config = create_test_config(multi_worker_dir, eval_n_jobs=3, seed=123)

        # Run both evaluators
        single_worker_scores = run_evaluator(single_worker_config)
        multi_worker_scores = run_evaluator(multi_worker_config)

        # Compare results structure
        assert single_worker_scores.keys() == multi_worker_scores.keys(), \
            f"Score keys differ: {single_worker_scores.keys()} vs {multi_worker_scores.keys()}"

        assert 'aggregated' in single_worker_scores, "Missing aggregated scores"
        assert 'per_datapoint' in single_worker_scores, "Missing per_datapoint scores"

        # Compare aggregated metrics (should be identical)
        single_agg = single_worker_scores['aggregated']
        multi_agg = multi_worker_scores['aggregated']

        assert single_agg.keys() == multi_agg.keys(), \
            f"Aggregated keys differ: {single_agg.keys()} vs {multi_agg.keys()}"

        for key in single_agg.keys():
            single_value = single_agg[key]
            multi_value = multi_agg[key]

            # Allow small floating point differences
            if isinstance(single_value, (int, float)) and isinstance(multi_value, (int, float)):
                assert abs(single_value - multi_value) < 1e-6, \
                    f"Aggregated scores differ for {key}: {single_value} vs {multi_value}"
            else:
                assert single_value == multi_value, \
                    f"Aggregated scores differ for {key}: {single_value} vs {multi_value}"

        # Compare per_datapoint metrics (this tests execution order!)
        single_per_dp = single_worker_scores['per_datapoint']
        multi_per_dp = multi_worker_scores['per_datapoint']

        assert single_per_dp.keys() == multi_per_dp.keys(), \
            f"Per-datapoint keys differ: {single_per_dp.keys()} vs {multi_per_dp.keys()}"

        for key in single_per_dp.keys():
            single_list = single_per_dp[key]
            multi_list = multi_per_dp[key]

            assert len(single_list) == len(multi_list), \
                f"Per-datapoint list lengths differ for {key}: {len(single_list)} vs {len(multi_list)}"

            # Compare each element in order - this verifies execution order is preserved
            for i, (single_val, multi_val) in enumerate(zip(single_list, multi_list)):
                if isinstance(single_val, (int, float)) and isinstance(multi_val, (int, float)):
                    assert abs(single_val - multi_val) < 1e-6, \
                        f"Per-datapoint values differ at index {i} for {key}: {single_val} vs {multi_val}"
                else:
                    assert single_val == multi_val, \
                        f"Per-datapoint values differ at index {i} for {key}: {single_val} vs {multi_val}"

        print(f"✓ Single worker scores: {single_worker_scores}")
        print(f"✓ Multi worker scores: {multi_worker_scores}")
        print("✓ Both evaluation methods produced identical results!")


def test_base_evaluator_deterministic_results():
    """
    Test that multiple runs with the same configuration produce identical results.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        all_scores = []

        for run_idx in range(3):
            run_dir = os.path.join(temp_dir, f'run_{run_idx}')
            os.makedirs(run_dir, exist_ok=True)

            config = create_test_config(run_dir, eval_n_jobs=2, seed=456)
            scores = run_evaluator(config)
            all_scores.append(scores)

        # Verify all runs produce identical results
        reference_scores = all_scores[0]
        for run_idx, scores in enumerate(all_scores[1:], 1):
            assert scores.keys() == reference_scores.keys(), \
                f"Run {run_idx} keys differ from reference"

            # Compare aggregated scores
            ref_agg = reference_scores['aggregated']
            run_agg = scores['aggregated']
            for key in ref_agg.keys():
                ref_value = ref_agg[key]
                run_value = run_agg[key]

                if isinstance(ref_value, (int, float)) and isinstance(run_value, (int, float)):
                    assert abs(ref_value - run_value) < 1e-6, \
                        f"Run {run_idx} aggregated differs for {key}: {ref_value} vs {run_value}"
                else:
                    assert ref_value == run_value, \
                        f"Run {run_idx} aggregated differs for {key}: {ref_value} vs {run_value}"

            # Compare per_datapoint scores (order should be identical for same seed)
            ref_per_dp = reference_scores['per_datapoint']
            run_per_dp = scores['per_datapoint']
            for key in ref_per_dp.keys():
                ref_list = ref_per_dp[key]
                run_list = run_per_dp[key]
                assert len(ref_list) == len(run_list), \
                    f"Run {run_idx} per_datapoint length differs for {key}"

                for i, (ref_val, run_val) in enumerate(zip(ref_list, run_list)):
                    if isinstance(ref_val, (int, float)) and isinstance(run_val, (int, float)):
                        assert abs(ref_val - run_val) < 1e-6, \
                            f"Run {run_idx} per_datapoint differs at index {i} for {key}: {ref_val} vs {run_val}"
                    else:
                        assert ref_val == run_val, \
                            f"Run {run_idx} per_datapoint differs at index {i} for {key}: {ref_val} vs {run_val}"

        print(f"✓ All {len(all_scores)} runs produced identical results!")


def test_base_evaluator_different_worker_counts():
    """
    Test that different worker counts all produce equivalent results.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        worker_counts = [1, 2, 4]
        all_scores = []

        for worker_count in worker_counts:
            run_dir = os.path.join(temp_dir, f'workers_{worker_count}')
            os.makedirs(run_dir, exist_ok=True)

            config = create_test_config(run_dir, eval_n_jobs=worker_count, seed=789)
            scores = run_evaluator(config)
            all_scores.append((worker_count, scores))

        # Compare all results with the first (reference)
        _, reference_scores = all_scores[0]

        for worker_count, scores in all_scores[1:]:
            assert scores.keys() == reference_scores.keys(), \
                f"Worker count {worker_count} keys differ from reference"

            # Compare aggregated scores
            ref_agg = reference_scores['aggregated']
            worker_agg = scores['aggregated']
            for key in ref_agg.keys():
                ref_value = ref_agg[key]
                worker_value = worker_agg[key]

                if isinstance(ref_value, (int, float)) and isinstance(worker_value, (int, float)):
                    assert abs(ref_value - worker_value) < 1e-6, \
                        f"Worker count {worker_count} aggregated differs for {key}: {ref_value} vs {worker_value}"
                else:
                    assert ref_value == worker_value, \
                        f"Worker count {worker_count} aggregated differs for {key}: {ref_value} vs {worker_value}"

            # Compare per_datapoint scores (order should be identical for same seed)
            ref_per_dp = reference_scores['per_datapoint']
            worker_per_dp = scores['per_datapoint']
            for key in ref_per_dp.keys():
                ref_list = ref_per_dp[key]
                worker_list = worker_per_dp[key]
                assert len(ref_list) == len(worker_list), \
                    f"Worker count {worker_count} per_datapoint length differs for {key}"

                for i, (ref_val, worker_val) in enumerate(zip(ref_list, worker_list)):
                    if isinstance(ref_val, (int, float)) and isinstance(worker_val, (int, float)):
                        assert abs(ref_val - worker_val) < 1e-6, \
                            f"Worker count {worker_count} per_datapoint differs at index {i} for {key}: {ref_val} vs {worker_val}"
                    else:
                        assert ref_val == worker_val, \
                            f"Worker count {worker_count} per_datapoint differs at index {i} for {key}: {ref_val} vs {worker_val}"

        print(f"✓ All worker counts {worker_counts} produced identical results!")


def test_base_evaluator_file_structure():
    """
    Test that both single and multi-worker evaluators create the expected file structure.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        configs = [
            create_test_config(os.path.join(temp_dir, 'single'), eval_n_jobs=1),
            create_test_config(os.path.join(temp_dir, 'multi'), eval_n_jobs=3)
        ]

        for config in configs:
            os.makedirs(config['work_dir'], exist_ok=True)
            run_evaluator(config)

            # Check that expected files exist
            expected_files = [
                'evaluation_scores.json',
                'eval_0.log',
                'git_0.log',
                'config.json'
            ]

            for filename in expected_files:
                filepath = os.path.join(config['work_dir'], filename)
                assert os.path.exists(filepath), f"Expected file {filename} not found in {config['work_dir']}"

            # Verify evaluation_scores.json is valid JSON with proper structure
            scores_path = os.path.join(config['work_dir'], 'evaluation_scores.json')
            with open(scores_path, 'r') as f:
                scores = json.load(f)
                assert isinstance(scores, dict), "Evaluation scores should be a dictionary"
                assert 'aggregated' in scores, "Aggregated scores should be present"
                assert 'per_datapoint' in scores, "Per-datapoint scores should be present"
                assert 'accuracy' in scores['aggregated'], "Accuracy metric should be present in aggregated"
                assert 'reduced' in scores['aggregated'], "Reduced metric should be present in aggregated"

        print("✓ Both evaluators created expected file structure!")


def test_base_evaluator_error_equivalency():
    """
    Test that single-worker (sequential) and multi-worker evaluators handle errors identically.
    Both should fail at the same datapoint with the same error.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create datasets that will error at index 5
        error_indices = [5]

        # Test single worker (sequential) error handling
        single_worker_dir = os.path.join(temp_dir, 'single_worker_error')
        os.makedirs(single_worker_dir, exist_ok=True)

        single_worker_config = {
            'work_dir': single_worker_dir,
            'seed': 42,
            'eval_n_jobs': 1,  # Sequential execution
            'eval_dataset': {
                'class': ErrorTestDataset,
                'args': {
                    'num_examples': 10,
                    'error_indices': error_indices,
                    'seed': 42
                }
            },
            'eval_dataloader': {
                'class': DataLoader,
                'args': {
                    'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0,
                    'drop_last': False,
                    'collate_fn': BaseCollator()
                }
            },
            'model': {
                'class': SimpleTestModel,
                'args': {
                    'num_classes': 10,
                    'input_size': 3*32*32
                }
            },
            'metric': {
                'class': SimpleTestMetric,
                'args': {}
            }
        }

        # Test multi-worker error handling
        multi_worker_dir = os.path.join(temp_dir, 'multi_worker_error')
        os.makedirs(multi_worker_dir, exist_ok=True)

        multi_worker_config = single_worker_config.copy()
        multi_worker_config['work_dir'] = multi_worker_dir
        multi_worker_config['eval_n_jobs'] = 3  # Parallel execution

        # Both should raise the same error
        single_worker_error = None
        multi_worker_error = None

        # Test single worker execution
        try:
            run_evaluator(single_worker_config)
        except Exception as e:
            single_worker_error = e

        # Test multi-worker execution
        try:
            run_evaluator(multi_worker_config)
        except Exception as e:
            multi_worker_error = e

        # Both should have failed with errors
        assert single_worker_error is not None, "Single worker should have raised an error"
        assert multi_worker_error is not None, "Multi worker should have raised an error"

        # Error types should be the same (both should propagate the ValueError)
        assert type(single_worker_error) == type(multi_worker_error), \
            f"Error types differ: {type(single_worker_error)} vs {type(multi_worker_error)}"

        # Error messages should contain the same dataset error
        assert "Test error at dataset index 5" in str(single_worker_error), \
            f"Single worker error should mention index 5: {single_worker_error}"
        assert "Test error at dataset index 5" in str(multi_worker_error), \
            f"Multi worker error should mention index 5: {multi_worker_error}"

        print(f"✓ Both execution methods failed with same error: {type(single_worker_error).__name__}")
        print(f"✓ Single worker error: {single_worker_error}")
        print(f"✓ Multi worker error: {multi_worker_error}")


def test_base_evaluator_early_vs_late_error():
    """
    Test error behavior when errors occur early vs late in the evaluation.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test early error (index 1)
        early_error_dir = os.path.join(temp_dir, 'early_error')
        os.makedirs(early_error_dir, exist_ok=True)

        early_error_config = create_test_config(early_error_dir, eval_n_jobs=2, seed=42)
        early_error_config['eval_dataset'] = {
            'class': ErrorTestDataset,
            'args': {
                'num_examples': 10,
                'error_indices': [1],  # Early error
                'seed': 42
            }
        }

        # Test late error (index 8)
        late_error_dir = os.path.join(temp_dir, 'late_error')
        os.makedirs(late_error_dir, exist_ok=True)

        late_error_config = create_test_config(late_error_dir, eval_n_jobs=2, seed=42)
        late_error_config['eval_dataset'] = {
            'class': ErrorTestDataset,
            'args': {
                'num_examples': 10,
                'error_indices': [8],  # Late error
                'seed': 42
            }
        }

        # Both should fail with ValueError mentioning the specific index
        with pytest.raises(Exception) as early_exc_info:
            run_evaluator(early_error_config)
        assert "Test error at dataset index 1" in str(early_exc_info.value)

        with pytest.raises(Exception) as late_exc_info:
            run_evaluator(late_error_config)
        assert "Test error at dataset index 8" in str(late_exc_info.value)

        print("✓ Both early and late errors handled correctly with fail-fast behavior")


def test_base_evaluator_multiple_error_points():
    """
    Test that evaluation fails on the first error when multiple error points exist.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        multi_error_dir = os.path.join(temp_dir, 'multi_error')
        os.makedirs(multi_error_dir, exist_ok=True)

        # Dataset with errors at indices 3 and 7
        multi_error_config = create_test_config(multi_error_dir, eval_n_jobs=2, seed=42)
        multi_error_config['eval_dataset'] = {
            'class': ErrorTestDataset,
            'args': {
                'num_examples': 10,
                'error_indices': [3, 7],  # Multiple potential errors
                'seed': 42
            }
        }

        # Should fail on the first error encountered (due to fail-fast)
        with pytest.raises(Exception) as exc_info:
            run_evaluator(multi_error_config)

        error_message = str(exc_info.value)
        # Should fail at index 3 or 7, but consistently on the same one due to deterministic execution
        assert ("Test error at dataset index 3" in error_message or
                "Test error at dataset index 7" in error_message), \
            f"Error should mention index 3 or 7: {error_message}"

        print(f"✓ Multi-error scenario handled with fail-fast: {error_message}")
