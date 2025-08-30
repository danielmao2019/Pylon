import pytest
import torch
import os
import shutil
import joblib
import copy
from debuggers.wrappers.sequential_debugger import SequentialDebugger
from debuggers.forward_debugger import ForwardDebugger
from debuggers.base_debugger import BaseDebugger
from runners.trainers.supervised_single_task_trainer import SupervisedSingleTaskTrainer
from configs.examples.linear.config import config as linear_config


class DebugForwardDebugger(ForwardDebugger):
    """Debug forward debugger that captures linear layer outputs."""

    def process_forward(self, module, input, output):
        return {
            'module_name': type(module).__name__,
            'output_shape': list(output.shape) if isinstance(output, torch.Tensor) else None,
            'output_mean': float(output.mean()) if isinstance(output, torch.Tensor) else None,
            'input_shape': list(input[0].shape) if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor) else None
        }


class DebugBaseDebugger(BaseDebugger):
    """Debug base debugger for integration testing."""

    def __call__(self, datapoint, model):
        # Access real model outputs for meaningful debugging
        outputs = datapoint.get('outputs', torch.tensor([0.0]))
        return {
            'model_output_mean': float(outputs.mean()) if isinstance(outputs, torch.Tensor) else 0.0,
            'model_output_shape': list(outputs.shape) if isinstance(outputs, torch.Tensor) else [],
            'debugger_type': 'base_debugger'
        }


def create_simple_test_dataset_config():
    """Create a simple dataset config for testing without complex transforms."""
    from data.datasets.random_datasets import BaseRandomDataset

    return {
        'class': BaseRandomDataset,
        'args': {
            'num_examples': 6,  # Small dataset for testing
            'base_seed': 0,
            'gen_func_config': {
                'inputs': {
                    'x': (
                        torch.randn,
                        {'size': (2,), 'dtype': torch.float32},
                    ),
                },
                'labels': {
                    'y': (
                        torch.randn,
                        {'size': (2,), 'dtype': torch.float32},
                    ),
                },
            },
            'transforms_cfg': None,  # No transforms to avoid seed issues
        },
    }


def create_linear_config_with_debugger(work_dir, tot_epochs=3, checkpoint_method=2):
    """Create linear experiment config with debugger for integration testing."""
    # Start with the real linear config
    config = copy.deepcopy(linear_config)

    # Override specific settings for testing
    config['work_dir'] = work_dir
    config['epochs'] = tot_epochs
    config['checkpoint_method'] = checkpoint_method

    # Shorten the training/val seeds for shorter test runs
    config['train_seeds'] = [0] * tot_epochs
    config['val_seeds'] = [42] * tot_epochs
    config['test_seed'] = 999

    # Use simpler dataset to avoid device and seed issues
    simple_dataset = create_simple_test_dataset_config()
    config['train_dataset'] = simple_dataset
    config['val_dataset'] = simple_dataset

    # Reduce batch size and workers for faster testing
    config['train_dataloader']['args']['batch_size'] = 2
    config['train_dataloader']['args']['num_workers'] = 0  # Disable multiprocessing
    config['val_dataloader']['args']['batch_size'] = 1     # Validation needs batch_size=1
    config['val_dataloader']['args']['num_workers'] = 0    # Disable multiprocessing

    # Add debugger configuration
    config['debugger'] = {
        'class': SequentialDebugger,
        'args': {
            'page_size_mb': 1,  # Small page size for testing
            'debuggers_config': [
                {
                    'name': 'linear_layer',
                    'debugger_config': {
                        'class': DebugForwardDebugger,
                        'args': {'layer_name': 'linear'}
                    }
                },
                {
                    'name': 'model_outputs',
                    'debugger_config': {
                        'class': DebugBaseDebugger,
                        'args': {}
                    }
                }
            ]
        }
    }

    return config


def validate_debugger_output(debugger_dir: str) -> None:
    """Validate debugger output files and content structure."""
    # Should have page files
    page_files = [f for f in os.listdir(debugger_dir) if f.startswith('page_') and f.endswith('.pkl')]
    assert len(page_files) > 0, f"No page files in {debugger_dir}"

    # Load and verify page content
    for page_file in page_files:
        page_path = os.path.join(debugger_dir, page_file)
        page_data = joblib.load(page_path)

        # Should be dict mapping datapoint_idx to debug_outputs
        assert isinstance(page_data, dict)

        for debug_outputs in page_data.values():
            # Each debug output should have our two debuggers
            assert 'linear_layer' in debug_outputs
            assert 'model_outputs' in debug_outputs

            # Verify forward debugger captured linear layer data
            linear_data = debug_outputs['linear_layer']
            assert linear_data is not None
            assert 'module_name' in linear_data
            assert linear_data['module_name'] == 'Linear'
            assert 'output_shape' in linear_data
            assert 'input_shape' in linear_data

            # Verify base debugger data
            output_data = debug_outputs['model_outputs']
            assert 'model_output_mean' in output_data
            assert 'model_output_shape' in output_data
            assert output_data['debugger_type'] == 'base_debugger'


def test_linear_experiment_integration_with_debugger():
    """Test complete integration using real linear experiment with debugger."""
    test_work_dir = "./logs/tests/linear_debugger_integration"

    # Ensure parent directory exists
    os.makedirs("./logs/tests", exist_ok=True)

    # Clean up any existing test directory
    if os.path.exists(test_work_dir):
        shutil.rmtree(test_work_dir)

    # Create config based on real linear experiment
    config = create_linear_config_with_debugger(
        work_dir=test_work_dir,
        tot_epochs=3,
        checkpoint_method=2  # Save every 2 epochs + last
    )

    # Create and run real SupervisedSingleTaskTrainer
    trainer = SupervisedSingleTaskTrainer(config, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    trainer.run()

    # Verify results: debugger data should only exist for checkpoint epochs [1, 2]
    expected_checkpoint_indices = [1, 2]  # Every 2 epochs + last for 3 epochs

    for epoch_idx in range(3):
        epoch_path = os.path.join(test_work_dir, f'epoch_{epoch_idx}')
        debugger_path = os.path.join(epoch_path, 'debugger')

        if epoch_idx in expected_checkpoint_indices:
            # Should have debugger data
            assert os.path.exists(debugger_path), f"Missing debugger dir in epoch_{epoch_idx}"
            validate_debugger_output(debugger_path)
        else:
            # Should not have debugger data
            assert not os.path.exists(debugger_path), f"Unexpected debugger dir in epoch_{epoch_idx}"


def test_forward_hook_registration_and_execution():
    """Test that forward hooks are properly registered and execute during model forward pass."""
    test_work_dir = "./logs/tests/hook_test"

    # Ensure parent directory exists
    os.makedirs("./logs/tests", exist_ok=True)

    # Clean up any existing test directory
    if os.path.exists(test_work_dir):
        shutil.rmtree(test_work_dir)

    # Create config based on linear experiment
    config = create_linear_config_with_debugger(
        work_dir=test_work_dir,
        tot_epochs=1,
        checkpoint_method='all'
    )

    # Create trainer and initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SupervisedSingleTaskTrainer(config, device)
    trainer._init_components_()

    model = trainer.model
    debugger = trainer.debugger

    # Verify hooks were registered on linear layer
    assert len(debugger.forward_debuggers) == 1
    assert 'linear' in debugger.forward_debuggers

    # Verify no hooks captured initially
    linear_debugger = debugger.debuggers['linear_layer']
    assert linear_debugger.last_capture is None

    # Create test input (matches linear model input format)
    test_input = {'x': torch.randn(2, 2, dtype=torch.float32, device=device)}

    debugger.enabled = True

    with torch.no_grad():
        # Run model forward pass - this should trigger hooks
        model_output = model(test_input)

    # Verify hooks captured data during forward pass
    assert linear_debugger.last_capture is not None

    # Verify captured data structure
    linear_capture = linear_debugger.last_capture
    assert linear_capture['module_name'] == 'Linear'
    assert linear_capture['output_shape'] == [2, 2]  # Expected linear output shape
    assert linear_capture['input_shape'] == [2, 2]   # Expected linear input shape
    assert isinstance(linear_capture['output_mean'], float)

    # Create test datapoint
    datapoint = {
        'inputs': test_input,
        'outputs': model_output,
        'meta_info': {'idx': [0]}
    }

    # Call debugger - should return captured hook data
    debug_outputs = debugger(datapoint, model)

    # Verify all debuggers returned data
    assert len(debug_outputs) == 2
    assert 'linear_layer' in debug_outputs
    assert 'model_outputs' in debug_outputs

    # Verify forward hook data is returned
    assert debug_outputs['linear_layer'] == linear_capture
    assert debug_outputs['model_outputs']['debugger_type'] == 'base_debugger'
