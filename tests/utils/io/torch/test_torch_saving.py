import os
import tempfile
import torch
import pytest
from utils.io.torch import save_torch, load_torch


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_save_torch_tensor(temp_dir):
    """Test saving a simple torch tensor."""
    filepath = os.path.join(temp_dir, "test_tensor.pt")
    
    # Create test tensor
    original_tensor = torch.rand(10, 5)
    
    # Save tensor
    save_torch(obj=original_tensor, filepath=filepath)
    
    # Verify file was created
    assert os.path.exists(filepath)
    assert os.path.getsize(filepath) > 0
    
    # Load back and verify
    loaded_tensor = torch.load(filepath)
    assert torch.equal(original_tensor, loaded_tensor)


def test_save_torch_dict(temp_dir):
    """Test saving a dictionary of tensors."""
    filepath = os.path.join(temp_dir, "test_dict.pt")
    
    # Create test dictionary
    original_dict = {
        'tensor1': torch.rand(5, 5),
        'tensor2': torch.randint(0, 10, (3, 3)),
        'scalar': torch.tensor(42.0),
        'metadata': {
            'name': 'test_dict',
            'version': 1
        }
    }
    
    # Save dictionary
    save_torch(obj=original_dict, filepath=filepath)
    
    # Load back and verify
    loaded_dict = torch.load(filepath)
    assert torch.equal(original_dict['tensor1'], loaded_dict['tensor1'])
    assert torch.equal(original_dict['tensor2'], loaded_dict['tensor2'])
    assert torch.equal(original_dict['scalar'], loaded_dict['scalar'])
    assert original_dict['metadata'] == loaded_dict['metadata']


def test_save_torch_model_state_dict(temp_dir):
    """Test saving a model state dict."""
    filepath = os.path.join(temp_dir, "model_state.pt")
    
    # Create simple model
    model = torch.nn.Linear(10, 5)
    original_state_dict = model.state_dict()
    
    # Save state dict
    save_torch(obj=original_state_dict, filepath=filepath)
    
    # Load back and verify
    loaded_state_dict = torch.load(filepath)
    for key in original_state_dict:
        assert torch.equal(original_state_dict[key], loaded_state_dict[key])


def test_save_torch_auto_create_directory(temp_dir):
    """Test that directories are automatically created."""
    nested_dir = os.path.join(temp_dir, "models", "checkpoints")
    filepath = os.path.join(nested_dir, "model.pt")
    
    # Directory doesn't exist yet
    assert not os.path.exists(nested_dir)
    
    # Save tensor - should create directories
    tensor = torch.rand(2, 2)
    save_torch(obj=tensor, filepath=filepath)
    
    # Directory and file should now exist
    assert os.path.exists(nested_dir)
    assert os.path.exists(filepath)
    
    # Verify content
    loaded_tensor = torch.load(filepath)
    assert torch.equal(tensor, loaded_tensor)


def test_save_torch_atomic_write(temp_dir):
    """Test atomic write behavior (temp file + rename)."""
    filepath = os.path.join(temp_dir, "atomic_test.pt")
    tensor = torch.rand(10, 10)
    
    save_torch(obj=tensor, filepath=filepath)
    
    # File should exist with correct content
    assert os.path.exists(filepath)
    loaded_tensor = torch.load(filepath)
    assert torch.equal(tensor, loaded_tensor)
    
    # No temporary files should remain
    temp_files = [f for f in os.listdir(temp_dir) if f.startswith('torch_') and f.endswith('.tmp')]
    assert len(temp_files) == 0


def test_save_torch_overwrite_existing(temp_dir):
    """Test that save_torch overwrites existing files."""
    filepath = os.path.join(temp_dir, "overwrite.pt")
    
    # Save first tensor
    first_tensor = torch.zeros(3, 3)
    save_torch(obj=first_tensor, filepath=filepath)
    
    assert os.path.exists(filepath)
    loaded_first = torch.load(filepath)
    assert torch.equal(first_tensor, loaded_first)
    
    # Save different tensor to same path
    second_tensor = torch.ones(5, 5)
    save_torch(obj=second_tensor, filepath=filepath)
    
    # Verify the file was overwritten
    loaded_second = torch.load(filepath)
    assert torch.equal(second_tensor, loaded_second)
    assert not torch.equal(first_tensor, loaded_second)


def test_save_torch_different_tensor_types(temp_dir):
    """Test saving different tensor types."""
    test_cases = [
        ('float32', torch.rand(5, 5, dtype=torch.float32)),
        ('float64', torch.rand(5, 5, dtype=torch.float64)),
        ('int32', torch.randint(0, 100, (5, 5), dtype=torch.int32)),
        ('int64', torch.randint(0, 100, (5, 5), dtype=torch.int64)),
        ('bool', torch.randint(0, 2, (5, 5), dtype=torch.bool)),
        ('uint8', torch.randint(0, 255, (5, 5), dtype=torch.uint8))
    ]
    
    for name, original_tensor in test_cases:
        filepath = os.path.join(temp_dir, f"test_{name}.pt")
        
        # Save and load
        save_torch(obj=original_tensor, filepath=filepath)
        loaded_tensor = torch.load(filepath)
        
        # Verify
        assert torch.equal(original_tensor, loaded_tensor)
        assert original_tensor.dtype == loaded_tensor.dtype


def test_save_torch_complex_nested_structure(temp_dir):
    """Test saving complex nested data structures."""
    filepath = os.path.join(temp_dir, "complex_structure.pt")
    
    # Create complex nested structure
    complex_data = {
        'model_state': {
            'layer1': {
                'weight': torch.rand(10, 5),
                'bias': torch.rand(10)
            },
            'layer2': {
                'weight': torch.rand(5, 1),
                'bias': torch.rand(5)
            }
        },
        'optimizer_state': {
            'state': {},
            'param_groups': [
                {'lr': 0.001, 'momentum': 0.9}
            ]
        },
        'training_info': {
            'epoch': 42,
            'loss_history': [1.5, 1.2, 0.9, 0.7],
            'best_accuracy': 0.95
        },
        'metadata': {
            'created_by': 'test_script',
            'version': '1.0.0'
        }
    }
    
    # Save and load
    save_torch(obj=complex_data, filepath=filepath)
    loaded_data = torch.load(filepath)
    
    # Verify structure and contents
    assert torch.equal(
        complex_data['model_state']['layer1']['weight'],
        loaded_data['model_state']['layer1']['weight']
    )
    assert torch.equal(
        complex_data['model_state']['layer1']['bias'],
        loaded_data['model_state']['layer1']['bias']
    )
    assert torch.equal(
        complex_data['model_state']['layer2']['weight'],
        loaded_data['model_state']['layer2']['weight']
    )
    assert torch.equal(
        complex_data['model_state']['layer2']['bias'],
        loaded_data['model_state']['layer2']['bias']
    )
    
    assert loaded_data['optimizer_state'] == complex_data['optimizer_state']
    assert loaded_data['training_info'] == complex_data['training_info']
    assert loaded_data['metadata'] == complex_data['metadata']


def test_save_torch_empty_tensors(temp_dir):
    """Test saving empty tensors."""
    filepath = os.path.join(temp_dir, "empty_tensors.pt")
    
    empty_data = {
        'empty_1d': torch.empty(0),
        'empty_2d': torch.empty(0, 5),
        'empty_3d': torch.empty(0, 0, 0),
        'regular': torch.rand(2, 2)
    }
    
    # Save and load
    save_torch(obj=empty_data, filepath=filepath)
    loaded_data = torch.load(filepath)
    
    # Verify shapes and types
    assert loaded_data['empty_1d'].shape == (0,)
    assert loaded_data['empty_2d'].shape == (0, 5)
    assert loaded_data['empty_3d'].shape == (0, 0, 0)
    assert torch.equal(empty_data['regular'], loaded_data['regular'])


def test_save_torch_with_requires_grad(temp_dir):
    """Test saving tensors with gradient requirements."""
    filepath = os.path.join(temp_dir, "with_grad.pt")
    
    # Create tensors with different gradient requirements
    data = {
        'no_grad': torch.rand(3, 3, requires_grad=False),
        'with_grad': torch.rand(3, 3, requires_grad=True),
        'computed_grad': torch.rand(3, 3, requires_grad=True)
    }
    
    # Add some gradients
    loss = data['computed_grad'].sum()
    loss.backward()
    
    # Save and load
    save_torch(obj=data, filepath=filepath)
    loaded_data = torch.load(filepath)
    
    # Verify gradient requirements are preserved
    assert loaded_data['no_grad'].requires_grad == False
    assert loaded_data['with_grad'].requires_grad == True
    assert loaded_data['computed_grad'].requires_grad == True


def test_save_torch_large_data(temp_dir):
    """Test saving large data structures."""
    filepath = os.path.join(temp_dir, "large_data.pt")
    
    # Create reasonably large data
    large_data = {
        'large_tensor': torch.rand(1000, 500),
        'multiple_tensors': [torch.rand(100, 100) for _ in range(10)],
        'nested_structure': {
            f'tensor_{i}': torch.rand(50, 50) for i in range(20)
        }
    }
    
    # Save and load
    save_torch(obj=large_data, filepath=filepath)
    loaded_data = torch.load(filepath)
    
    # Verify structure
    assert loaded_data['large_tensor'].shape == (1000, 500)
    assert len(loaded_data['multiple_tensors']) == 10
    assert len(loaded_data['nested_structure']) == 20
    
    # Verify a few specific tensors
    assert torch.equal(large_data['large_tensor'], loaded_data['large_tensor'])
    assert torch.equal(large_data['multiple_tensors'][0], loaded_data['multiple_tensors'][0])


def test_save_torch_various_shapes(temp_dir):
    """Test saving tensors with various shapes."""
    shapes = [
        (1,),           # 1D
        (5, 5),         # 2D square
        (3, 7),         # 2D rectangular
        (2, 3, 4),      # 3D
        (1, 1, 1, 1),   # 4D
        (2, 1, 3, 1, 5) # 5D
    ]
    
    for i, shape in enumerate(shapes):
        filepath = os.path.join(temp_dir, f"shape_{i}.pt")
        
        # Create tensor with specific shape
        original_tensor = torch.rand(shape)
        save_torch(obj=original_tensor, filepath=filepath)
        
        # Verify file was created and load back
        assert os.path.exists(filepath)
        loaded_tensor = torch.load(filepath)
        assert loaded_tensor.shape == shape
        assert torch.equal(original_tensor, loaded_tensor)


def test_save_torch_mixed_data_types(temp_dir):
    """Test saving files with mixed data types."""
    filepath = os.path.join(temp_dir, "mixed_types.pt")
    
    mixed_data = {
        'float_tensor': torch.rand(3, 3, dtype=torch.float32),
        'int_tensor': torch.randint(0, 10, (3, 3), dtype=torch.int64),
        'bool_tensor': torch.randint(0, 2, (3, 3), dtype=torch.bool),
        'string': 'test_string',
        'number': 42,
        'list': [1, 2, 3, 4, 5],
        'nested_dict': {
            'inner_tensor': torch.ones(2, 2),
            'inner_string': 'nested'
        }
    }
    
    # Save and load
    save_torch(obj=mixed_data, filepath=filepath)
    loaded_data = torch.load(filepath)
    
    # Verify all data types are preserved
    assert torch.equal(mixed_data['float_tensor'], loaded_data['float_tensor'])
    assert torch.equal(mixed_data['int_tensor'], loaded_data['int_tensor'])
    assert torch.equal(mixed_data['bool_tensor'], loaded_data['bool_tensor'])
    assert mixed_data['string'] == loaded_data['string']
    assert mixed_data['number'] == loaded_data['number']
    assert mixed_data['list'] == loaded_data['list']
    assert torch.equal(mixed_data['nested_dict']['inner_tensor'], 
                      loaded_data['nested_dict']['inner_tensor'])
    assert mixed_data['nested_dict']['inner_string'] == loaded_data['nested_dict']['inner_string']


def test_save_torch_device_preservation(temp_dir):
    """Test that device information is preserved during save."""
    # Test CPU tensor
    cpu_filepath = os.path.join(temp_dir, "cpu_tensor.pt")
    cpu_tensor = torch.rand(5, 5, device='cpu')
    
    save_torch(obj=cpu_tensor, filepath=cpu_filepath)
    loaded_cpu = torch.load(cpu_filepath)
    assert loaded_cpu.device.type == 'cpu'
    assert torch.equal(cpu_tensor, loaded_cpu)
    
    # Test CUDA tensor (if available)
    if torch.cuda.is_available():
        cuda_filepath = os.path.join(temp_dir, "cuda_tensor.pt")
        cuda_tensor = torch.rand(5, 5, device='cuda')
        
        save_torch(obj=cuda_tensor, filepath=cuda_filepath)
        loaded_cuda = torch.load(cuda_filepath)
        assert loaded_cuda.device.type == 'cuda'
        assert torch.equal(cuda_tensor, loaded_cuda)


def test_save_torch_concurrent_writes(temp_dir):
    """Test atomic writes prevent race conditions during concurrent saves."""
    import threading
    import time
    
    base_filepath = os.path.join(temp_dir, "concurrent")
    
    errors = []
    success_count = 0
    
    def save_worker(worker_id):
        nonlocal success_count
        try:
            filepath = f"{base_filepath}_{worker_id}.pt"
            data = {'worker_id': worker_id, 'tensor': torch.rand(10, 10)}
            save_torch(obj=data, filepath=filepath)
            success_count += 1
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads that save to different files
    threads = []
    for i in range(5):
        thread = threading.Thread(target=save_worker, args=(i,))
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert success_count == 5
    
    # Verify all files were created and are readable
    for i in range(5):
        filepath = f"{base_filepath}_{i}.pt"
        assert os.path.exists(filepath)
        data = torch.load(filepath)
        assert data['worker_id'] == i
        assert data['tensor'].shape == (10, 10)


def test_save_torch_error_handling(temp_dir):
    """Test error handling in save_torch."""
    # Test with invalid directory permissions (if possible)
    invalid_path = "/root/nonexistent/model.pt"  # Likely to fail on most systems
    
    with pytest.raises(RuntimeError, match="Error saving torch file"):
        save_torch(obj=torch.rand(2, 2), filepath=invalid_path)


def test_save_torch_file_permissions(temp_dir):
    """Test that saved files have correct permissions."""
    filepath = os.path.join(temp_dir, "permissions.pt")
    tensor = torch.rand(3, 3)
    
    save_torch(obj=tensor, filepath=filepath)
    
    # Verify file exists and is readable
    assert os.path.exists(filepath)
    assert os.access(filepath, os.R_OK)
    
    # Try to read the file to ensure it's valid
    loaded_tensor = torch.load(filepath)
    assert torch.equal(tensor, loaded_tensor)


def test_save_torch_integration_with_load_torch(temp_dir):
    """Test integration between save_torch and load_torch functions."""
    filepath = os.path.join(temp_dir, "integration.pt")
    
    # Create test data
    original_data = {
        'tensor': torch.rand(5, 5),
        'nested': {
            'inner_tensor': torch.randint(0, 10, (3, 3)),
            'value': 42
        }
    }
    
    # Save with save_torch
    save_torch(obj=original_data, filepath=filepath)
    
    # Load with load_torch
    loaded_data = load_torch(filepath=filepath)
    
    # Verify perfect round-trip
    assert torch.equal(original_data['tensor'], loaded_data['tensor'])
    assert torch.equal(original_data['nested']['inner_tensor'], 
                      loaded_data['nested']['inner_tensor'])
    assert original_data['nested']['value'] == loaded_data['nested']['value']


def test_save_torch_cleanup_on_error(temp_dir):
    """Test that temporary files are cleaned up on error."""
    import unittest.mock
    
    filepath = os.path.join(temp_dir, "cleanup_test.pt")
    tensor = torch.rand(3, 3)
    
    # Mock torch.save to raise an exception
    with unittest.mock.patch('torch.save', side_effect=RuntimeError("Mocked error")):
        with pytest.raises(RuntimeError, match="Error saving torch file"):
            save_torch(obj=tensor, filepath=filepath)
    
    # Verify no temporary files remain
    temp_files = [f for f in os.listdir(temp_dir) if f.startswith('torch_') and f.endswith('.tmp')]
    assert len(temp_files) == 0
    
    # Verify target file was not created
    assert not os.path.exists(filepath)
