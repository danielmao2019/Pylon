import os
import tempfile
import torch
import pytest
from utils.io.torch import load_torch


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_load_torch_tensor(temp_dir):
    """Test loading a simple torch tensor."""
    filepath = os.path.join(temp_dir, "test_tensor.pt")
    
    # Create and save test tensor
    original_tensor = torch.rand(10, 5)
    torch.save(original_tensor, filepath)
    
    # Load tensor back
    loaded_tensor = load_torch(filepath=filepath)
    
    # Verify tensors are identical
    assert torch.equal(original_tensor, loaded_tensor)
    assert original_tensor.shape == loaded_tensor.shape
    assert original_tensor.dtype == loaded_tensor.dtype


def test_load_torch_dict(temp_dir):
    """Test loading a dictionary of tensors."""
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
    torch.save(original_dict, filepath)
    
    # Load dictionary back
    loaded_dict = load_torch(filepath=filepath)
    
    # Verify contents
    assert torch.equal(original_dict['tensor1'], loaded_dict['tensor1'])
    assert torch.equal(original_dict['tensor2'], loaded_dict['tensor2'])
    assert torch.equal(original_dict['scalar'], loaded_dict['scalar'])
    assert original_dict['metadata'] == loaded_dict['metadata']


def test_load_torch_model_state_dict(temp_dir):
    """Test loading a model state dict."""
    filepath = os.path.join(temp_dir, "model_state.pt")
    
    # Create simple model
    model = torch.nn.Linear(10, 5)
    original_state_dict = model.state_dict()
    
    # Save state dict
    torch.save(original_state_dict, filepath)
    
    # Load state dict back
    loaded_state_dict = load_torch(filepath=filepath)
    
    # Verify all parameters match
    for key in original_state_dict:
        assert torch.equal(original_state_dict[key], loaded_state_dict[key])


def test_load_torch_with_map_location_cpu(temp_dir):
    """Test loading torch file with CPU map location."""
    filepath = os.path.join(temp_dir, "test_cpu.pt")
    
    # Create tensor (could be on any device)
    original_tensor = torch.rand(5, 5)
    torch.save(original_tensor, filepath)
    
    # Load with CPU map location
    loaded_tensor = load_torch(filepath=filepath, map_location='cpu')
    
    # Verify tensor is on CPU
    assert loaded_tensor.device.type == 'cpu'
    assert torch.equal(original_tensor.cpu(), loaded_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_torch_with_map_location_cuda(temp_dir):
    """Test loading torch file with CUDA map location."""
    filepath = os.path.join(temp_dir, "test_cuda.pt")
    
    # Create tensor on CPU
    original_tensor = torch.rand(5, 5)
    torch.save(original_tensor, filepath)
    
    # Load with CUDA map location
    loaded_tensor = load_torch(filepath=filepath, map_location='cuda')
    
    # Verify tensor is on CUDA
    assert loaded_tensor.device.type == 'cuda'
    assert torch.equal(original_tensor, loaded_tensor.cpu())


def test_load_torch_without_map_location(temp_dir):
    """Test loading torch file without specifying map location."""
    filepath = os.path.join(temp_dir, "test_default.pt")
    
    # Create and save tensor
    original_tensor = torch.rand(3, 3)
    torch.save(original_tensor, filepath)
    
    # Load without map_location
    loaded_tensor = load_torch(filepath=filepath)
    
    # Should work and preserve data
    assert torch.equal(original_tensor, loaded_tensor)


def test_load_torch_different_tensor_types(temp_dir):
    """Test loading different tensor types."""
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
        torch.save(original_tensor, filepath)
        loaded_tensor = load_torch(filepath=filepath)
        
        # Verify
        assert torch.equal(original_tensor, loaded_tensor)
        assert original_tensor.dtype == loaded_tensor.dtype


def test_load_torch_complex_nested_structure(temp_dir):
    """Test loading complex nested data structures."""
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
    torch.save(complex_data, filepath)
    loaded_data = load_torch(filepath=filepath)
    
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


def test_load_torch_empty_tensors(temp_dir):
    """Test loading empty tensors."""
    filepath = os.path.join(temp_dir, "empty_tensors.pt")
    
    empty_data = {
        'empty_1d': torch.empty(0),
        'empty_2d': torch.empty(0, 5),
        'empty_3d': torch.empty(0, 0, 0),
        'regular': torch.rand(2, 2)
    }
    
    # Save and load
    torch.save(empty_data, filepath)
    loaded_data = load_torch(filepath=filepath)
    
    # Verify shapes and types
    assert loaded_data['empty_1d'].shape == (0,)
    assert loaded_data['empty_2d'].shape == (0, 5)
    assert loaded_data['empty_3d'].shape == (0, 0, 0)
    assert torch.equal(empty_data['regular'], loaded_data['regular'])


def test_load_torch_with_requires_grad(temp_dir):
    """Test that gradient requirements are preserved."""
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
    torch.save(data, filepath)
    loaded_data = load_torch(filepath=filepath)
    
    # Verify gradient requirements are preserved
    assert loaded_data['no_grad'].requires_grad == False
    assert loaded_data['with_grad'].requires_grad == True
    assert loaded_data['computed_grad'].requires_grad == True
    
    # Note: gradients themselves are not preserved through save/load
    # This is expected PyTorch behavior


def test_load_torch_nonexistent_file():
    """Test error handling for non-existent files."""
    with pytest.raises(RuntimeError, match="File does not exist"):
        load_torch(filepath="nonexistent.pt")


def test_load_torch_empty_file(temp_dir):
    """Test error handling for empty files."""
    filepath = os.path.join(temp_dir, "empty.pt")
    
    # Create empty file
    open(filepath, 'w').close()
    
    with pytest.raises(RuntimeError, match="File is empty"):
        load_torch(filepath=filepath)


def test_load_torch_corrupted_file(temp_dir):
    """Test error handling for corrupted files."""
    filepath = os.path.join(temp_dir, "corrupted.pt")
    
    # Create file with invalid content
    with open(filepath, 'w') as f:
        f.write("This is not a valid PyTorch file")
    
    with pytest.raises(RuntimeError, match="Error loading torch file"):
        load_torch(filepath=filepath)


def test_load_torch_various_shapes(temp_dir):
    """Test loading tensors with various shapes."""
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
        torch.save(original_tensor, filepath)
        
        # Load and verify
        loaded_tensor = load_torch(filepath=filepath)
        assert loaded_tensor.shape == shape
        assert torch.equal(original_tensor, loaded_tensor)


def test_load_torch_large_tensors(temp_dir):
    """Test loading reasonably large tensors."""
    filepath = os.path.join(temp_dir, "large_tensor.pt")
    
    # Create a reasonably large tensor (not too large to avoid test timeouts)
    large_tensor = torch.rand(1000, 1000)
    torch.save(large_tensor, filepath)
    
    # Load and verify
    loaded_tensor = load_torch(filepath=filepath)
    
    assert loaded_tensor.shape == (1000, 1000)
    assert torch.equal(large_tensor, loaded_tensor)


def test_load_torch_mixed_data_types(temp_dir):
    """Test loading files with mixed data types."""
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
    torch.save(mixed_data, filepath)
    loaded_data = load_torch(filepath=filepath)
    
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


def test_load_torch_multiple_devices(temp_dir):
    """Test loading tensors that were saved from different devices."""
    # Test CPU tensor
    cpu_filepath = os.path.join(temp_dir, "cpu_tensor.pt")
    cpu_tensor = torch.rand(5, 5, device='cpu')
    torch.save(cpu_tensor, cpu_filepath)
    
    loaded_cpu = load_torch(filepath=cpu_filepath)
    assert loaded_cpu.device.type == 'cpu'
    assert torch.equal(cpu_tensor, loaded_cpu)
    
    # Test CUDA tensor (if available)
    if torch.cuda.is_available():
        cuda_filepath = os.path.join(temp_dir, "cuda_tensor.pt")
        cuda_tensor = torch.rand(5, 5, device='cuda')
        torch.save(cuda_tensor, cuda_filepath)
        
        # Load without map_location (should preserve device)
        loaded_cuda = load_torch(filepath=cuda_filepath)
        assert loaded_cuda.device.type == 'cuda'
        assert torch.equal(cuda_tensor, loaded_cuda)
        
        # Load with CPU map_location
        loaded_cuda_to_cpu = load_torch(filepath=cuda_filepath, map_location='cpu')
        assert loaded_cuda_to_cpu.device.type == 'cpu'
        assert torch.equal(cuda_tensor.cpu(), loaded_cuda_to_cpu)


def test_load_torch_round_trip_consistency(temp_dir):
    """Test multiple save/load cycles maintain consistency."""
    filepath = os.path.join(temp_dir, "round_trip.pt")
    
    # Start with original data
    original_data = {
        'tensor': torch.rand(10, 10),
        'int_tensor': torch.randint(0, 100, (5, 5)),
        'metadata': {'iteration': 0}
    }
    
    current_data = original_data
    
    # Perform multiple save/load cycles
    for i in range(3):
        # Update metadata to track iterations
        current_data['metadata']['iteration'] = i
        
        # Save and load
        torch.save(current_data, filepath)
        current_data = load_torch(filepath=filepath)
        
        # Verify core tensors remain unchanged
        assert torch.equal(original_data['tensor'], current_data['tensor'])
        assert torch.equal(original_data['int_tensor'], current_data['int_tensor'])
        assert current_data['metadata']['iteration'] == i


def test_load_torch_concurrent_access(temp_dir):
    """Test concurrent loading of the same file."""
    import threading
    
    filepath = os.path.join(temp_dir, "concurrent.pt")
    test_data = {'tensor': torch.rand(20, 20), 'value': 42}
    torch.save(test_data, filepath)
    
    results = []
    errors = []
    
    def load_worker():
        try:
            result = load_torch(filepath=filepath)
            results.append(result)
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads that load the same file
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=load_worker)
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 5
    
    # All results should be identical
    for result in results:
        assert torch.equal(test_data['tensor'], result['tensor'])
        assert test_data['value'] == result['value']
