"""Test cases for determinism utilities."""
import torch
import pytest
from utils.determinism import set_seed, set_determinism


def test_set_seed():
    """Test that set_seed produces deterministic results."""
    # Test 1: Same seed should produce same random numbers
    set_seed(42)
    tensor1 = torch.randn(10)
    tensor2 = torch.randn(10)
    
    set_seed(42)
    tensor3 = torch.randn(10)
    tensor4 = torch.randn(10)
    
    assert torch.allclose(tensor1, tensor3), "First random tensor should be identical with same seed"
    assert torch.allclose(tensor2, tensor4), "Second random tensor should be identical with same seed"
    
    # Test 2: Different seeds should produce different random numbers
    set_seed(42)
    tensor5 = torch.randn(10)
    
    set_seed(43)
    tensor6 = torch.randn(10)
    
    assert not torch.allclose(tensor5, tensor6), "Random tensors should be different with different seeds"


def test_set_determinism():
    """Test that set_determinism produces deterministic results."""
    # Test 1: Same sequence of operations should produce same results
    set_determinism()
    set_seed(42)
    
    # Create a model and some data
    model = torch.nn.Linear(10, 1)
    data = torch.randn(32, 10)
    
    # First forward pass
    output1 = model(data)
    loss1 = output1.mean()
    loss1.backward()
    
    # Reset everything
    set_determinism()
    set_seed(42)
    
    # Create same model and data
    model2 = torch.nn.Linear(10, 1)
    data2 = torch.randn(32, 10)
    
    # Second forward pass
    output2 = model2(data2)
    loss2 = output2.mean()
    loss2.backward()
    
    # Check results
    assert torch.allclose(data, data2), "Input data should be identical"
    assert torch.allclose(model.weight, model2.weight), "Model weights should be identical"
    assert torch.allclose(model.bias, model2.bias), "Model biases should be identical"
    assert torch.allclose(output1, output2), "Outputs should be identical"
    assert torch.allclose(loss1, loss2), "Losses should be identical"
    assert torch.allclose(model.weight.grad, model2.weight.grad), "Weight gradients should be identical"
    assert torch.allclose(model.bias.grad, model2.bias.grad), "Bias gradients should be identical"


def test_deterministic_dataloader():
    """Test that dataloaders produce deterministic results with set_seed."""
    # Create a simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.data = torch.randn(size, 10)
            self.labels = torch.randn(size, 1)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return {'inputs': self.data[idx], 'labels': self.labels[idx]}
    
    # Test 1: Same seed should produce same data order
    set_seed(42)
    dataset1 = SimpleDataset()
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=32, shuffle=True)
    batch1 = next(iter(dataloader1))
    
    set_seed(42)
    dataset2 = SimpleDataset()
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=True)
    batch2 = next(iter(dataloader2))
    
    assert torch.allclose(batch1['inputs'], batch2['inputs']), "Dataloader should produce same data order with same seed"
    assert torch.allclose(batch1['labels'], batch2['labels']), "Dataloader should produce same labels order with same seed"
    
    # Test 2: Different seeds should produce different data order
    set_seed(42)
    dataset3 = SimpleDataset()
    dataloader3 = torch.utils.data.DataLoader(dataset3, batch_size=32, shuffle=True)
    batch3 = next(iter(dataloader3))
    
    set_seed(43)
    dataset4 = SimpleDataset()
    dataloader4 = torch.utils.data.DataLoader(dataset4, batch_size=32, shuffle=True)
    batch4 = next(iter(dataloader4))
    
    assert not torch.allclose(batch3['inputs'], batch4['inputs']), "Dataloader should produce different data order with different seeds"
    assert not torch.allclose(batch3['labels'], batch4['labels']), "Dataloader should produce different labels order with different seeds"
