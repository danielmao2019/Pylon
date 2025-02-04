import pytest
import torch
from data.datasets import GANDataset


@pytest.fixture
def source_dataset(request):
    """Creates a source dataset with a fixed latent dimension and dummy image data."""
    latent_dim = 128
    device = torch.device(request.param)
    
    # Create dummy image data for multiple datapoints
    num_samples = 5
    source = [{'inputs': {'image': torch.rand(3, 64, 64, device=device)}} for _ in range(num_samples)]

    return GANDataset(latent_dim=latent_dim, source=source, device=device)


@pytest.mark.parametrize("source_dataset", ["cpu", "cuda"], indirect=True)
def test_gan_dataset_properties(source_dataset):
    """Checks tensor shapes, dtypes, and device placement for all datapoints in the dataset."""
    for idx in range(len(source_dataset.source)):  # Iterate over all datapoints
        inputs, labels, _ = source_dataset._load_datapoint(idx)
        
        # Check tensor shapes
        assert inputs['z'].shape == (source_dataset.latent_dim,), f"Incorrect shape at idx {idx}"
        assert labels['image'].shape == (3, 64, 64), f"Incorrect image shape at idx {idx}"
        
        # Check tensor dtypes
        assert inputs['z'].dtype == torch.float32, f"Incorrect z dtype at idx {idx}"
        assert labels['image'].dtype == torch.float32, f"Incorrect image dtype at idx {idx}"
        
        # Check tensor device
        assert inputs['z'].device == source_dataset.device, f"Incorrect z device at idx {idx}"
        assert labels['image'].device == source_dataset.device, f"Incorrect image device at idx {idx}"


@pytest.mark.parametrize("source_dataset", ["cpu", "cuda"], indirect=True)
def test_reproducibility(source_dataset):
    """Checks that the dataset generates the same sample when the RNG state is restored."""
    for idx in range(len(source_dataset.source)):
        torch.manual_seed(42)  # Set a fixed seed
        inputs1, _, meta_info1 = source_dataset._load_datapoint(idx)

        # Restore RNG state and generate again
        torch.set_rng_state(meta_info1['cpu_rng_state'])
        torch.cuda.set_rng_state(meta_info1['gpu_rng_state'])  # No-op if CUDA is not available
        inputs2, _, _ = source_dataset._load_datapoint(idx)

        assert torch.allclose(inputs1['z'], inputs2['z']), f"Latent vector not reproducible at idx {idx}"
