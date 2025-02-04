import pytest
import torch
from data.datasets import MNISTDataset
from .gan_dataset import GANDataset


@pytest.fixture
def source_dataset(request):
    """Fixture for creating a GANDataset instance with an MNIST source dataset."""
    split, device = request.param  # Unpack the test parameters
    latent_dim = 128

    # Load MNIST dataset as the source
    source = MNISTDataset(
        data_root="./data/datasets/soft_links/MNIST",
        split=split,
    )

    return GANDataset(latent_dim=latent_dim, source=source, device=device)


@pytest.mark.parametrize("source_dataset", [
    ("train", "cpu"),
    ("test", "cpu"),
    ("train", "cuda"),
    ("test", "cuda"),
], indirect=True)
def test_gan_dataset_properties(source_dataset):
    """Checks tensor shapes, dtypes, and device placement for all datapoints in the GANDataset."""
    for idx in range(len(source_dataset.source)):  # Loop through the entire dataset
        inputs, labels, meta_info = source_dataset._load_datapoint(idx)

        # Shape checks
        assert inputs['z'].shape == (source_dataset.latent_dim,), f"Incorrect latent vector shape at idx {idx}"
        assert labels['image'].shape == (1, 28, 28), f"Incorrect image shape at idx {idx}"

        # Dtype checks
        assert inputs['z'].dtype == torch.float32, f"Incorrect latent vector dtype at idx {idx}"
        assert labels['image'].dtype == torch.float32, f"Incorrect image dtype at idx {idx}"

        # Device checks
        assert inputs['z'].device == source_dataset.device, f"Latent vector not on correct device at idx {idx}"
        assert labels['image'].device == source_dataset.device, f"Image not on correct device at idx {idx}"

        # Meta info check
        assert "cpu_rng_state" in meta_info, f"Missing cpu_rng_state in meta_info at idx {idx}"
        assert "gpu_rng_state" in meta_info, f"Missing gpu_rng_state in meta_info at idx {idx}"


@pytest.mark.parametrize("source_dataset", [
    ("train", "cpu"),
    ("train", "cuda"),
], indirect=True)
def test_reproducibility(source_dataset):
    """Checks that the dataset generates the same sample when the RNG state is restored."""
    for idx in range(len(source_dataset.source)):
        torch.manual_seed(42)  # Set a fixed seed
        inputs1, _, meta_info1 = source_dataset._load_datapoint(idx)

        # Restore RNG state and generate again
        torch.set_rng_state(meta_info1['cpu_rng_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(meta_info1['gpu_rng_state'])
            
        inputs2, _, _ = source_dataset._load_datapoint(idx)

        assert torch.allclose(inputs1['z'], inputs2['z']), f"Latent vector not reproducible at idx {idx}"
