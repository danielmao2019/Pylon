import pytest
import torch
from data.datasets import MNISTDataset
from .gan_dataset import GANDataset


@pytest.fixture
def dataset(request):
    """Fixture for creating a GANDataset instance with an MNIST source dataset."""
    split, device = request.param  # Unpack the test parameters
    latent_dim = 128

    # Load MNIST dataset as the source
    source = MNISTDataset(
        data_root="./data/datasets/soft_links/MNIST",
        split=split, device=device,
    )

    return GANDataset(latent_dim=latent_dim, source=source, device=device)


@pytest.mark.parametrize("dataset", [
    ("train", "cpu"),
    ("test", "cpu"),
    ("train", "cuda"),
    ("test", "cuda"),
], indirect=True)
def test_gan_dataset_properties(dataset):
    """Checks tensor shapes, dtypes, and device placement for all datapoints in the GANDataset."""
    for idx in range(len(dataset.source)):  # Loop through the entire dataset
        datapoint = dataset[idx]
        inputs, labels, meta_info = datapoint['inputs'], datapoint['labels'], datapoint['meta_info']

        # Shape checks
        assert inputs['z'].shape == (dataset.latent_dim,), f"Incorrect latent vector shape at idx {idx}"
        assert labels['image'].shape == (1, 28, 28), f"Incorrect image shape at idx {idx}"

        # Dtype checks
        assert inputs['z'].dtype == torch.float32, f"Incorrect latent vector dtype at idx {idx}"
        assert labels['image'].dtype == torch.float32, f"Incorrect image dtype at idx {idx}"

        # Device checks
        assert inputs['z'].device.type == dataset.device.type, f"{inputs['z'].device=}, {dataset.device=}, {idx=}"
        assert labels['image'].device.type == dataset.device.type, f"{labels['image'].device=}, {dataset.device=}, {idx=}"

        # Meta info check
        assert "cpu_rng_state" in meta_info, f"Missing cpu_rng_state in meta_info at idx {idx}"
        assert "gpu_rng_state" in meta_info, f"Missing gpu_rng_state in meta_info at idx {idx}"


@pytest.mark.parametrize("dataset", [
    ("train", "cpu"),
    ("train", "cuda"),
], indirect=True)
def test_reproducibility(dataset):
    """Checks that the dataset generates the same sample when the RNG state is restored."""
    for idx in range(len(dataset.source)):
        torch.manual_seed(42)  # Set a fixed seed
        datapoint = dataset[idx]
        inputs1, meta_info1 = datapoint['inputs'], datapoint['meta_info']

        # Restore RNG state and generate again
        torch.set_rng_state(meta_info1['cpu_rng_state'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(meta_info1['gpu_rng_state'].cpu())

        datapoint = dataset[idx]
        inputs2 = datapoint['inputs']

        assert torch.allclose(inputs1['z'], inputs2['z']), f"Latent vector not reproducible at idx {idx}"
