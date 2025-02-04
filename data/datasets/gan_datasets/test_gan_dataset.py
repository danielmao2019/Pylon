import pytest
from .gan_dataset import GANDataset
import torch
import data


@pytest.fixture
def source_dataset(request):
    """Fixture for creating an MNISTDataset instance with specified split and device."""
    split, device = request.param  # Unpack the parameterized values
    latent_dim = 128
    
    # Create dummy image data for multiple datapoints
    source = data.datasets.MNISTDataset(
        data_root="./data/datasets/soft_links/MNIST",
        split=split,
    )
    return GANDataset(latent_dim=latent_dim, source=source, device=device)


@pytest.mark.parametrize("mnist_dataset", [
    ("train", "cpu"),
    ("test", "cpu"),
    ("train", "cuda"),
    ("test", "cuda"),
], indirect=True)
def test_gan_dataset_properties(mnist_dataset):
    """Checks tensor shapes, dtypes, and device placement for all datapoints in the dataset."""
    for idx in range(len(mnist_dataset.source)):  # Loop through entire dataset
        inputs, labels, meta_info = mnist_dataset._load_datapoint(idx)

        # Shape checks
        assert inputs['image'].shape == (1, 28, 28), f"Incorrect image shape at idx {idx}"
        assert labels['label'].shape == (), f"Incorrect label shape at idx {idx}"

        # Dtype checks
        assert inputs['image'].dtype == torch.float32, f"Incorrect image dtype at idx {idx}"
        assert labels['label'].dtype == torch.int64, f"Incorrect label dtype at idx {idx}"

        # Device checks
        assert inputs['image'].device == mnist_dataset.device, f"Image not on correct device at idx {idx}"
        assert labels['label'].device == mnist_dataset.device, f"Label not on correct device at idx {idx}"

        # Meta info check
        assert meta_info['index'] == idx, f"Meta info index mism atch at idx {idx}"


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
