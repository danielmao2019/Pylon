from typing import Dict, Any
import pytest
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets import MNISTDataset
from data.datasets.gan_datasets.gan_dataset import GANDataset


def validate_inputs(inputs: Dict[str, Any], dataset) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert inputs['z'].shape == (dataset.latent_dim,), f"Incorrect latent vector shape"
    assert inputs['z'].dtype == torch.float32, f"Incorrect latent vector dtype"
    assert inputs['z'].device.type == dataset.device.type, f"{inputs['z'].device=}, {dataset.device=}"


def validate_labels(labels: Dict[str, Any], dataset) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert labels['image'].shape == (1, 28, 28), f"Incorrect image shape"
    assert labels['image'].dtype == torch.float32, f"Incorrect image dtype"
    assert labels['image'].device.type == dataset.device.type, f"{labels['image'].device=}, {dataset.device=}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert "cpu_rng_state" in meta_info, f"Missing cpu_rng_state in meta_info"
    assert "gpu_rng_state" in meta_info, f"Missing gpu_rng_state in meta_info"
    assert 'idx' in meta_info, f"meta_info should contain 'idx' key: {meta_info.keys()=}"
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] should match datapoint index: {meta_info['idx']=}, {datapoint_idx=}"


@pytest.fixture
def dataset(request, mnist_data_root):
    """Fixture for creating a GANDataset instance with an MNIST source dataset."""
    split, device = request.param  # Unpack the test parameters
    latent_dim = 128

    # Use MNIST data root fixture
    source = MNISTDataset(
        data_root=mnist_data_root,
        split=split, 
        device=device,
    )
    
    dataset = GANDataset(latent_dim=latent_dim, source=source, device=device)
    return dataset


@pytest.mark.parametrize("dataset", [
    ("train", "cpu"),
    ("test", "cpu"),
    ("train", "cuda"),
    ("test", "cuda"),
], indirect=True)
def test_gan_dataset_properties(dataset, max_samples, get_samples_to_test):
    """Checks tensor shapes, dtypes, and device placement for all datapoints in the GANDataset."""
    assert isinstance(dataset, torch.utils.data.Dataset), f"Expected torch.utils.data.Dataset, got {type(dataset)}"
    assert len(dataset) > 0, "Dataset should not be empty"

    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'], dataset)
        validate_labels(datapoint['labels'], dataset)
        validate_meta_info(datapoint['meta_info'], idx)

    num_samples = get_samples_to_test(len(dataset.source), max_samples)
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)


@pytest.mark.parametrize("dataset", [
    ("train", "cpu"),
    ("train", "cuda"),
], indirect=True)
def test_reproducibility(dataset, max_samples, get_samples_to_test):
    """Checks that the dataset generates the same sample when the RNG state is restored."""
    assert isinstance(dataset, torch.utils.data.Dataset), f"Expected torch.utils.data.Dataset, got {type(dataset)}"
    assert len(dataset) > 0, "Dataset should not be empty"

    def test_reproducibility_for_idx(idx: int) -> None:
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

    num_samples = get_samples_to_test(len(dataset.source), max_samples)
    indices = list(range(num_samples))
    with ThreadPoolExecutor() as executor:
        executor.map(test_reproducibility_for_idx, indices)
