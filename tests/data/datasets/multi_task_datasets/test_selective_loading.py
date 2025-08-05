import pytest
import tempfile
import torch
from data.datasets.multi_task_datasets.multi_mnist_dataset import MultiMNISTDataset


@pytest.mark.parametrize("selected_labels,expected_keys", [
    (['left'], ['left']),
    (['right'], ['right']),
    (['left', 'right'], ['left', 'right']),
    (None, ['left', 'right']),  # Default case
])
def test_selective_label_loading(selected_labels, expected_keys):
    """Test that only selected labels are loaded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MultiMNISTDataset(
            data_root=temp_dir,
            split="train",
            labels=selected_labels
        )
        
        # Check selected_labels attribute
        if selected_labels is None:
            assert dataset.selected_labels == dataset.LABEL_NAMES
        else:
            assert dataset.selected_labels == selected_labels
        
        # Load a datapoint and check only expected labels are present
        datapoint = dataset[0]
        
        assert isinstance(datapoint, dict)
        assert 'labels' in datapoint
        
        labels = datapoint['labels']
        assert isinstance(labels, dict)
        assert set(labels.keys()) == set(expected_keys)
        
        # Verify label values are tensors with correct types
        for key in expected_keys:
            assert isinstance(labels[key], torch.Tensor)
            assert labels[key].dtype == torch.int64


def test_selective_loading_preserves_values():
    """Test that selective loading produces same values as full loading."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load all labels
        dataset_full = MultiMNISTDataset(
            data_root=temp_dir,
            split="train"
        )
        
        # Load only left label
        dataset_left = MultiMNISTDataset(
            data_root=temp_dir,
            split="train",
            labels=["left"]
        )
        
        # Same index should produce same values
        datapoint_full = dataset_full[0]
        datapoint_left = dataset_left[0]
        
        inputs_full = datapoint_full['inputs']
        labels_full = datapoint_full['labels']
        inputs_left = datapoint_left['inputs']
        labels_left = datapoint_left['labels']
        
        # Images should be identical
        assert torch.equal(inputs_full['image'], inputs_left['image'])
        
        # Left label should be identical
        assert torch.equal(labels_full['left'], labels_left['left'])
        
        # Right label should not be present in selective dataset
        assert 'right' not in labels_left


def test_cache_version_discrimination():
    """Test that different selected labels produce different cache versions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Different labels should have DIFFERENT cache versions since we now
        # avoid disk I/O by loading only selected labels
        dataset1 = MultiMNISTDataset(
            data_root=temp_dir,
            split="train",
            labels=["left"]
        )
        
        dataset2 = MultiMNISTDataset(
            data_root=temp_dir,
            split="train",
            labels=["right"]
        )
        
        dataset3 = MultiMNISTDataset(
            data_root=temp_dir,
            split="train",
            labels=["left", "right"]
        )
        
        # All should have different cache version hashes
        hash1 = dataset1.get_cache_version_hash()
        hash2 = dataset2.get_cache_version_hash()
        hash3 = dataset3.get_cache_version_hash()
        
        assert hash1 != hash2
        assert hash2 != hash3
        assert hash1 != hash3


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_invalid_labels_not_in_label_names():
    """Test that invalid label names raise AssertionError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(AssertionError) as exc_info:
            MultiMNISTDataset(
                data_root=temp_dir,
                split="train",
                labels=["invalid_label"]
            )
        assert "not in LABEL_NAMES" in str(exc_info.value)


def test_empty_labels_list():
    """Test that empty labels list raises AssertionError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(AssertionError) as exc_info:
            MultiMNISTDataset(
                data_root=temp_dir,
                split="train",
                labels=[]
            )
        assert "labels list must not be empty" in str(exc_info.value)


def test_non_list_labels():
    """Test that non-list labels raise AssertionError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(AssertionError) as exc_info:
            MultiMNISTDataset(
                data_root=temp_dir,
                split="train",
                labels="left"  # String instead of list
            )
        assert "labels must be list" in str(exc_info.value)


def test_non_string_elements_in_labels():
    """Test that non-string elements in labels list raise AssertionError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(AssertionError) as exc_info:
            MultiMNISTDataset(
                data_root=temp_dir,
                split="train",
                labels=[1, 2]  # Integers instead of strings
            )
        assert "All labels must be strings" in str(exc_info.value)
