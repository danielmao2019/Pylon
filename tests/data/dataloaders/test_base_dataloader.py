import pytest
from .base_dataloader import BaseDataLoader
import torch


class TestDataset(torch.utils.data.Dataset):
    """A simple dataset returning indices as data for easy verification."""
    def __init__(self, size):
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.mark.parametrize("dataset_size, batch_size, last_mode, expected_num_batches, expected_last_batch_size", [
    (14, 32, 'fill', 1, 32),  # One batch, filled to 32
    (14, 32, 'drop', 0, 0),   # No batches (dropped because incomplete)
    (14, 32, 'keep', 1, 14),  # One batch, size 14 (kept as is)
    (32, 32, 'fill', 1, 32),  # One perfect batch (no filling needed)
    (32, 32, 'drop', 1, 32),  # One perfect batch
    (32, 32, 'keep', 1, 32),  # One perfect batch
    (50, 32, 'fill', 2, 32),  # Two batches, second batch filled
    (50, 32, 'drop', 1, 32),  # Only one complete batch, second dropped
    (50, 32, 'keep', 2, 18),  # Second batch has 18 samples
    (5, 32, 'fill', 1, 32),   # Only one batch, filled to 32
    (5, 32, 'drop', 0, 0),    # No batches (dropped because incomplete)
    (5, 32, 'keep', 1, 5),    # One batch, size 5 (kept as is)
])
def test_base_dataloader(dataset_size, batch_size, last_mode, expected_num_batches, expected_last_batch_size):
    dataset = TestDataset(size=dataset_size)
    dataloader = BaseDataLoader(dataset, batch_size=batch_size, last_mode=last_mode)

    batches = list(dataloader)
    assert len(batches) == expected_num_batches, f"Expected {expected_num_batches} batches, got {len(batches)}"

    if batches:
        last_batch = batches[-1]
        assert len(last_batch) == expected_last_batch_size, f"Expected last batch size {expected_last_batch_size}, got {len(last_batch)}"


@pytest.mark.parametrize("shuffle", [True, False])
def test_dataloader_shuffle_behavior(shuffle):
    dataset = TestDataset(size=50)
    dataloader = BaseDataLoader(dataset, batch_size=10, shuffle=shuffle)

    batches1 = list(dataloader)
    batches2 = list(BaseDataLoader(dataset, batch_size=10, shuffle=shuffle))

    if shuffle:
        assert not all(torch.equal(b1, b2) for b1, b2 in zip(batches1, batches2)), "Shuffled batches should not be identical"
    else:
        assert all(torch.equal(b1, b2) for b1, b2 in zip(batches1, batches2)), "Non-shuffled batches should be identical"


def test_dataloader_with_empty_dataset():
    dataset = TestDataset(size=0)
    dataloader = BaseDataLoader(dataset, batch_size=10, last_mode='fill')
    assert len(list(dataloader)) == 0, "Dataloader should yield no batches for an empty dataset"
