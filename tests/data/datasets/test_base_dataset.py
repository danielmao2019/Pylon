from typing import Any, Tuple, List, Dict, Optional
import pytest
import torch
from data.datasets.base_dataset import BaseDataset
from utils.ops import buffer_equal


class TestDataset(BaseDataset):

    SPLIT_OPTIONS = ['train', 'val', 'test', 'weird']
    INPUT_NAMES = ['input']
    LABEL_NAMES = ['label']

    def _init_annotations(self) -> None:
        # all splits are the same
        self.annotations = list(range(100))

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        return {'input': self.annotations[idx]}, {'label': self.annotations[idx]}, {}


@pytest.mark.parametrize("indices, expected", [
    (
        None,
        {'train': list(range(100)), 'val': list(range(100)), 'test': list(range(100)), 'weird': list(range(100))},
    ),
    (
        {'train': [2, 4, 6, 8]},
        {'train': [2, 4, 6, 8], 'val': list(range(100)), 'test': list(range(100)), 'weird': list(range(100))},
    ),
    (
        {'train': [2], 'val': [2, 4], 'test': [2, 4, 6], 'weird': [2, 4, 6, 8]},
        {'train': [2], 'val': [2, 4], 'test': [2, 4, 6], 'weird': [2, 4, 6, 8]},
    ),
])
def test_base_dataset_None(
    indices: Optional[Dict[str, List[int]]],
    expected: Dict[str, List[int]],
) -> None:
    dataset = TestDataset(split=None, indices=indices)
    assert dataset.split is None and not hasattr(dataset, 'split_percentage')
    assert not hasattr(dataset, 'indices') and hasattr(dataset, 'split_indices')
    assert hasattr(dataset, 'split_subsets')
    assert list(dataset.split_subsets.keys()) == ['train', 'val', 'test', 'weird']
    for split in ['train', 'val', 'test', 'weird']:
        split_subset = dataset.split_subsets[split]
        assert split_subset.split == split, f"{split_subset.split=}, {split=}"
        assert split_subset.indices == (indices.get(split, None) if indices else None)
        assert not hasattr(split_subset, 'split_indices')
        assert not hasattr(split_subset, 'split_subsets')
        assert list(datapoint['inputs']['input'] for datapoint in split_subset) == expected[split]
        assert list(datapoint['labels']['label'] for datapoint in split_subset) == expected[split]


@pytest.mark.parametrize("split, expected", [
    (
        (0.8, 0.1, 0.1, 0.0),
        {'train': 80, 'val': 10, 'test': 10, 'weird': 0},
    ),
    (
        (0.7, 0.1, 0.1, 0.1),
        {'train': 70, 'val': 10, 'test': 10, 'weird': 10},
    ),
])
def test_base_dataset_tuple(
    split: Tuple[float, ...],
    expected: Dict[str, int],
) -> None:
    dataset = TestDataset(split=split, indices=None)
    assert not hasattr(dataset, 'split') and type(dataset.split_percentages) == tuple
    assert not hasattr(dataset, 'indices') and not hasattr(dataset, 'split_indices')
    assert hasattr(dataset, 'split_subsets')
    assert list(dataset.split_subsets.keys()) == ['train', 'val', 'test', 'weird']
    for split in ['train', 'val', 'test', 'weird']:
        split_subset = dataset.split_subsets[split]
        assert split_subset.split == split, f"{split_subset.split=}, {split=}"
        assert not hasattr(split_subset, 'split_percentages')
        assert not hasattr(split_subset, 'indices') and not hasattr(split_subset, 'split_indices')
        assert not hasattr(split_subset, 'split_subsets')
        assert len(split_subset) == expected[split], \
            f"{split=}, {len(split_subset)=}, {expected[split]=}"
    assert set.intersection(*[set(
        (
            dataset.split_subsets[split][idx]['inputs']['input'],
            dataset.split_subsets[split][idx]['labels']['label'],
        ) for idx in range(len(dataset.split_subsets[split]))
    ) for split in ['train', 'val', 'test', 'weird']]) == set()


def test_base_dataset_cache():
    # check dataset
    dataset_cached = TestDataset(split='train', indices=list(range(10)), use_cache=True)
    dataset = TestDataset(split='train', indices=list(range(10)), use_cache=False)
    assert len(dataset_cached) == len(dataset) == 10
    for _ in range(2):
        for idx in range(10):
            assert dataset_cached[idx] == dataset[idx]
    # check dataloader
    dataloader_cached = torch.utils.data.DataLoader(
        dataset=dataset_cached, shuffle=False, batch_size=2,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=False, batch_size=2,
    )
    assert len(dataloader_cached) == len(dataloader)
    for _ in range(2):
        for dp_cached, dp in zip(dataloader_cached, dataloader):
            assert buffer_equal(dp_cached, dp)
