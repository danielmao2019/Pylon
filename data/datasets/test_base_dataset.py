from typing import Any, Tuple, List, Dict, Union, Optional
import pytest
import torch
from .base_dataset import BaseDataset


class TestDataset(BaseDataset):

    SPLIT_OPTIONS = ['train', 'val', 'test', 'weird']
    INPUT_NAMES = ['input']
    LABEL_NAMES = ['label']

    def _init_annotations_(self, split: Optional[str]) -> None:
        # all splits are the same
        self.annotations = list(range(100))

    def _load_example_(self, idx: int) -> Tuple[
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
def test_base_dataset_all(
    indices: Optional[Union[List[int], Dict[str, List[int]]]],
    expected: Dict[str, List[int]],
) -> None:
    dataset = TestDataset(split=None, indices=indices)
    assert hasattr(dataset, 'split_subsets')
    for option in ['train', 'val', 'test', 'weird']:
        split_subset = dataset.split_subsets[option]
        assert list(split_subset[idx]['inputs']['input'] for idx in range(len(split_subset))) == expected[option]
        assert list(split_subset[idx]['labels']['label'] for idx in range(len(split_subset))) == expected[option]


@pytest.mark.parametrize("split, indices, expected", [
    (
        (0.8, 0.1, 0.1, 0.0), None,
        {'train': 80, 'val': 10, 'test': 10, 'weird': 0},
    ),
    (
        (0.7, 0.1, 0.1, 0.1), None,
        {'train': 70, 'val': 10, 'test': 10, 'weird': 10},
    ),
    (
        (0.8, 0.1, 0.1, 0.0), list(range(10)),
        {'train': 8, 'val': 1, 'test': 1, 'weird': 0},
    ),
])
def test_base_dataset_split(
    split: Optional[Union[str, Tuple[float, ...]]],
    indices: Optional[Union[List[int], Dict[str, List[int]]]],
    expected: Dict[str, int],
) -> None:
    dataset = TestDataset(split=split, indices=indices)
    assert hasattr(dataset, 'split_subsets')
    for option in ['train', 'val', 'test', 'weird']:
        assert len(dataset.split_subsets[option]) == expected[option], \
            f"{option=}, {len(dataset.split_subsets[option])=}, {expected[option]=}"
    assert set.intersection(*[set(
        (
            dataset.split_subsets[option][idx]['inputs']['input'],
            dataset.split_subsets[option][idx]['labels']['label'],
        ) for idx in range(len(dataset.split_subsets[option]))
    ) for option in ['train', 'val', 'test', 'weird']]) == set()
