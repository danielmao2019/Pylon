import pytest
from .i3pe_dataset import I3PEDataset


def test_i3pe_dataset() -> None:
    dataset = I3PEDataset(
        source=
        dataset_size=
        exchange_ratio=0.1,
    )
    for idx in range(len(dataset)):
        datapoint = dataset[idx]
