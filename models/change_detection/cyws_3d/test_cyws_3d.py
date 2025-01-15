import pytest
from .cyws_3d import CYWS3D
import data


def test_cyws_3d():
    model = CYWS3D()
    dataset = data.datasets.KC3DDataset(
        data_root="./data/datasets/soft_links/KC3D",
        split='train', use_ground_truth_registration=True,
    )
    for idx in range(3):
        datapoint = dataset[idx]
        _ = model(datapoint['inputs'])
