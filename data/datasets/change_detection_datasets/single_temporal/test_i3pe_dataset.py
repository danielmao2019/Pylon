from .i3pe_dataset import I3PEDataset
from data.datasets import Bi2SingleTemporal, SYSU_CD_Dataset


def test_i3pe_dataset() -> None:
    source = Bi2SingleTemporal(SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split='train'))
    dataset = I3PEDataset(source=source, dataset_size=len(source), exchange_ratio=0.1)
    for idx in range(len(dataset)):
        _ = dataset[idx]
