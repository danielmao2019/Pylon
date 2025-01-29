from .cdrl_dataset import CDRLDataset
from data.datasets import Bi2SingleTemporal, LevirCdDataset


def test_cdrl_dataset() -> None:
    source = Bi2SingleTemporal(LevirCdDataset(data_root="./data/datasets/soft_links/LEVIR_CD", split='train'))
    transform_12 = 
    transform_21 = 
    dataset = CDRLDataset(
        source=source,
        dataset_size=len(source),
        transform_12=transform_12,
        transform_21=transform_21,
    )
