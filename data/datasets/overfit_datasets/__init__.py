"""
DATA.DATASETS.OVERFIT_DATASETS API
"""
from data.datasets.overfit_datasets.overfit_base_dataset import OverfitBaseDataset
from data.datasets.overfit_datasets.semantic_segmentation import SemanticSegmentationOverfitDataset


__all__ = (
    'OverfitBaseDataset',
    'SemanticSegmentationOverfitDataset',
)
