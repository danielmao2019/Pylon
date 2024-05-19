"""
DATA.DATASETS.RANDOM_DATASETS API
"""
from data.datasets.random_datasets.base_random_dataset import BaseRandomDataset
from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset
from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset


__all__ = (
    'BaseRandomDataset',
    'ClassificationRandomDataset',
    'SemanticSegmentationRandomDataset',
)
