"""
DATA.DATASETS API
"""
from data.datasets.base_dataset import BaseDataset
from data.datasets import random_datasets
from data.datasets.projection_dataset_wrapper import ProjectionDatasetWrapper

# Multi-Task Learning datasets
from data.datasets.multi_task_datasets.multi_mnist_dataset import MultiMNISTDataset
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset
from data.datasets.multi_task_datasets.multi_task_facial_landmark_dataset import MultiTaskFacialLandmarkDataset
from data.datasets.multi_task_datasets.city_scapes_dataset import CityScapesDataset
from data.datasets.multi_task_datasets.nyu_v2_dataset import NYUv2Dataset

# Change Detection datasets
from data.datasets.change_detection_datasets.oscd_dataset import OSCDDataset
from data.datasets.change_detection_datasets.levir_cd_dataset import LevirCdDataset
from data.datasets.change_detection_datasets.xview2_dataset import xView2Dataset
from data.datasets.change_detection_datasets.kc_3d_dataset import KC3DDataset
from data.datasets.change_detection_datasets.air_change_dataset import AirChangeDataset


__all__ = (
    'BaseDataset',
    'random_datasets',
    'ProjectionDatasetWrapper',
    # Multi-Task Learning datasets
    'MultiMNISTDataset',
    'CelebADataset',
    'MultiTaskFacialLandmarkDataset',
    'CityScapesDataset',
    'NYUv2Dataset',
    # Change Detection datasets
    'OSCDDataset',
    'LevirCdDataset',
    'xView2Dataset',
    'KC3DDataset',
    'AirChangeDataset',
)
