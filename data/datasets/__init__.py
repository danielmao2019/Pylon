"""
DATA.DATASETS API
"""
from data.datasets.base_dataset import BaseDataset
from data.datasets.base_synthetic_dataset import BaseSyntheticDataset
from data.datasets import random_datasets
from data.datasets.projection_dataset_wrapper import ProjectionDatasetWrapper

# Semantic Segmentation datasets
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset

# GAN datasets
from data.datasets.gan_datasets.gan_dataset import GANDataset

# Multi-Task Learning datasets
from data.datasets.multi_task_datasets.multi_mnist_dataset import MultiMNISTDataset
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset
from data.datasets.multi_task_datasets.multi_task_facial_landmark_dataset import MultiTaskFacialLandmarkDataset
from data.datasets.multi_task_datasets.city_scapes_dataset import CityScapesDataset
from data.datasets.multi_task_datasets.nyu_v2_dataset import NYUv2Dataset

# Change Detection datasets
## Bi-Temporal
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset
## Single-Temporal
from data.datasets.change_detection_datasets.single_temporal.bi2single_temporal_dataset import Bi2SingleTemporal
from data.datasets.change_detection_datasets.single_temporal.i3pe_dataset import I3PEDataset
from data.datasets.change_detection_datasets.single_temporal.ppsl_dataset import PPSLDataset


__all__ = (
    'BaseDataset',
    'BaseSyntheticDataset',
    'random_datasets',
    'ProjectionDatasetWrapper',
    # Semantic Segmentation datasets
    'WHU_BD_Dataset',
    # GAN datasets
    'GANDataset',
    # Multi-Task Learning datasets
    'MultiMNISTDataset',
    'CelebADataset',
    'MultiTaskFacialLandmarkDataset',
    'CityScapesDataset',
    'NYUv2Dataset',
    # Change Detection datasets
    ## Bi-Temporal
    'OSCDDataset',
    'LevirCdDataset',
    'xView2Dataset',
    'KC3DDataset',
    'AirChangeDataset',
    'CDDDataset',
    'SYSU_CD_Dataset',
    ## Single-Temporal
    'Bi2SingleTemporal',
    'I3PEDataset',
    'PPSLDataset',
)
