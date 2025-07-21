"""
DATA.DATASETS API
"""
from data.datasets.base_dataset import BaseDataset
from data.datasets.base_synthetic_dataset import BaseSyntheticDataset
from data.datasets import random_datasets
from data.datasets.projection_dataset_wrapper import ProjectionDatasetWrapper

# torchvision datasets
from data.datasets.torchvision_datasets.mnist import MNISTDataset

# Semantic Segmentation datasets
from data.datasets.semantic_segmentation_datasets.coco_stuff_164k_dataset import COCOStuff164KDataset
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset

# GAN datasets
from data.datasets.gan_datasets.gan_dataset import GANDataset

# Multi-Task Learning datasets
from data.datasets.multi_task_datasets.multi_mnist_dataset import MultiMNISTDataset
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset
from data.datasets.multi_task_datasets.multi_task_facial_landmark_dataset import MultiTaskFacialLandmarkDataset
from data.datasets.multi_task_datasets.city_scapes_dataset import CityScapesDataset
from data.datasets.multi_task_datasets.nyu_v2_dataset import NYUv2Dataset
from data.datasets.multi_task_datasets.ade_20k_dataset import ADE20KDataset

# Change Detection datasets
## Bi-Temporal
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset
## Single-Temporal
from data.datasets.change_detection_datasets.single_temporal.bi2single_temporal_dataset import Bi2SingleTemporal
from data.datasets.change_detection_datasets.single_temporal.i3pe_dataset import I3PEDataset
from data.datasets.change_detection_datasets.single_temporal.ppsl_dataset import PPSLDataset

# Point Cloud Registration
from data.datasets.pcr_datasets.single_temporal_pcr_dataset import SingleTemporalPCRDataset
from data.datasets.pcr_datasets.bi_temporal_pcr_dataset import BiTemporalPCRDataset
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
from data.datasets.pcr_datasets.threedmatch_dataset import ThreeDMatchDataset, ThreeDLoMatchDataset
from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset
from data.datasets.pcr_datasets.lidar_camera_pose_pcr_dataset import LiDARCameraPosePCRDataset


__all__ = (
    'BaseDataset',
    'BaseSyntheticDataset',
    'random_datasets',
    'ProjectionDatasetWrapper',
    # torchvision datasets
    'MNISTDataset',
    # Semantic Segmentation datasets
    'COCOStuff164KDataset',
    'WHU_BD_Dataset',
    # GAN datasets
    'GANDataset',
    # Multi-Task Learning datasets
    'MultiMNISTDataset',
    'CelebADataset',
    'MultiTaskFacialLandmarkDataset',
    'CityScapesDataset',
    'NYUv2Dataset',
    'ADE20KDataset',
    # Change Detection datasets
    ## Bi-Temporal
    'OSCDDataset',
    'LevirCdDataset',
    'xView2Dataset',
    'KC3DDataset',
    'AirChangeDataset',
    'CDDDataset',
    'SYSU_CD_Dataset',
    'Urb3DCDDataset',
    'SLPCCDDataset',
    ## Single-Temporal
    'Bi2SingleTemporal',
    'I3PEDataset',
    'PPSLDataset',
    # Point Cloud Registration
    'SingleTemporalPCRDataset',
    'BiTemporalPCRDataset',
    'KITTIDataset',
    'ThreeDMatchDataset',
    'ThreeDLoMatchDataset',
    'ModelNet40Dataset',
    'LiDARCameraPosePCRDataset',
)
