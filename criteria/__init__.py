"""
CRITERIA API
"""
from criteria.base_criterion import BaseCriterion
from criteria import common
from criteria import vision_2d
from criteria import vision_3d
from criteria import diffusion
from criteria import wrappers
from criteria.wrappers import *
from criteria.vision_2d.classification import *
from criteria.vision_2d.detection import *
from criteria.vision_2d.dense_prediction import *
from criteria.vision_2d.generative import *
from criteria.vision_2d.change_detection import *
from criteria.vision_3d.dense_prediction import *
from criteria.vision_3d.detection import *
from criteria.diffusion import *
from criteria.change_detection import *


__all__ = (
    'BaseCriterion',
    'common',
    'vision_2d',
    'vision_3d',
    'diffusion',
    'wrappers',
    # Wrappers
    'MultiTaskCriterion',
    'SingleTaskCriterion',
    'PyTorchCriterionWrapper',
    
    # 2D Vision - Classification
    'ClassificationCriterion',
    
    # 2D Vision - Detection
    'ObjectDetectionCriterion',
    
    # 2D Vision - Dense Prediction
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'SemanticSegmentationCriterion',
    'SpatialCrossEntropyCriterion',
    'IoULoss',
    'SSIMLoss',
    'InstanceSegmentationCriterion',
    
    # 2D Vision - Change Detection
    'SNUNetCriterion',
    'FTNLoss',
    'PPSLCriterion',
    'DsifnCriterion',
    'EdgeLoss',
    'FPNLoss',
    'STMambaBCDCriterion',
    'CDMaskFormerCriterion',
    
    # 2D Vision - Generative
    'GenerativeRCriterion',
    
    # 3D Vision - Dense Prediction
    '3DSemanticSegmentationCriterion',
    
    # 3D Vision - Detection
    '3DObjectDetectionCriterion',
    
    # Diffusion
    'CCDMCriterion',
)
