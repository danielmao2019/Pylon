"""
MODELS.POINT_CLOUD_REGISTRATION.CLASSIC API
"""
from models.point_cloud_registration.classic.icp import ICP
from models.point_cloud_registration.classic.ransac_fpfh import RANSAC_FPFH
from models.point_cloud_registration.classic.teaserplusplus import TeaserPlusPlus


__all__ = (
    'ICP',
    'RANSAC_FPFH',
    'TeaserPlusPlus',
)
