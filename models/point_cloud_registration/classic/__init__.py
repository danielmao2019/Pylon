"""
MODELS.POINT_CLOUD_REGISTRATION.CLASSIC API
"""
from models.point_cloud_registration.classic.icp import icp
from models.point_cloud_registration.classic.teaserplusplus import teaserplusplus
from models.point_cloud_registration.classic.ransac_fpfh import ransac_fpfh


__all__ = (
    'icp',
    'teaserplusplus',
    'ransac_fpfh',
)
