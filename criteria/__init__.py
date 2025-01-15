"""
CRITERIA API
"""
from criteria.base_criterion import BaseCriterion
from criteria import common
from criteria import vision_2d
from criteria import diffusion
from criteria import wrappers


__all__ = (
    'BaseCriterion',
    'common',
    'vision_2d',
    'diffusion',
    'wrappers',
)
