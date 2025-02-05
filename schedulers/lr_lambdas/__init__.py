"""
SCHEDULERS.LR_LAMBDAS API
"""
from schedulers.lr_lambdas.constant import ConstantLambda
from schedulers.lr_lambdas.warmup import WarmupLambda


__all__ = (
    'ConstantLambda',
    'WarmupLambda',
)
