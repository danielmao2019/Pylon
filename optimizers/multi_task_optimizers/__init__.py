"""
OPTIMIZERS.MULTI_TASK_OPTIMIZERS API
"""
from optimizers.multi_task_optimizers.mtl_optimizer import MTLOptimizer
# gradient manipulation methods
from optimizers.multi_task_optimizers.gradient_manipulation.rgw import RGWOptimizer
from optimizers.multi_task_optimizers.gradient_manipulation.pcgrad import PCGradOptimizer
from optimizers.multi_task_optimizers.gradient_manipulation.gradvac import GradVacOptimizer
from optimizers.multi_task_optimizers.gradient_manipulation.mgda import MGDAOptimizer
from optimizers.multi_task_optimizers.gradient_manipulation.cagrad import CAGradOptimizer
from optimizers.multi_task_optimizers.gradient_manipulation.graddrop import GradDropOptimizer
from optimizers.multi_task_optimizers.gradient_manipulation.nashmtl import NashMTLOptimizer
from optimizers.multi_task_optimizers.gradient_manipulation.alignedmtl import AlignedMTLOptimizer
# gradient balancing methods
from optimizers.multi_task_optimizers.gradient_balancing.rlw import RLWOptimizer
from optimizers.multi_task_optimizers.gradient_balancing.gls import GLSOptimizer
from optimizers.multi_task_optimizers.gradient_balancing.dwa import DWAOptimizer
from optimizers.multi_task_optimizers.gradient_balancing.uncertainty import UncertaintyOptimizer
from optimizers.multi_task_optimizers.gradient_balancing.gradnorm import GradNormOptimizer
from optimizers.multi_task_optimizers.gradient_balancing.famo import FAMOOptimizer
# gradient regularization methods
from optimizers.multi_task_optimizers.gradient_regularization.cosreg import CosRegOptimizer
# others
from optimizers.multi_task_optimizers.gradient_manipulation.imtl import IMTLOptimizer


__all__ = (
    'MTLOptimizer',
    # gradient manipulation methods
    'RGWOptimizer',
    'PCGradOptimizer',
    'GradVacOptimizer',
    'MGDAOptimizer',
    'CAGradOptimizer',
    'GradDropOptimizer',
    'NashMTLOptimizer',
    'AlignedMTLOptimizer',
    # gradient balancing methods
    'RLWOptimizer',
    'GLSOptimizer',
    'DWAOptimizer',
    'UncertaintyOptimizer',
    'GradNormOptimizer',
    'FAMOOptimizer',
    # gradient regularization methods
    'CosRegOptimizer',
    # others
    'IMTLOptimizer',
)
