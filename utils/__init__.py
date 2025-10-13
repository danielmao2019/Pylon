"""
UTILS API
"""

from pathlib import Path
import importlib.util
import sys
import types

from utils import dynamic_executor
from utils import automation
from utils import builders
from utils import conversions
from utils import determinism
from utils import input_checks
from utils import io
from utils import logging
from utils import models
from utils import ops
from utils import point_cloud_ops

from utils import gradients
from utils import timeout

from utils import object_detection
from utils import semantic_segmentation


_2dgs_ops_module = types.ModuleType(__name__ + ".2dgs_ops")
_rendering_spec = importlib.util.spec_from_file_location(
    __name__ + ".2dgs_ops.rendering",
    Path(__file__).resolve().parent / "2dgs_ops" / "rendering.py",
)
if _rendering_spec.loader is None:  # pragma: no cover - defensive guard
    raise ImportError("Unable to load utils.2dgs_ops.rendering")
_rendering_module = importlib.util.module_from_spec(_rendering_spec)
_rendering_spec.loader.exec_module(_rendering_module)
setattr(_2dgs_ops_module, "rendering", _rendering_module)
_2dgs_ops_module.__all__ = getattr(_rendering_module, "__all__", [])
sys.modules[_2dgs_ops_module.__name__] = _2dgs_ops_module
sys.modules[_rendering_spec.name] = _rendering_module
setattr(sys.modules[__name__], "two_d_gaussian_ops", _2dgs_ops_module)
two_d_gaussian_ops = _2dgs_ops_module


__all__ = (
    'dynamic_executor',
    'automation',
    'builders',
    'conversions',
    'determinism',
    'input_checks',
    'io',
    'logging',
    'models',
    'ops',
    'point_cloud_ops',
    'gradients',
    'timeout',
    'object_detection',
    'semantic_segmentation',
    'two_d_gaussian_ops',
)
