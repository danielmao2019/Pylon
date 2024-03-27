"""Code borrowed from Yifan Wu.
"""
import numpy
import torch


class NormalizeDepth:

    def __init__(self, type: str):
        self.type = type

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        assert type(depth) == torch.Tensor, f"{type(depth)=}"
        method_name = f'_{self.type}_'
        assert hasattr(self, method_name), f"{method_name=}"
        method = getattr(self, method_name)
        depth = depth.numpy()
        depth = method(depth)
        depth = torch.from_numpy(depth)
        return depth

    def _min_max_flip_(self, depth: numpy.ndarray) -> numpy.ndarray:
        depth_max = numpy.nanmax(depth)
        depth_min = numpy.nanmin(depth)
        depth = numpy.nan_to_num(depth, copy=False, nan=depth_max)
        depth = 1 - (depth - depth_min) / (depth_max - depth_min)
        return depth

    def _min_max_(self, depth: numpy.ndarray) -> numpy.ndarray:
        depth_max = numpy.nanmax(depth)
        depth_min = numpy.nanmin(depth)
        depth = numpy.nan_to_num(depth, copy=False, nan=depth_max)
        depth = (depth - depth_min) / (depth_max - depth_min)
        return depth

    def _uoais_(self, depth: numpy.ndarray) -> numpy.ndarray:
        depth_min, depth_max = [2500, 15000]
        depth[depth > depth_max] = depth_max
        depth[depth < depth_min] = depth_min
        depth = (depth - depth_min) / (depth_max - depth_min)
        return depth
