import json
import math
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from plyfile import PlyData


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
):
    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return float(delay_rate * log_lerp)

    return helper


class BasicModel:
    def setup_functions(self) -> None:
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def eval(self) -> None:
        return

    def train(self) -> None:
        return

    def set_appearance(self, num_cameras: int) -> None:
        self.embedding_appearance = None

    def oneupSHdegree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_coarse_interval(self, opt: SimpleNamespace) -> None:
        return

    def set_anchor_mask(self, *args: Any) -> None:
        self._anchor_mask = torch.ones(
            self._anchor.shape[0], dtype=torch.bool, device="cuda"
        )

    def map_to_int_level(
        self, pred_level: torch.Tensor, cur_level: int
    ) -> torch.Tensor:
        if self.dist2level == 'floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == 'round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == 'ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == 'progressive':
            pred_level = torch.clamp(
                pred_level + 1.0, min=0.9999, max=cur_level + 0.9999
            )
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = self._level.squeeze(dim=1) == int_level
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")

        return int_level


class OctreeGS_3DGS(BasicModel):
    def __init__(self, **model_kwargs: Any) -> None:
        self.active_sh_degree = 0
        for key, value in model_kwargs.items():
            setattr(self, key, value)

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._level = torch.empty(0)
        self._extra_level = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._anchor_mask = torch.empty(0, dtype=torch.bool)

        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.optimizer = None
        self.spatial_lr_scale = 0.0
        self.coarse_intervals: List[float] = []
        self.setup_functions()

    @property
    def get_anchor(self) -> torch.Tensor:
        return self._anchor

    @property
    def get_level(self) -> torch.Tensor:
        return self._level

    @property
    def get_extra_level(self) -> torch.Tensor:
        return self._extra_level

    @property
    def get_offset(self) -> torch.Tensor:
        return self._offset

    @property
    def get_features(self) -> torch.Tensor:
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    def set_coarse_interval(self, opt: SimpleNamespace) -> None:
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1 / opt.coarse_factor
            a1 = opt.coarse_iter * (1 - q) / (1 - q**num_level)
            temp_interval = 0.0
            for i in range(num_level):
                interval = a1 * q**i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_anchor_mask(
        self, cam_center: torch.Tensor, iteration: int, resolution_scale: float
    ) -> None:
        dist = (
            torch.sqrt(torch.sum((self._anchor - cam_center) ** 2, dim=1))
            * resolution_scale
        )
        pred_level = (
            torch.log2(self.standard_dist / dist) / math.log2(self.fork)
            + self._extra_level
        )

        if getattr(self, 'progressive', False):
            coarse_index = (
                np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
            )
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = self._level.squeeze(dim=1) <= int_level

    def load_ply(self, path: str) -> None:
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])

        anchor = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        ).astype(np.float32)
        levels = np.asarray(plydata.elements[0]["level"])[..., np.newaxis].astype(
            np.int16
        )
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[
            ..., np.newaxis
        ].astype(np.float32)
        offset_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_offset")
        ]
        offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(
                np.float32
            )
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((anchor.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((anchor.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(
            extra_levels, dtype=torch.float, device="cuda"
        ).squeeze(dim=1)
        self._offset = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._anchor_mask = torch.ones(
            self._anchor.shape[0], dtype=torch.bool, device="cuda"
        )
        self.active_sh_degree = self.max_sh_degree
        self.levels = round(self.levels)
        if self.init_level == -1:
            self.init_level = int(self.levels / 2)

    def generate_neural_gaussians(
        self,
        viewpoint_camera: Any,
        visible_mask: torch.Tensor = None,
        ape_code: int = -1,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        if visible_mask is None:
            visible_mask = torch.ones(
                self.get_anchor.shape[0],
                dtype=torch.bool,
                device=self.get_anchor.device,
            )

        anchor = self.get_anchor[visible_mask]
        grid_offsets = self.get_offset[visible_mask]
        scaling = self.get_scaling[visible_mask]
        opacity = self.get_opacity[visible_mask]
        rotation = self.get_rotation[visible_mask]
        color = self.get_features[visible_mask]

        if self.dist2level == "progressive":
            prog = self._prog_ratio[visible_mask]
            transition_mask = self.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            opacity = opacity * prog

        offsets = grid_offsets.view([-1, 3]) * scaling[:, :3]
        scaling = scaling[:, 3:]

        xyz = anchor + offsets
        mask = torch.ones(xyz.shape[0], dtype=torch.bool, device="cuda")

        return xyz, color, opacity, scaling, rotation, self.active_sh_degree, mask


