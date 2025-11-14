import json
import math
import os
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from plyfile import PlyData


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class BasicModel:

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def eval(self):
        return

    def train(self):
        return

    def set_appearance(self, num_cameras):
        self.embedding_appearance = None

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_coarse_interval(self, opt):
        return

    def set_anchor_mask(self, *args):
        self._anchor_mask = torch.ones(
            self._anchor.shape[0], dtype=torch.bool, device="cuda"
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if (
                'mlp' in group['name']
                or 'conv' in group['name']
                or 'embedding' in group['name']
            ):
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # statis grad information to guide liftting.
    def training_statis(self, render_pkg, width, height):
        viewspace_point_tensor = render_pkg["viewspace_points"]
        update_filter = render_pkg["visibility_filter"]
        anchor_visible_mask = render_pkg["visible_mask"]
        offset_selection_mask = render_pkg["selection_mask"]
        opacity = render_pkg["opacity"]
        # update opacity stats

        temp_opacity = torch.zeros(
            offset_selection_mask.shape[0], dtype=torch.float32, device="cuda"
        )
        temp_opacity[offset_selection_mask] = opacity.clone().view(-1).detach()

        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)

        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = (
            anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        )
        combined_mask = torch.zeros_like(
            self.offset_gradient_accum, dtype=torch.bool
        ).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad = viewspace_point_tensor.grad.squeeze(0)  # [N, 2]
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        grad_norm = torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if (
                'mlp' in group['name']
                or 'conv' in group['name']
                or 'embedding' in group['name']
            ):
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def get_remove_duplicates(
        self, grid_coords, selected_grid_coords_unique, num_overlap=1, use_chunk=True
    ):
        counts = torch.zeros(
            selected_grid_coords_unique.shape[0],
            dtype=torch.int,
            device=selected_grid_coords_unique.device,
        )

        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (
                1 if grid_coords.shape[0] % chunk_size != 0 else 0
            )
            for i in range(max_iters):
                chunk = grid_coords[i * chunk_size : (i + 1) * chunk_size]
                matches = (
                    selected_grid_coords_unique.unsqueeze(1) == chunk.unsqueeze(0)
                ).all(-1)
                counts += matches.sum(dim=1)
        else:
            matches = (
                selected_grid_coords_unique.unsqueeze(1) == grid_coords.unsqueeze(0)
            ).all(-1)
            counts = matches.sum(dim=1)

        remove_duplicates = counts >= num_overlap

        return remove_duplicates

    def map_to_int_level(self, pred_level, cur_level):
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

    def save_mlp_checkpoints(self, path):  # split or unite
        return

    def load_mlp_checkpoints(self, path):
        return

    def clean(self):
        del self.opacity_accum
        del self.anchor_demon
        del self.offset_gradient_accum
        del self.offset_denom
        torch.cuda.empty_cache()


class GaussianLoDModel(BasicModel):

    def __init__(self, **model_kwargs):

        for key, value in model_kwargs.items():
            setattr(self, key, value)

        self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        self._extra_level = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)

        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(self.view_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1),
            ).cuda()

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh(),
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7 * self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(
                self.feat_dim + self.view_dim + self.appearance_dim, self.feat_dim
            ),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3 * self.n_offsets),
            nn.Sigmoid(),
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_level(self):
        return self._level

    @property
    def get_extra_level(self):
        return self._extra_level

    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    @property
    def get_offset(self):
        return self._offset

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(
                num_cameras, self.appearance_dim
            ).cuda()
        else:
            self.embedding_appearance = None

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    def set_coarse_interval(self, opt):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1 / opt.coarse_factor
            a1 = opt.coarse_iter * (1 - q) / (1 - q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q**i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = (
                    torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale])
                    .float()
                    .cuda()
                )
                self.cam_infos = torch.cat(
                    (self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0
                )
                dist = torch.sqrt(torch.sum((points - cam_center) ** 2, dim=1))
                dist_max = torch.quantile(dist, self.dist_ratio)
                dist_min = torch.quantile(dist, 1 - self.dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, self.dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - self.dist_ratio)
        self.standard_dist = dist_max
        if self.levels == -1:
            self.levels = (
                torch.round(torch.log2(dist_max / dist_min) / math.log2(self.fork))
                .int()
                .item()
                + 1
            )
        if self.init_level == -1:
            self.init_level = int(self.levels / 2)

    def octree_sample(self, data):
        torch.cuda.synchronize()
        t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda()
        for cur_level in range(self.levels):
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            new_positions = (
                torch.unique(torch.round((data - self.init_pos) / cur_size), dim=0)
                * cur_size
                + self.init_pos
            )
            new_positions += self.padding * cur_size
            new_level = (
                torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda")
                * cur_level
            )
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        torch.cuda.synchronize()
        t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        dist = (
            torch.sqrt(torch.sum((self.get_anchor - cam_center) ** 2, dim=1))
            * resolution_scale
        )
        pred_level = (
            torch.log2(self.standard_dist / dist) / math.log2(self.fork)
            + self._extra_level
        )

        if self.progressive:
            coarse_index = (
                np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
            )
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = self._level.squeeze(dim=1) <= int_level

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        dist = (
            torch.sqrt(torch.sum((self.get_anchor - cam_center) ** 2, dim=1))
            * resolution_scale
        )
        pred_level = (
            torch.log2(self.standard_dist / dist) / math.log2(self.fork)
            + self._extra_level
        )
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = self._level.squeeze(dim=1) <= int_level

    def load_ply(self, path):
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
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(
                np.float32
            )

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_anchor_feat")
        ]
        anchor_feat_names = sorted(
            anchor_feat_names, key=lambda x: int(x.split('_')[-1])
        )
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(
                np.float32
            )

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

        self._anchor_feat = nn.Parameter(
            torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
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
        self._anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False)
        )
        self._anchor_mask = torch.ones(
            self._anchor.shape[0], dtype=torch.bool, device="cuda"
        )
        self.levels = round(self.levels)
        if self.init_level == -1:
            self.init_level = int(self.levels / 2)

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        if self.use_feat_bank:
            self.mlp_feature_bank = torch.jit.load(
                os.path.join(path, 'feature_bank_mlp.pt')
            ).cuda()
        if self.appearance_dim > 0:
            self.embedding_appearance = torch.jit.load(
                os.path.join(path, 'embedding_appearance.pt')
            ).cuda()

    def generate_neural_gaussians(
        self, viewpoint_camera, visible_mask=None, ape_code=-1
    ):
        # view frustum filtering for acceleration
        if visible_mask is None:
            visible_mask = torch.ones(
                self.get_anchor.shape[0],
                dtype=torch.bool,
                device=self.get_anchor.device,
            )

        anchor = self.get_anchor[visible_mask]
        feat = self.get_anchor_feat[visible_mask]
        grid_offsets = self.get_offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]

        # get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if self.use_feat_bank:
            bank_weight = self.get_featurebank_mlp(ob_view).unsqueeze(
                dim=1
            )  # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = (
                feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
                + feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
                + feat[:, ::1, :1] * bank_weight[:, :, 2:]
            )
            feat = feat.squeeze(dim=-1)  # [n, c]

        cat_local_view = torch.cat([feat, ob_view], dim=1)  # [N, c+3]

        if self.appearance_dim > 0:
            if ape_code < 0:
                camera_indicies = (
                    torch.ones_like(
                        cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
                    )
                    * viewpoint_camera.uid
                )
                appearance = self.get_appearance(camera_indicies)
            else:
                camera_indicies = (
                    torch.ones_like(
                        cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
                    )
                    * ape_code[0]
                )
                appearance = self.get_appearance(camera_indicies)

        # get offset's opacity
        neural_opacity = self.get_opacity_mlp(cat_local_view)  # [N, k]

        if self.dist2level == "progressive":
            prog = self._prog_ratio[visible_mask]
            transition_mask = self.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = neural_opacity > 0.0
        mask = mask.view(-1)

        # select opacity
        opacity = neural_opacity[mask]

        # get offset's color
        if self.appearance_dim > 0:
            color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = self.get_color_mlp(cat_local_view)
        color = color.reshape([anchor.shape[0] * self.n_offsets, 3])  # [mask]

        # get offset's cov
        scale_rot = self.get_cov_mlp(cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0] * self.n_offsets, 7])  # [mask]

        # offsets
        offsets = grid_offsets.view([-1, 3])  # [mask]

        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(
            concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets
        )
        concatenated_all = torch.cat(
            [concatenated_repeated, color, scale_rot, offsets], dim=-1
        )
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split(
            [6, 3, 3, 7, 3], dim=-1
        )

        # post-process cov
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(
            scale_rot[:, :3]
        )  # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:, 3:7])

        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:, :3]
        xyz = repeat_anchor + offsets

        return xyz, color, opacity, scaling, rot, None, mask


@torch.no_grad()
def load_octree_gs(
    model_dir: Union[str, Path], device: Union[str, torch.device] = 'cuda'
) -> GaussianLoDModel:
    model_path = Path(model_dir)
    assert model_path.is_dir(), f"3DGS model directory does not exist: {model_dir}"

    cfg_path = model_path / "cfg_args"
    with open(cfg_path, "r", encoding="utf-8") as cfg_file:
        cfg_text = cfg_file.read()

    cfg = eval(cfg_text)
    assert type(cfg) == Namespace
    assert hasattr(cfg, 'model_config')
    assert type(cfg.model_config) == dict
    assert 'kwargs' in cfg.model_config
    model_kwargs = cfg.model_config['kwargs']

    cameras_json_path = model_path / "cameras.json"
    with open(cameras_json_path, mode='r') as f:
        num_cameras = len(json.load(f))

    ply_relpath = Path("point_cloud") / "iteration_30000" / "point_cloud.ply"
    ply_path = model_path / ply_relpath
    assert ply_path.is_file(), (
        "3DGS model directory does not contain expected point cloud file: "
        f"{ply_path}"
    )

    checkpoint_relpath = Path("point_cloud") / "iteration_30000"
    checkpoint_path = model_path / checkpoint_relpath
    assert checkpoint_path.is_dir()

    opt = SimpleNamespace(
        coarse_iter=10000,
        coarse_factor=1.5,
    )

    gaussians = GaussianLoDModel(**model_kwargs)
    gaussians.set_appearance(num_cameras=num_cameras)
    gaussians.load_ply(str(ply_path))
    gaussians.load_mlp_checkpoints(str(checkpoint_path))
    gaussians.eval()
    gaussians.set_coarse_interval(opt)

    gaussian_device = gaussians.get_anchor.device
    target_device = torch.device(device)
    assert target_device.type == gaussian_device.type, (
        "Loaded 3DGS model resides on device '{}' but caller requested '{}'. "
        "Original 3DGS checkpoints are stored on GPU only.".format(
            gaussian_device, target_device
        )
    )

    # Assert that accumulation buffers are empty for inference models
    assert (
        gaussians.opacity_accum.shape[0] == 0
    ), f"Expected empty opacity_accum for inference, got shape {gaussians.opacity_accum.shape}"
    assert (
        gaussians.anchor_demon.shape[0] == 0
    ), f"Expected empty anchor_demon for inference, got shape {gaussians.anchor_demon.shape}"
    assert (
        gaussians.offset_gradient_accum.shape[0] == 0
    ), f"Expected empty offset_gradient_accum for inference, got shape {gaussians.offset_gradient_accum.shape}"
    assert (
        gaussians.offset_denom.shape[0] == 0
    ), f"Expected empty offset_denom for inference, got shape {gaussians.offset_denom.shape}"

    return gaussians
