import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import torch

from models.three_d.octree_gs.model import OctreeGS_3DGS


@torch.no_grad()
def load_octree_gs_3dgs(
    model_dir: Union[str, Path], device: Union[str, torch.device] = 'cuda'
) -> OctreeGS_3DGS:
    model_path = Path(model_dir)
    assert model_path.is_dir(), f"OctreeGS model directory does not exist: {model_dir}"

    cfg_path = model_path / "cfg_args"
    assert cfg_path.is_file(), f"OctreeGS cfg_args missing: {cfg_path}"
    with cfg_path.open("r", encoding="utf-8") as cfg_file:
        cfg_text = cfg_file.read()

    cfg = eval(cfg_text)
    assert isinstance(
        cfg, Namespace
    ), f"cfg_args at '{cfg_path}' did not evaluate to Namespace (got {type(cfg)})"
    assert hasattr(cfg, 'base_model'), f"cfg_args at '{cfg_path}' missing base_model"
    assert isinstance(
        cfg.base_model, str
    ), f"cfg_args at '{cfg_path}' base_model must be str, got {type(cfg.base_model)}"
    assert (
        cfg.base_model == '3dgs'
    ), f"cfg_args at '{cfg_path}' base_model must be '3dgs' for OctreeGS (got {cfg.base_model!r})"
    assert hasattr(
        cfg, 'model_config'
    ), f"cfg_args at '{cfg_path}' missing model_config"
    model_config = cfg.model_config
    assert isinstance(
        model_config, dict
    ), f"cfg_args at '{cfg_path}' model_config must be dict, got {type(model_config)}"
    assert (
        'kwargs' in model_config
    ), f"cfg_args at '{cfg_path}' model_config missing 'kwargs'"
    assert model_config.get('name') == 'GaussianLoDModel', (
        "cfg_args model_config.name must be 'GaussianLoDModel' for octree_gs "
        f"(got {model_config.get('name')!r})"
    )
    model_kwargs = model_config['kwargs']

    cameras_json_path = model_path / "cameras.json"
    assert (
        cameras_json_path.is_file()
    ), f"OctreeGS cameras.json missing: {cameras_json_path}"
    with cameras_json_path.open('r', encoding='utf-8') as cameras_file:
        num_cameras = len(json.load(cameras_file))
    assert (
        num_cameras > 0
    ), f"cameras.json at '{cameras_json_path}' must list at least one camera"

    ply_relpath = Path("point_cloud") / "iteration_30000" / "point_cloud.ply"
    ply_path = model_path / ply_relpath
    assert ply_path.is_file(), (
        "OctreeGS model directory does not contain expected point cloud file: "
        f"{ply_path}"
    )
    checkpoint_path = ply_path.parent
    assert (
        checkpoint_path.is_dir()
    ), f"OctreeGS checkpoint directory missing: {checkpoint_path}"

    gaussians = OctreeGS_3DGS(**model_kwargs)
    gaussians.set_appearance(num_cameras=num_cameras)
    gaussians.load_ply(str(ply_path))
    gaussians.eval()

    opt = SimpleNamespace(
        coarse_iter=10000,
        coarse_factor=1.5,
    )
    gaussians.set_coarse_interval(opt)

    gaussian_device = gaussians.get_anchor.device
    target_device = torch.device(device)
    assert target_device.type == gaussian_device.type, (
        "Loaded OctreeGS model resides on device '{}' but caller requested '{}'. "
        "OctreeGS checkpoints are stored on GPU only.".format(
            gaussian_device, target_device
        )
    )

    assert gaussians.opacity_accum.shape[0] == 0, (
        "Expected empty opacity_accum for inference, got shape "
        f"{gaussians.opacity_accum.shape}"
    )
    assert gaussians.anchor_demon.shape[0] == 0, (
        "Expected empty anchor_demon for inference, got shape "
        f"{gaussians.anchor_demon.shape}"
    )
    assert gaussians.offset_gradient_accum.shape[0] == 0, (
        "Expected empty offset_gradient_accum for inference, got shape "
        f"{gaussians.offset_gradient_accum.shape}"
    )
    assert gaussians.offset_denom.shape[0] == 0, (
        "Expected empty offset_denom for inference, got shape "
        f"{gaussians.offset_denom.shape}"
    )

    return gaussians
