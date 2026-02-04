import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.validation import validate_camera_intrinsics
from data.structures.three_d.nerfstudio.validate import (
    MODALITY_SPECS,
    validate_applied_transform_data,
    validate_camera_model_data,
    validate_data,
    validate_frames_data,
    validate_intrinsic_params,
    validate_intrinsics_data,
    validate_ply_file_path_data,
    validate_resolution_data,
    validate_split_filenames_data,
)


def load_intrinsic_params(data: Dict[str, Any]) -> Dict[str, float | int]:
    keys = ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"]
    return {key: data[key] for key in keys}


def load_resolution(data: Dict[str, Any]) -> Tuple[int, int]:
    return (data["h"], data["w"])


def load_camera_model(data: Dict[str, Any]) -> str:
    return data["camera_model"]


def load_intrinsics(
    data: Dict[str, Any], device: str | torch.device = torch.device("cpu")
) -> torch.Tensor:
    intrinsics = torch.tensor(
        [
            [
                float(data["fl_x"]),
                0.0,
                float(data["cx"]),
            ],
            [
                0.0,
                float(data["fl_y"]),
                float(data["cy"]),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=torch.device(device),
    )
    validate_camera_intrinsics(intrinsics)
    return intrinsics


def load_applied_transform(data: Dict[str, Any]) -> np.ndarray:
    return np.asarray(data["applied_transform"], dtype=np.float32)


def load_ply_file_path(data: Dict[str, Any]) -> str:
    return data["ply_file_path"]


def load_cameras(
    data: Dict[str, Any], device: str | torch.device = torch.device("cpu")
) -> Cameras:
    frames: List[Any] = data["frames"]
    intrinsics = load_intrinsics(data=data, device=device)
    batched_intrinsics = intrinsics.repeat(len(frames), 1, 1)
    batched_extrinsics = torch.stack(
        [
            torch.tensor(
                frame["transform_matrix"],
                dtype=torch.float32,
                device=device,
            )
            for frame in frames
        ],
        dim=0,
    )
    conventions = ["opengl"] * len(frames)
    names: List[str | None] = [Path(frame["file_path"]).stem for frame in frames]
    ids: List[int | None] = [frame.get("colmap_image_id") for frame in frames]
    return Cameras(
        intrinsics=batched_intrinsics,
        extrinsics=batched_extrinsics,
        conventions=conventions,
        names=names,
        ids=ids,
        device=device,
    )


def load_modalities(data: Dict[str, Any]) -> List[str]:
    frames: List[Any] = data["frames"]
    return [
        modality for modality, spec in MODALITY_SPECS.items() if spec[0] in frames[0]
    ]


def load_filenames(data: Dict[str, Any]) -> List[str]:
    frames: List[Any] = data["frames"]
    return [Path(frame["file_path"]).stem for frame in frames]


def load_split_filenames(
    data: Dict[str, Any],
) -> Tuple[List[str] | None, List[str] | None, List[str] | None]:
    if "train_filenames" not in data:
        return None, None, None
    return (
        data["train_filenames"],
        data["val_filenames"],
        data["test_filenames"],
    )


def load_nerfstudio_data(
    filepath: str | Path,
    device: str | torch.device = torch.device("cuda"),
) -> Tuple[
    Dict[str, Any],
    Dict[str, float | int],
    Tuple[int, int],
    str,
    torch.Tensor,
    np.ndarray,
    str,
    Cameras,
    List[str],
    List[str],
    List[str] | None,
    List[str] | None,
    List[str] | None,
]:
    # Input validations
    assert isinstance(filepath, (str, Path)), f"{type(filepath)=}"
    assert isinstance(device, (str, torch.device)), f"{type(device)=}"

    # Input normalizations
    path = Path(filepath).resolve()
    target_device = torch.device(device)

    assert path.is_file(), f"transforms.json not found: {path}"
    with path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)

    validate_data(data)
    validate_intrinsic_params(data)
    validate_resolution_data(data)
    validate_camera_model_data(data)
    validate_intrinsics_data(data)
    validate_applied_transform_data(data)
    validate_ply_file_path_data(data=data, root_dir=path.parent)
    validate_frames_data(data=data, root_dir=path.parent)
    validate_split_filenames_data(data)

    intrinsic_params = load_intrinsic_params(data)
    resolution = load_resolution(data)
    camera_model = load_camera_model(data)
    intrinsics = load_intrinsics(data=data, device=target_device)
    applied_transform = load_applied_transform(data)
    ply_file_path = load_ply_file_path(data)
    train_filenames, val_filenames, test_filenames = load_split_filenames(data)
    cameras = load_cameras(data=data, device=target_device)
    modalities = load_modalities(data)
    filenames = load_filenames(data)

    return (
        data,
        intrinsic_params,
        resolution,
        camera_model,
        intrinsics,
        applied_transform,
        ply_file_path,
        cameras,
        modalities,
        filenames,
        train_filenames,
        val_filenames,
        test_filenames,
    )
