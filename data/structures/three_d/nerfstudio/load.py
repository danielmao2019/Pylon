import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.camera.intrinsics.validation import (
    validate_camera_intrinsics_params,
)
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


def load_nerfstudio_data(
    filepath: Union[str, Path],
    device: Union[str, torch.device] = torch.device("cuda"),
) -> Tuple[
    Dict[str, Any],
    Dict[str, Union[float, int]],
    Tuple[int, int],
    str,
    torch.Tensor,
    np.ndarray,
    str,
    Cameras,
    List[str],
    Optional[List[str]],
    Optional[List[str]],
    Optional[List[str]],
]:
    """Open a NerfStudio transforms.json, validate each section of the record it holds, and return the record beside every section read out of it.

    Args:
        filepath: Path of the transforms.json file.
        device: Device the K matrix and the cameras are placed on.

    Returns:
        The raw record dict, its intrinsic params (`fl_x`, `fl_y`, `cx`, `cy`, `k1`, `k2`, `p1`, `p2`), its (h, w) resolution, its camera model name, the float32 [3, 3] pinhole K matrix, the float32 [3, 4] applied transform, the ply file path relative to the record's directory, the Cameras batch posed camera-to-world in the OpenGL convention, the modality names, and the train / val / test filename lists (each None when the record carries no split).
    """

    def _validate_inputs() -> None:
        assert isinstance(filepath, (str, Path)), f"{type(filepath)=}"
        assert isinstance(device, (str, torch.device)), f"{type(device)=}"

    _validate_inputs()

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
        train_filenames,
        val_filenames,
        test_filenames,
    )


def load_intrinsic_params(data: Dict[str, Any]) -> Dict[str, Union[float, int]]:
    """Pick the focal, principal-point and k1, k2, p1, p2 distortion entries out of a NerfStudio transforms record, keeping the record's own key names.

    Args:
        data: The validated transforms record dict.

    Returns:
        The `fl_x`, `fl_y`, `cx`, `cy`, `k1`, `k2`, `p1`, `p2` entries of the record.
    """
    keys = ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"]
    return {key: data[key] for key in keys}


def load_resolution(data: Dict[str, Any]) -> Tuple[int, int]:
    """Read the image size a NerfStudio transforms record states, height first.

    Args:
        data: The validated transforms record dict.

    Returns:
        The record's (h, w).
    """
    return data["h"], data["w"]


def load_camera_model(data: Dict[str, Any]) -> str:
    """Read the camera model name a NerfStudio transforms record states.

    Args:
        data: The validated transforms record dict.

    Returns:
        The record's `camera_model` entry.
    """
    return data["camera_model"]


def load_intrinsics(
    data: Dict[str, Any], device: Union[str, torch.device] = torch.device("cpu")
) -> torch.Tensor:
    """Build the 3x3 pinhole K matrix of a NerfStudio transforms record's fl_x, fl_y, cx and cy, then pass those and its h and w as Python scalars to the standard-frame pinhole params validation.

    Args:
        data: The validated transforms record dict.
        device: Device the K matrix is placed on.

    Returns:
        The float32 [3, 3] pinhole K matrix `[[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]]` in the standard (pixel raster) convention.
    """
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
    validate_camera_intrinsics_params(
        model="pinhole",
        intr_convention="standard",
        params={
            "fx": float(data["fl_x"]),
            "fy": float(data["fl_y"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
            "h": int(data["h"]),
            "w": int(data["w"]),
        },
    )
    return intrinsics


def load_applied_transform(data: Dict[str, Any]) -> np.ndarray:
    """Read the applied_transform a NerfStudio transforms record carries, as a float32 array.

    Args:
        data: The validated transforms record dict.

    Returns:
        The record's applied transform as a float32 [3, 4] numpy array.
    """
    return np.asarray(data["applied_transform"], dtype=np.float32)


def load_ply_file_path(data: Dict[str, Any]) -> str:
    """Read the point cloud path a NerfStudio transforms record names.

    Args:
        data: The validated transforms record dict.

    Returns:
        The record's `ply_file_path` entry, relative to the record's directory.
    """
    return data["ply_file_path"]


def load_cameras(
    data: Dict[str, Any], device: Union[str, torch.device] = torch.device("cpu")
) -> Cameras:
    """Read the frames of one NerfStudio transforms record as the cameras that posed them.

    Args:
        data: The validated transforms record dict.
        device: Device the cameras are placed on.

    Returns:
        A Cameras batch of one camera per frame in record order, named by the stem of the frame's `file_path` and identified by its `colmap_im_id` (None when absent), with float32 standard-convention pinhole intrinsics and each frame's `transform_matrix` as camera-to-world extrinsics in the OpenGL convention.
    """
    frames: List[Any] = data["frames"]
    intrinsics_params = {
        "fx": float(data["fl_x"]),
        "fy": float(data["fl_y"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
        "h": int(data["h"]),
        "w": int(data["w"]),
    }
    # The record's one top-level pinhole governs every frame, so its params broadcast to the batch.
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={
            key: torch.full((len(frames),), value, dtype=torch.float32, device=device)
            for key, value in intrinsics_params.items()
        },
        intr_convention="standard",
        device=device,
    )
    extrinsics = CameraExtrinsics(
        extrinsics=torch.tensor(
            [frame["transform_matrix"] for frame in frames],
            dtype=torch.float32,
            device=device,
        ),
        extr_convention="opengl",
        device=device,
    )
    names: List[Optional[str]] = [Path(frame["file_path"]).stem for frame in frames]
    ids: List[Optional[int]] = []
    for frame in frames:
        if "colmap_im_id" in frame:
            ids.append(frame["colmap_im_id"])
        else:
            ids.append(None)
    return Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        names=names,
        ids=ids,
        device=device,
    )


def load_modalities(data: Dict[str, Any]) -> List[str]:
    """Name the modalities a NerfStudio record's frames carry, judged by which modality path keys (each MODALITY_SPECS spec's first entry) its first frame holds.

    Args:
        data: The validated transforms record dict, whose frames all carry the same modality keys.

    Returns:
        The names of the modalities whose path key the first frame holds, in MODALITY_SPECS order.
    """
    frames: List[Any] = data["frames"]
    return [
        modality for modality, spec in MODALITY_SPECS.items() if spec[0] in frames[0]
    ]


def load_split_filenames(
    data: Dict[str, Any],
) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
    """Read the train, val and test filename lists of a NerfStudio transforms record, a None for each when it carries no train_filenames.

    Args:
        data: The validated transforms record dict, carrying all three split lists or none of them.

    Returns:
        The record's train, val and test filename lists, or three Nones when it carries no split.
    """
    if "train_filenames" not in data:
        return None, None, None
    return data["train_filenames"], data["val_filenames"], data["test_filenames"]
