from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.validation import validate_camera_intrinsics


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


def load_filenames(data: Dict[str, Any]) -> List[str]:
    frames: List[Any] = data["frames"]
    return [frame["file_path"] for frame in frames]


def load_split_filenames(
    data: Dict[str, Any],
) -> Tuple[List[str] | None, List[str] | None, List[str] | None]:
    if "train_filenames" not in data:
        return None, None, None
    return data["train_filenames"], data["val_filenames"], data["test_filenames"]
