from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.validation import validate_camera_intrinsics


def validate_data(data: Dict[str, Any]) -> None:
    # Input validations
    assert isinstance(data, dict), f"{type(data)=}"


def validate_device(device: torch.device) -> None:
    # Input validations
    assert isinstance(device, torch.device), f"{type(device)=}"


def validate_intrinsic_params(data: Dict[str, Any]) -> None:
    # Input validations
    assert "fl_x" in data, "nerfstudio.json missing fl_x"
    assert "fl_y" in data, "nerfstudio.json missing fl_y"
    assert "cx" in data, "nerfstudio.json missing cx"
    assert "cy" in data, "nerfstudio.json missing cy"
    assert isinstance(data["fl_x"], float), f"{type(data['fl_x'])=}"
    assert isinstance(data["fl_y"], float), f"{type(data['fl_y'])=}"
    assert isinstance(data["cx"], float), f"{type(data['cx'])=}"
    assert isinstance(data["cy"], float), f"{type(data['cy'])=}"
    assert "k1" in data, "nerfstudio.json missing k1"
    assert "k2" in data, "nerfstudio.json missing k2"
    assert "p1" in data, "nerfstudio.json missing p1"
    assert "p2" in data, "nerfstudio.json missing p2"
    assert isinstance(data["k1"], float), f"{type(data['k1'])=}"
    assert isinstance(data["k2"], float), f"{type(data['k2'])=}"
    assert isinstance(data["p1"], float), f"{type(data['p1'])=}"
    assert isinstance(data["p2"], float), f"{type(data['p2'])=}"
    assert float(data["k1"]) == 0.0, f"k1 must be 0, got {data['k1']}"
    assert float(data["k2"]) == 0.0, f"k2 must be 0, got {data['k2']}"
    assert float(data["p1"]) == 0.0, f"p1 must be 0, got {data['p1']}"
    assert float(data["p2"]) == 0.0, f"p2 must be 0, got {data['p2']}"


def validate_intrinsics_data(data: Dict[str, Any]) -> None:
    # Input validations
    assert float(data["fl_x"]) > 0.0, "fl_x must be positive"
    assert float(data["fl_y"]) > 0.0, "fl_y must be positive"
    assert float(data["cx"]) >= 0.0, "cx must be non-negative"
    assert float(data["cy"]) >= 0.0, "cy must be non-negative"


def validate_resolution_data(data: Dict[str, Any]) -> None:
    # Input validations
    assert "w" in data and "h" in data, "nerfstudio.json must include w and h"
    assert isinstance(data["w"], int), f"{type(data['w'])=}"
    assert isinstance(data["h"], int), f"{type(data['h'])=}"
    assert (
        data["w"] > 0 and data["h"] > 0
    ), f"w/h must be positive, got {data['w']}, {data['h']}"
    assert "cx" in data and "cy" in data, "nerfstudio.json must include cx and cy"
    assert isinstance(data["cx"], float), f"{type(data['cx'])=}"
    assert isinstance(data["cy"], float), f"{type(data['cy'])=}"
    assert (
        data["cx"] > 0.0 and data["cy"] > 0.0
    ), f"cx/cy must be positive, got {data['cx']}, {data['cy']}"
    assert data["w"] == int(round(2 * float(data["cx"]))), "w must equal 2*cx"
    assert data["h"] == int(round(2 * float(data["cy"]))), "h must equal 2*cy"


def validate_resolution(
    resolution: Tuple[int, int], intrinsic_params: Dict[str, Any]
) -> None:
    # Input validations
    assert isinstance(resolution, tuple), f"{type(resolution)=}"
    assert len(resolution) == 2, f"{resolution=}"
    assert isinstance(resolution[0], int), f"{type(resolution[0])=}"
    assert isinstance(resolution[1], int), f"{type(resolution[1])=}"
    assert (
        resolution[0] > 0 and resolution[1] > 0
    ), f"h/w must be positive, got {resolution[0]}, {resolution[1]}"
    assert isinstance(intrinsic_params, dict), f"{type(intrinsic_params)=}"
    assert (
        "cx" in intrinsic_params and "cy" in intrinsic_params
    ), "intrinsic_params must include cx and cy"
    assert isinstance(intrinsic_params["cx"], float), f"{type(intrinsic_params['cx'])=}"
    assert isinstance(intrinsic_params["cy"], float), f"{type(intrinsic_params['cy'])=}"
    assert intrinsic_params["cx"] > 0.0 and intrinsic_params["cy"] > 0.0, (
        "intrinsic_params cx/cy must be positive, "
        f"got {intrinsic_params['cx']}, {intrinsic_params['cy']}"
    )
    assert resolution[1] == int(
        round(2 * float(intrinsic_params["cx"]))
    ), "w must equal 2*cx"
    assert resolution[0] == int(
        round(2 * float(intrinsic_params["cy"]))
    ), "h must equal 2*cy"


def validate_camera_model_data(data: Dict[str, Any]) -> None:
    # Input validations
    assert "camera_model" in data, "nerfstudio.json missing camera_model"
    assert (
        data["camera_model"] == "OPENCV"
    ), f"Unsupported camera_model: {data['camera_model']}"


def validate_camera_model(camera_model: str) -> None:
    # Input validations
    assert isinstance(camera_model, str), f"{type(camera_model)=}"
    assert camera_model == "OPENCV", f"Unsupported camera_model: {camera_model}"


def validate_intrinsics(intrinsics: torch.Tensor) -> None:
    # Input validations
    assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"

    validate_camera_intrinsics(intrinsics)


def validate_applied_transform_data(data: Dict[str, Any]) -> None:
    # Input validations
    assert "applied_transform" in data, "nerfstudio.json missing applied_transform"
    assert np.asarray(data["applied_transform"], dtype=np.float32).shape == (3, 4)


def validate_applied_transform(applied_transform: np.ndarray) -> None:
    # Input validations
    assert isinstance(applied_transform, np.ndarray), f"{type(applied_transform)=}"
    assert applied_transform.shape == (3, 4), f"{applied_transform.shape=}"


def validate_ply_file_path_data(data: Dict[str, Any]) -> None:
    # Input validations
    assert "ply_file_path" in data, "nerfstudio.json missing ply_file_path"
    assert isinstance(data["ply_file_path"], str), f"{type(data['ply_file_path'])=}"


def validate_ply_file_path(ply_file_path: str) -> None:
    # Input validations
    assert isinstance(ply_file_path, str), f"{type(ply_file_path)=}"


def validate_frames_data(data: Dict[str, Any]) -> None:
    # Input validations
    assert "frames" in data, "nerfstudio.json missing frames"
    assert isinstance(data["frames"], list), "frames must be a list"
    assert data["frames"], "frames must be non-empty"
    assert all("file_path" in frame for frame in data["frames"])
    assert all(
        isinstance(frame["file_path"], str)
        and frame["file_path"].startswith("images/")
        and frame["file_path"].endswith(".png")
        for frame in data["frames"]
    )
    assert all(
        ("mask_path" not in frame)
        or (
            isinstance(frame["mask_path"], str)
            and frame["mask_path"].startswith("masks/")
            and frame["mask_path"].endswith(".png")
        )
        for frame in data["frames"]
    )
    assert all("transform_matrix" in frame for frame in data["frames"])
    assert all(
        ("colmap_image_id" not in frame) or isinstance(frame["colmap_image_id"], int)
        for frame in data["frames"]
    )


def validate_split_filenames_data(data: Dict[str, Any]) -> None:
    # Input validations
    assert (
        "train_filenames" in data
        and "val_filenames" in data
        and "test_filenames" in data
    ) or (
        "train_filenames" not in data
        and "val_filenames" not in data
        and "test_filenames" not in data
    ), "train/val/test filenames must all be provided together or all omitted"
    assert "train_filenames" not in data or isinstance(
        data["train_filenames"], list
    ), f"{type(data['train_filenames'])=}"
    assert "val_filenames" not in data or isinstance(
        data["val_filenames"], list
    ), f"{type(data['val_filenames'])=}"
    assert "test_filenames" not in data or isinstance(
        data["test_filenames"], list
    ), f"{type(data['test_filenames'])=}"
    assert (
        "train_filenames" not in data or data["train_filenames"]
    ), "train_filenames must be non-empty"
    assert (
        "val_filenames" not in data or data["val_filenames"]
    ), "val_filenames must be non-empty"
    assert (
        "test_filenames" not in data or data["test_filenames"]
    ), "test_filenames must be non-empty"
    assert "train_filenames" not in data or {
        frame["file_path"] for frame in data["frames"]
    } == set(data["train_filenames"]) | set(data["val_filenames"]) | set(
        data["test_filenames"]
    ), "train/val/test filenames must match frames file_path entries"


def validate_cameras(cameras: Cameras) -> None:
    # Input validations
    assert isinstance(cameras, Cameras), f"{type(cameras)=}"


def validate_filenames(filenames: List[str]) -> None:
    # Input validations
    assert isinstance(filenames, list), f"{type(filenames)=}"
    assert filenames, "filenames must be non-empty"
    assert all(isinstance(item, str) for item in filenames), f"{filenames=}"


def validate_split_filenames(
    train: Optional[List[str]],
    val: Optional[List[str]],
    test: Optional[List[str]],
    filenames: List[str],
) -> None:
    # Input validations
    assert (train is None and val is None and test is None) or (
        train is not None and val is not None and test is not None
    ), "train/val/test filenames must all be provided together or all omitted"
    assert train is None or isinstance(train, list), f"{type(train)=}"
    assert val is None or isinstance(val, list), f"{type(val)=}"
    assert test is None or isinstance(test, list), f"{type(test)=}"
    assert isinstance(filenames, list), f"{type(filenames)=}"
    assert filenames, "filenames must be non-empty"
    assert train is None or train, "train_filenames must be non-empty"
    assert val is None or val, "val_filenames must be non-empty"
    assert test is None or test, "test_filenames must be non-empty"
    assert train is None or all(isinstance(item, str) for item in train), f"{train=}"
    assert val is None or all(isinstance(item, str) for item in val), f"{val=}"
    assert test is None or all(isinstance(item, str) for item in test), f"{test=}"
    assert all(isinstance(item, str) for item in filenames), f"{filenames=}"
    assert train is None or set(filenames) == set(train) | set(val) | set(
        test
    ), "train/val/test filenames must match frames file_path entries"
