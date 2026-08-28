from typing import TYPE_CHECKING, List, Optional, Union

import torch

if TYPE_CHECKING:
    from data.structures.three_d.camera.extrinsics.camera_extrinsics import (
        CameraExtrinsics,
    )
    from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
        CameraIntrinsics,
    )


def validate_cameras_attributes(
    intrinsics: List["CameraIntrinsics"],
    extrinsics: List["CameraExtrinsics"],
    names: List[Optional[str]],
    ids: List[Optional[int]],
    device: Optional[Union[str, torch.device]],
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Validate the parallel per-camera lists, names / ids, device, and dtype for Cameras.

    Single-entry validation for ``Cameras.__init__``; also validates the
    inter-relationship that all four per-camera lists are equal length.

    Args:
        intrinsics: Per-camera list of CameraIntrinsics.
        extrinsics: Per-camera list of CameraExtrinsics.
        names: Per-camera list of optional names.
        ids: Per-camera list of optional ids.
        device: Optional device target for the cameras, a string or torch.device.
        dtype: Optional floating dtype target for the cameras.

    Returns:
        None.
    """
    assert len(intrinsics) == len(extrinsics) == len(names) == len(ids), (
        "Expected the per-camera intrinsics / extrinsics / names / ids lists to be "
        f"equal length. {len(intrinsics)=} {len(extrinsics)=} {len(names)=} {len(ids)=}"
    )
    for intrinsic, extrinsic, name, id in zip(intrinsics, extrinsics, names, ids):
        validate_camera_attributes(
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            name=name,
            id=id,
            device=device,
            dtype=dtype,
        )
    assert device is None or isinstance(device, (str, torch.device)), (
        "Expected Cameras device to be None, a string, or torch.device. "
        f"{type(device)=}"
    )
    assert dtype is None or isinstance(dtype, torch.dtype), (
        "Expected Cameras dtype to be None or a torch dtype. " f"{type(dtype)=}"
    )
    if dtype is not None:
        assert torch.empty((), dtype=dtype).is_floating_point(), (
            "Expected Cameras dtype to be floating. " f"{dtype=}"
        )


def validate_camera_attributes(
    intrinsics: "CameraIntrinsics",
    extrinsics: "CameraExtrinsics",
    name: Optional[str],
    id: Optional[int],
    device: Optional[Union[str, torch.device]],
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Validate the parts and the name / id / device / dtype for a Camera.

    Single-entry validation for ``Camera.__init__``; asserts the parts are a
    CameraIntrinsics / CameraExtrinsics and validates the name / id / device,
    relying on each part's own validation for its internals.

    Args:
        intrinsics: Candidate CameraIntrinsics.
        extrinsics: Candidate CameraExtrinsics.
        name: Candidate camera name, None or a string.
        id: Candidate camera id, None or an integer.
        device: Optional device target for the camera, a string or torch.device.
        dtype: Optional floating dtype target for the camera.

    Returns:
        None.
    """
    from data.structures.three_d.camera.extrinsics.camera_extrinsics import (
        CameraExtrinsics,
    )
    from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
        CameraIntrinsics,
    )

    assert isinstance(intrinsics, CameraIntrinsics), (
        "Expected Camera intrinsics to be a CameraIntrinsics. " f"{type(intrinsics)=}"
    )
    assert isinstance(extrinsics, CameraExtrinsics), (
        "Expected Camera extrinsics to be a CameraExtrinsics. " f"{type(extrinsics)=}"
    )
    assert name is None or isinstance(name, str), (
        "Expected Camera name to be None or a string. " f"{type(name)=}"
    )
    assert id is None or isinstance(id, int), (
        "Expected Camera id to be None or an integer. " f"{type(id)=}"
    )
    assert device is None or isinstance(device, (str, torch.device)), (
        "Expected Camera device to be None, a string, or torch.device. "
        f"{type(device)=}"
    )
    assert dtype is None or isinstance(dtype, torch.dtype), (
        "Expected Camera dtype to be None or a torch dtype. " f"{type(dtype)=}"
    )
    if dtype is not None:
        assert torch.empty((), dtype=dtype).is_floating_point(), (
            "Expected Camera dtype to be floating. " f"{dtype=}"
        )
