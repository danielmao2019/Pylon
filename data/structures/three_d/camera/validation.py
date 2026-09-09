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
    intrinsics: "CameraIntrinsics",
    extrinsics: "CameraExtrinsics",
    names: Optional[List[Optional[str]]],
    ids: Optional[List[Optional[int]]],
    device: Optional[Union[str, torch.device]],
    dtype: Optional[torch.dtype],
) -> None:
    """Validate the batched component pair, its parallel metadata, device, and dtype for Cameras.

    Single-entry validation for ``Cameras.__init__``; the component checks are
    shape-agnostic, so the batched pair takes the same ones a single camera does,
    plus the cross-component agreement on the leading batch axis.

    Args:
        intrinsics: Candidate batched CameraIntrinsics whose params are each ``[B]`` torch.Tensor.
        extrinsics: Candidate batched CameraExtrinsics whose cam2world matrix is a ``[B, 4, 4]`` torch.Tensor.
        names: None, or a per-camera list of optional names parallel to the batch axis.
        ids: None, or a per-camera list of optional ids parallel to the batch axis.
        device: Optional device target for the batch, a string or torch.device.
        dtype: Optional floating dtype target for the batch.

    Returns:
        None.
    """
    validate_camera_attributes(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        name=None,
        id=None,
        device=device,
        dtype=dtype,
    )

    assert extrinsics.extrinsics.ndim == 3, (
        "Expected the batched CameraExtrinsics cam2world matrix to carry exactly one "
        f"leading batch axis, i.e. shape [B, 4, 4]. {extrinsics.extrinsics.shape=}"
    )
    batch_size = extrinsics.extrinsics.shape[0]
    assert len(intrinsics.params) > 0, (
        "Expected the batched CameraIntrinsics to carry at least one param. "
        f"{list(intrinsics.params.keys())=}"
    )
    for key, value in intrinsics.params.items():
        assert value.shape == (batch_size,), (
            "Expected every batched CameraIntrinsics param to carry the same leading "
            f"batch axis as the CameraExtrinsics. {key=} {value.shape=} {batch_size=}"
        )

    # __init__ fills them in after this runs, so an unnamed batch arrives here as None.
    assert names is None or len(names) == batch_size, (
        "Expected the per-camera names to be None or parallel to the batch axis. "
        f"{names=} {batch_size=}"
    )

    assert ids is None or len(ids) == batch_size, (
        "Expected the per-camera ids to be None or parallel to the batch axis. "
        f"{ids=} {batch_size=}"
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
    dtype: Optional[torch.dtype],
) -> None:
    """Validate the parts and the name / id / device / dtype for a Camera.

    Single-entry validation for ``Camera.__init__`` and, through ``validate_cameras_attributes``, for ``Cameras.__init__``; asserts the parts are a CameraIntrinsics / CameraExtrinsics that agree on device and dtype, and validates the name / id / device / dtype, relying on each part's own validation for its internals.

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
    # The device / dtype accessors read one component and describe both, so a disagreement makes them lie.
    assert intrinsics.device == extrinsics.device, (
        "Expected Camera components to share device. "
        f"{intrinsics.device=} {extrinsics.device=}"
    )
    assert intrinsics.dtype == extrinsics.dtype, (
        "Expected Camera components to share dtype. "
        f"{intrinsics.dtype=} {extrinsics.dtype=}"
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
