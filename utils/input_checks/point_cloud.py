from typing import Any, Dict
import torch


def check_point_cloud(pc: Dict[str, torch.Tensor]) -> None:
    assert isinstance(pc, dict)
    assert all(isinstance(k, str) for k in pc.keys())
    assert all(isinstance(v, torch.Tensor) for v in pc.values())

    assert 'pos' in pc, f"{pc.keys()=}"
    check_pc_xyz(obj=pc['pos'])

    if 'rgb' in pc:
        check_pc_rgb(obj=pc['rgb'])

    length = pc['pos'].shape[0]
    assert all(pc[key].shape[0] == length for key in pc.keys() if key != 'pos'), \
        f"{{{', '.join(f'{k}: {pc[k].shape}' for k in pc.keys())}}}"

    device = pc['pos'].device
    assert all(pc[key].device == device for key in pc.keys() if key != 'pos'), \
        f"{{{', '.join(f'{k}: {pc[k].device}' for k in pc.keys())}}}"


def check_pc_xyz(obj: Any) -> None:
    assert isinstance(obj, torch.Tensor), f"Expected torch.Tensor, got {type(obj)}"
    assert obj.ndim == 2, f"Expected 2D tensor, got shape {obj.shape}"
    assert obj.shape[0] > 0, f"Expected positive number of points, got {obj.shape[0]}"
    assert obj.shape[1] == 3, f"Expected shape (N, 3), got {obj.shape}"
    assert obj.is_floating_point(), f"Expected floating point tensor, got {obj.dtype}"
    assert not torch.isnan(obj).any(), "Tensor contains NaN values"
    assert not torch.isinf(obj).any(), "Tensor contains Inf values"


def check_pc_rgb(obj: Any) -> None:
    """No guarantee if this is in int format or in float format.
    """
    assert isinstance(obj, torch.Tensor), f"Expected torch.Tensor, got {type(obj)}"
    assert obj.ndim == 2, f"Expected 2D tensor, got shape {obj.shape}"
    assert obj.shape[0] > 0, f"Expected positive number of points, got {obj.shape[0]}"
    assert obj.shape[1] == 3, f"Expected shape (N, 3), got {obj.shape}"
    assert not torch.isnan(obj).any(), "Tensor contains NaN values"
    assert not torch.isinf(obj).any(), "Tensor contains Inf values"
