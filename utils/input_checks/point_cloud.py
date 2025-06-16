from typing import Dict
import torch


def check_point_cloud(pc: Dict[str, torch.Tensor]) -> None:
    assert isinstance(pc, dict)
    assert all(isinstance(k, str) for k in pc.keys())
    assert all(isinstance(v, torch.Tensor) for v in pc.values())

    assert 'pos' in pc, f"{pc.keys()=}"
    assert pc['pos'].ndim == 2, f"{pc['pos'].shape=}"
    assert pc['pos'].shape[1] == 3, f"{pc['pos'].shape=}"

    length = pc['pos'].shape[0]
    assert all(pc[key].shape[0] == length for key in pc.keys() if key != 'pos'), \
        f"{{{', '.join(f'{k}: {pc[k].shape}' for k in pc.keys())}}}"

    device = pc['pos'].device
    assert all(pc[key].device == device for key in pc.keys() if key != 'pos'), \
        f"{{{', '.join(f'{k}: {pc[k].device}' for k in pc.keys())}}}"
