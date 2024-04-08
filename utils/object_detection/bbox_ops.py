import torch


def bbox_cxcywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    assert type(bbox) == torch.Tensor, f"{type(bbox)=}"
    assert bbox.shape[-1] == 4, f"{bbox.shape=}"
    x_c, y_c, w, h = bbox.unbind(-1)
    result = [
        (x_c - 0.5 * w),  # x0
        (y_c - 0.5 * h),  # y0
        (x_c + 0.5 * w),  # x1
        (y_c + 0.5 * h),  # y1
    ]
    return torch.stack(result, dim=-1)


def bbox_xyxy_to_cxcywh(bbox: torch.Tensor) -> torch.Tensor:
    assert type(bbox) == torch.Tensor, f"{type(bbox)=}"
    assert bbox.shape[-1] == 4, f"{bbox.shape=}"
    x0, y0, x1, y1 = bbox.unbind(-1)
    result = [
        (x0 + x1) / 2,  # x coord of center
        (y0 + y1) / 2,  # y coord of center
        (x1 - x0),  # width
        (y1 - y0),  # height
    ]
    return torch.stack(result, dim=-1)
