from typing import List, Sequence, Dict, Union, Any, Optional
import os
import json
import jsbeautifier
import numpy
import torch
import torchvision
from PIL import Image

from utils.input_checks import check_read_file, check_write_file
from utils.ops import apply_tensor_op, transpose_buffer, buffer_mean


def load_image(
    filepath: str,
    dtype: Optional[torch.dtype] = None,
    sub: Union[float, Sequence[float]] = None,
    div: Union[float, Sequence[float]] = None,
) -> torch.Tensor:
    # input checks
    check_read_file(path=filepath, ext=['.png', '.jpg', '.jpeg'])
    # load image
    image = Image.open(filepath)
    # convert to torch.Tensor
    image = _pil2torch(image)
    # normalize image
    if sub is not None or div is not None:
        image = _normalize(image, sub=sub, div=div)
    # convert data type
    if dtype is not None:
        assert type(dtype) == torch.dtype, f"{type(dtype)=}"
        image = image.type(dtype)
    return image


def _pil2torch(image: Image) -> torch.Tensor:
    mode = image.mode
    # convert to torch.Tensor
    if mode == 'RGB':
        image = torch.from_numpy(numpy.array(image))
        image = image.permute(2, 0, 1)
        assert image.ndim == 3 and image.shape[0] == 3, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}"
    elif mode == 'RGBA':
        image = torch.from_numpy(numpy.array(image))
        image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        assert image.ndim == 3 and image.shape[0] == 4, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}"
    elif mode in ['L', 'P']:
        image = torch.from_numpy(numpy.array(image))
        assert image.ndim == 2, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}"
    elif mode in ['I', 'I;16']:
        image = torch.from_numpy(numpy.array(image, dtype=numpy.int32))
        assert image.ndim == 2, f"{image.shape=}"
    else:
        raise NotImplementedError(f"Conversion from PIL image to PyTorch tensor not implemented for {mode=}.")
    return image


def _normalize(image: torch.Tensor, sub, div) -> torch.Tensor:
    image = image.type(torch.float32)
    if sub is not None:
        sub = torch.tensor(sub, dtype=torch.float32)
        if image.ndim == 3:
            assert sub.numel() == 3, f"{sub.shape=}"
            sub = sub.view(3, 1, 1)
        else:
            assert image.ndim == 2, f"{image.shape=}"
            assert sub.numel() == 1, f"{sub.shape=}"
        image = image - sub
    if div is not None:
        div = torch.tensor(div, dtype=torch.float32)
        if image.ndim == 3:
            if div.numel() == 1:
                div = torch.tensor([div]*3)
            assert div.numel() == 3, f"{div.shape=}"
            div = div.view(3, 1, 1)
        else:
            assert image.ndim == 2, f"{image.shape=}"
            assert div.numel() == 1, f"{div.shape=}"
        image = image / div
    return image


def save_image(tensor: torch.Tensor, filepath: str) -> None:
    check_write_file(path=filepath)
    if tensor.ndim == 3 and tensor.shape[0] == 3 and tensor.dtype == torch.float32:
        torchvision.utils.save_image(tensor=tensor, fp=filepath)
    elif tensor.ndim == 2 and tensor.dtype == torch.uint8:
        Image.fromarray(tensor.numpy()).save(filepath)
    else:
        raise NotImplementedError(f"Unrecognized tensor format: shape={tensor.shape}, dtype={tensor.dtype}.")


def serialize_tensor(obj: Any):
    return apply_tensor_op(func=lambda x: x.detach().tolist(), inputs=obj)


def save_json(obj: Any, filepath: str) -> None:
    assert (
        os.path.dirname(filepath) == "" or
        os.path.isdir(os.path.dirname(filepath))
    ), f"{filepath=}, {os.path.dirname(filepath)=}"
    obj = serialize_tensor(obj)
    with open(filepath, mode='w') as f:
        f.write(jsbeautifier.beautify(json.dumps(obj), jsbeautifier.default_options()))


def read_average_scores(filepaths: List[List[str]]) -> Dict[str, List[Any]]:
    r"""Average will be taken along the first dimension.
    """
    # input checks
    assert type(filepaths) == list, f"{type(filepaths)=}"
    assert all(type(experiment_fps) == list for experiment_fps in filepaths)
    assert all(len(experiment_fps) == len(filepaths[0]) for experiment_fps in filepaths)
    assert all(all(type(json_fp) == str for json_fp in experiment_fps) for experiment_fps in filepaths)
    assert all(all(json_fp.endswith('.json') for json_fp in experiment_fps) for experiment_fps in filepaths)
    assert all(all(os.path.isfile(json_fp) for json_fp in experiment_fps) for experiment_fps in filepaths)
    # initialize
    m = len(filepaths)
    n = len(filepaths[0])
    # read
    avg_scores: List[List[Dict[str, Any]]] = []
    for experiment_fps in filepaths:
        scores: List[Dict[str, Any]] = []
        for json_fp in experiment_fps:
            with open(json_fp, mode='r') as f:
                scores.append(json.load(f))
        avg_scores.append(scores)
    assert len(avg_scores) == m
    assert len(avg_scores[0]) == n
    avg_scores: List[Dict[str, Any]] = [
        buffer_mean([avg_scores[i][j] for i in range(m)]) for j in range(n)
    ]
    avg_scores: Dict[str, List[Any]] = transpose_buffer(avg_scores)
    return avg_scores
