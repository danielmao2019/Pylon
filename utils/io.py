from typing import Any, Optional
import os
import json
import jsbeautifier
import numpy
import torch
import torchvision
from PIL import Image
from .input_checks import check_read_file, check_write_file
from .ops import apply_tensor_op


def load_image(filepath: str, dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
    check_read_file(path=filepath)
    assert type(dtype) == torch.dtype, f"{type(dtype)=}"
    assert dtype in [torch.float32, torch.int64, torch.uint8, torch.bool], f"{dtype=}"
    # read from disk
    image = Image.open(filepath)
    # convert to torch.Tensor
    if image.mode == 'RGB':
        image = torch.from_numpy(numpy.array(image)).permute(2, 0, 1)
        assert image.dim() == 3 and image.shape[0] == 3, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}, {filepath=}"
    elif image.mode in ['L', 'I']:
        image = torch.from_numpy(numpy.array(image))
        assert image.dim() == 2, f"{image.shape=}"
        assert image.dtype == torch.uint8 if image.mode == 'L' else torch.int32, f"{image.dtype=}, {filepath=}"
    else:
        raise NotImplementedError(f"{image.mode=}")
    image = image.type(dtype)
    # transform
    if dtype == torch.float32:
        image = image / 255.
    return image


def save_image(tensor: torch.Tensor, filepath: str) -> None:
    check_write_file(path=filepath)
    if tensor.dim() == 3 and tensor.shape[0] == 3 and tensor.dtype == torch.float32:
        torchvision.utils.save_image(tensor=tensor, fp=filepath)
    elif tensor.dim() == 2 and tensor.dtype == torch.uint8:
        Image.fromarray(tensor.numpy()).save(filepath)
    else:
        raise NotImplementedError(f"Unrecognized tensor format: shape={tensor.shape}, dtype={tensor.dtype}.")


def serialize_tensor(obj: Any):
    return apply_tensor_op(func=lambda x: x.detach().tolist(), inputs=obj)


def save_json(obj: Any, filepath: str) -> None:
    assert os.path.isdir(os.path.dirname(filepath)), f"{filepath=}"
    obj = serialize_tensor(obj)
    with open(filepath, mode='w') as f:
        f.write(jsbeautifier.beautify(json.dumps(obj), jsbeautifier.default_options()))
