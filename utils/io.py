from typing import Any
import os
import json
import jsbeautifier
import numpy
import torch
from PIL import Image
from .ops import apply_tensor_op


def load_image(filepath: str, dtype: torch.dtype) -> torch.Tensor:
    assert type(filepath) == str, f"{type(filepath)=}"
    assert os.path.isfile(filepath), f"{filepath=}"
    assert type(dtype) == torch.dtype, f"{type(dtype)=}"
    assert dtype in [torch.float32, torch.uint8], f"{dtype=}"
    image = Image.open(filepath)
    assert image.mode == "RGB"
    image = torch.tensor(numpy.array(image)).permute(2, 0, 1)
    assert len(image.shape) == 3 and image.shape[0] == 3, f"{image.shape=}"
    assert image.dtype == torch.uint8, f"{image.dtype=}, {filepath=}"
    if dtype == torch.float32:
        image = image.type(torch.float32) / 255
    return image


def serialize_tensor(obj: Any):
    return apply_tensor_op(func=lambda x: x.detach().tolist(), inputs=obj)


def save_json(obj: Any, filepath: str) -> None:
    assert os.path.isdir(os.path.dirname(filepath)), f"{filepath=}"
    obj = serialize_tensor(obj)
    with open(filepath, mode='w') as f:
        f.write(jsbeautifier.beautify(json.dumps(obj), jsbeautifier.default_options()))
