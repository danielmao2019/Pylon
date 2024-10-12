from typing import List, Dict, Any, Optional
import os
import json
import jsbeautifier
import numpy
import torch
import torchvision
from PIL import Image
from .input_checks import check_read_file, check_write_file
from .ops import apply_tensor_op, transpose_buffer, average_buffer


def load_image(filepath: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    # input checks
    check_read_file(path=filepath)
    assert type(dtype) == torch.dtype, f"{type(dtype)=}"
    # read from disk
    image = Image.open(filepath)
    mode = image.mode
    image = torch.from_numpy(numpy.array(image))
    # convert to torch.Tensor
    if mode == 'RGB':
        image = image.permute(2, 0, 1)
        assert image.dim() == 3 and image.shape[0] == 3, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}, {filepath=}"
    elif mode == 'L':
        assert image.dim() == 2, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}, {filepath=}"
    elif mode == 'I':
        assert image.dim() == 2, f"{image.shape=}"
        assert image.dtype == torch.int32, f"{image.dtype=}, {filepath=}"
    else:
        raise NotImplementedError(f"{mode=}")
    # convert data type
    image = image.type(dtype)
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
        average_buffer([avg_scores[i][j] for i in range(m)]) for j in range(n)
    ]
    avg_scores: Dict[str, List[Any]] = transpose_buffer(avg_scores)
    return avg_scores
