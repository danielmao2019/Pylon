from typing import List, Sequence, Dict, Union, Any, Optional
import os
import json
import jsbeautifier
import numpy
import torch
import torchvision
from PIL import Image
import rasterio

from utils.input_checks import check_read_file, check_write_file
from utils.ops import apply_tensor_op, transpose_buffer, buffer_mean

from data.transforms.resize import ResizeMaps


def load_image(
    filepath: Optional[str] = None,
    filepaths: Optional[List[str]] = None,
    dtype: Optional[torch.dtype] = None,
    sub: Optional[Union[float, Sequence[float]]] = None,
    div: Optional[Union[float, Sequence[float]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    """
    Load an image or bands from file(s) into a PyTorch tensor.

    Args:
        filepath: Path to a single image file (.png, .jpg, .jpeg).
        filepaths: List of filepaths for bands in a .tif image.
        dtype: Desired output data type for the tensor (e.g., torch.float32).
        sub: Value(s) to subtract from the image for normalization.
        div: Value(s) to divide the image by for normalization.
        height: Desired height for resizing the image.
        width: Desired width for resizing the image.

    Returns:
        A PyTorch tensor of the loaded image.
    """
    # Validate inputs
    assert (filepath is None) ^ (filepaths is None), \
        "Exactly one of 'filepath' or 'filepaths' must be provided."
    if filepath is not None:
        check_read_file(path=filepath, ext=['.png', '.jpg', '.jpeg'])
    if filepaths is not None:
        assert isinstance(filepaths, list), \
            f"'filepaths' must be a list. Got {type(filepaths)}."
        for path in filepaths:
            check_read_file(path=path, ext=['.tif', '.tiff'])
    assert (height is None and width is None) or (height is not None and width is not None), \
        "Both 'height' and 'width' must be provided for resizing."

    # Load image data
    if filepath:
        image: torch.Tensor = _load_image(filepath)
    else:
        image: torch.Tensor = _load_multispectral_image(filepaths, height, width)

    # Normalize the image
    if sub is not None or div is not None:
        image = _normalize(image, sub=sub, div=div)

    # Resize the image
    if height is not None and width is not None:
        resize_op = ResizeMaps(size=(height, width))
        image = resize_op(image)

    # Convert data type
    if dtype is not None:
        assert isinstance(dtype, torch.dtype), \
            f"'dtype' must be a torch.dtype, got {type(dtype)}."
        image = image.to(dtype)

    return image


def _load_image(filepath: str) -> torch.Tensor:
    """
    Load a single image file into a PyTorch tensor.

    Args:
        filepath: Path to the image file.

    Returns:
        A PyTorch tensor representing the image.
    """
    image = Image.open(filepath)
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


def _load_multispectral_image(
    filepaths: List[str],
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    """
    Load multiple bands from separate .tif files into a single tensor.

    Args:
        filepaths: List of file paths to .tif band files.
        height: Desired height for resizing (optional).
        width: Desired width for resizing (optional).

    Returns:
        A PyTorch tensor where each band is a channel.
    """
    bands: List[torch.Tensor] = []
    for filepath in filepaths:
        with rasterio.open(filepath) as src:
            band = src.read(1)
        if band.dtype == numpy.uint16:
            band = band.astype(numpy.int64)
        band = torch.from_numpy(band).unsqueeze(0)
        assert band.ndim == 3 and band.size(0) == 1, f"{band.shape=}"
        if height is not None and width is not None:
            band = torchvision.transforms.functional.resize(band, size=(height, width))
        bands.append(band)
    return torch.cat(bands, dim=0)  # Concatenate bands along the channel dimension


def _normalize(image: torch.Tensor, sub: Optional[Union[float, Sequence[float]]], div: Optional[Union[float, Sequence[float]]]) -> torch.Tensor:
    """Normalize the image by subtracting and dividing channel-wise values."""
    image = image.to(torch.float32)

    def prepare_tensor(value, num_channels, ndim):
        """Helper to prepare broadcastable normalization tensors."""
        value_tensor = torch.tensor(value, dtype=torch.float32)
        if value_tensor.numel() not in {1, num_channels}:
            raise ValueError(f"Normalization value must match the number of channels or be scalar. Got {value_tensor.numel()=}.")
        if value_tensor.numel() == 1:
            value_tensor = value_tensor.repeat(num_channels)
        if ndim == 3:  # For 3D tensors (C, H, W), reshape for broadcasting
            return value_tensor.view(-1, 1, 1)
        return value_tensor  # Return as is for 2D tensors (H, W)

    if sub is not None:
        sub_tensor = prepare_tensor(
            value=sub, num_channels=image.shape[0] if image.ndim == 3 else 1, ndim=image.ndim,
        )
        image -= sub_tensor

    if div is not None:
        div_tensor = prepare_tensor(
            value=div, num_channels=image.shape[0] if image.ndim == 3 else 1, ndim=image.ndim,
        )
        div_tensor = div_tensor.clamp(min=1e-6)  # Prevent division by zero
        image /= div_tensor

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
