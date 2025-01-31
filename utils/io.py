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


def load_image(
    filepath: Optional[str] = None,
    filepaths: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    sub: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    div: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Load an image or bands from file(s) into a PyTorch tensor.

    Args:
        filepath: Path to a single image file (.png, .jpg, .jpeg, .bmp).
        filepaths: List of filepaths for bands in a .tif image or a single .tif file.
        height: Desired height for resizing bands (optional).
        width: Desired width for resizing bands (optional).
        dtype: Desired output data type for the tensor (e.g., torch.float32).
        sub: Value(s) to subtract from the image for normalization.
        div: Value(s) to divide the image by for normalization.

    Returns:
        A PyTorch tensor of the loaded image.
    """
    # Ensure only one input type is provided
    if (filepath is not None) == (filepaths is not None):
        raise ValueError("Exactly one of 'filepath' or 'filepaths' must be provided.")
    if filepath is not None:
        check_read_file(path=filepath, ext=['.png', '.jpg', '.jpeg', '.bmp'])

    # Normalize filepaths to a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]  # Convert single .tif file to list for consistency
    if filepaths is not None:
        assert isinstance(filepaths, list), \
            f"'filepaths' must be a list. Got {type(filepaths)}."
        for path in filepaths:
            check_read_file(path=path, ext=['.tif', '.tiff'])

    # Load image data
    if filepath:
        image: torch.Tensor = _load_image(filepath)
    else:
        image: torch.Tensor = _load_multispectral_image(filepaths, height, width)

    # Apply normalization
    if sub is not None or div is not None:
        image = _normalize(image, sub=sub, div=div)

    # Convert data type if specified
    if dtype is not None:
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"'dtype' must be a torch.dtype, got {type(dtype)}.")
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
    if not isinstance(filepaths, list) or len(filepaths) == 0:
        raise ValueError(f"Invalid filepaths: {filepaths}")

    if len(filepaths) == 1 and (height is not None or width is not None):
        raise ValueError("Height and width should be None when loading a single file.")

    bands: List[torch.Tensor] = []

    for path in filepaths:
        with rasterio.open(path) as src:
            band = src.read()
        if band.dtype == numpy.uint16:
            band = band.astype(numpy.int64)

        band_tensor = torch.from_numpy(band)

        if band_tensor.ndim != 3:  # Ensure correct shape
            raise ValueError(f"Unexpected shape {band_tensor.shape} for file {path}")

        bands.append(band_tensor)

    try:
        return torch.cat(bands, dim=0)  # Concatenating along the channel dimension
    except RuntimeError:
        # Determine target height and width if not provided
        if height is None:
            height = max(band.size(1) for band in bands)  # Use max instead of mean
        if width is None:
            width = max(band.size(2) for band in bands)

        # Resize bands and concatenate
        resized_bands = [torchvision.transforms.functional.resize(band, size=(height, width)) for band in bands]
        return torch.cat(resized_bands, dim=0)


def _normalize(
    image: torch.Tensor,
    sub: Optional[Union[float, Sequence[float], torch.Tensor]],
    div: Optional[Union[float, Sequence[float], torch.Tensor]],
) -> torch.Tensor:
    """Normalize the image by subtracting and dividing channel-wise values."""
    original_dtype = image.dtype
    image = image.to(torch.float32)

    def prepare_tensor(value, num_channels, ndim):
        """Prepare a broadcastable normalization tensor."""
        value_tensor = torch.as_tensor(value, dtype=torch.float32)

        if value_tensor.numel() not in {1, num_channels}:
            raise ValueError(
                f"Normalization value must match the number of channels or be scalar. Got {value_tensor.numel()} values for {num_channels} channels."
            )
        if value_tensor.numel() == 1:
            value_tensor = value_tensor.expand(num_channels)
        if ndim == 3:  # Reshape for broadcasting
            return value_tensor.view(-1, 1, 1)
        return value_tensor  # Keep shape for 2D tensors (H, W)

    if sub is not None:
        sub_tensor = prepare_tensor(sub, image.size(0) if image.ndim == 3 else 1, image.ndim)
        image = image - sub_tensor

    if div is not None:
        div_tensor = prepare_tensor(div, image.size(0) if image.ndim == 3 else 1, image.ndim)
        div_tensor = torch.where(div_tensor.abs() < 1.0e-06, 1.0e-06, div_tensor)  # Avoids division by zero issues
        image = image / div_tensor

    return image.to(original_dtype)


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
