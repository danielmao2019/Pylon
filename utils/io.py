from typing import Literal, List, Sequence, Dict, Union, Any, Optional
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
    sub: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    div: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    normalization: Optional[Literal["min-max", "mean-std"]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Load an image or bands from file(s) into a PyTorch tensor.

    Args:
        filepath: Path to a single image file (.png, .jpg, .jpeg, .bmp).
        filepaths: List of filepaths for bands in a .tif image or a single .tif file.
        height: Optional target height for resizing the image.
        width: Optional target width for resizing the image.
        sub: Scalar, sequence, or tensor to subtract from the image for normalization.
        div: Scalar, sequence, or tensor to divide the image by for normalization.
        normalization: Normalization method to apply:
            - "min-max": Rescales pixel values to [0, 1].
            - "mean-std": Standardizes the image using mean and standard deviation.
            Cannot be used together with `sub` or `div`.
        dtype: Desired output tensor data type (e.g., torch.float32).

    Returns:
        A PyTorch tensor containing the loaded image data.
    """
    # Ensure only one input type is provided
    if (filepath is not None) == (filepaths is not None):
        raise ValueError("Exactly one of 'filepath' or 'filepaths' must be provided.")

    # Load image data
    if filepath:
        image: torch.Tensor = _load_image(filepath)
    else:
        image: torch.Tensor = _load_multispectral_image(filepaths, height, width)

    # Apply normalization
    image = _normalize(image, sub=sub, div=div, normalization=normalization)

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
    # input checks
    check_read_file(path=filepath, ext=['.png', '.jpg', '.jpeg', '.bmp'])

    # load image
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
    filepaths: Union[str, List[str]],
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
    # input checks
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    assert isinstance(filepaths, list), \
        f"'filepaths' must be a list. Got {type(filepaths)}."
    for path in filepaths:
        check_read_file(path=path, ext=['.tif', '.tiff'])
    if len(filepaths) == 0:
        raise ValueError(f"Provided list of file paths is empty.")
    assert height is None or isinstance(height, int)
    assert width is None or isinstance(width, int)
    if len(filepaths) == 1 and (height is not None or width is not None):
        raise ValueError("Height and width should be None when loading a single file.")

    # load multi-spectral image
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
    normalization: Optional[str] = "min-max",
) -> torch.Tensor:
    """Normalize the image using subtraction, division, min-max, or mean-std normalization."""
    # input checks
    assert isinstance(image, torch.Tensor)
    assert normalization in {None, "min-max", "mean-std"}, f"Invalid normalization method: {normalization}"
    assert all((
        arg is None or isinstance(arg, (float, list, tuple, torch.Tensor))
    ) for arg in [sub, div])
    if normalization and (sub is not None or div is not None):
        raise ValueError("'normalization' cannot be used together with 'sub' or 'div'.")

    # If no normalization is required, return as is
    if sub is None and div is None and normalization is None:
        return image

    # normalize image
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
        return value_tensor.view(-1, 1, 1) if ndim == 3 else value_tensor

    if sub is not None:
        sub_tensor = prepare_tensor(sub, image.size(0) if image.ndim == 3 else 1, image.ndim)
        image = image - sub_tensor

    if div is not None:
        div_tensor = prepare_tensor(div, image.size(0) if image.ndim == 3 else 1, image.ndim)
        div_tensor = torch.where(div_tensor.abs() < 1.0e-6, 1.0e-6, div_tensor)  # Avoid division by zero issues
        image = image / div_tensor

    if normalization == "min-max":
        min_val, max_val = image.min(), image.max()
        image = (image - min_val) / (max_val - min_val + 1e-6)  # Avoid division by zero

    elif normalization == "mean-std":
        mean_val, std_val = image.mean(), image.std()
        std_val = torch.where(std_val < 1.0e-6, 1.0e-6, std_val)  # Avoid division by zero
        image = (image - mean_val) / std_val

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
