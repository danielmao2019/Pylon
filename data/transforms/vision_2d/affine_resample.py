from typing import Optional, Tuple

import torch

from data.transforms.base_transform import BaseTransform

SAMPLING_MODE_BOX = "box"
SAMPLING_MODE_TRIANGLE = "triangle"
SAMPLING_MODE_CUBIC = "cubic"
# The source is read as if surrounded by zeros, so a window falls off to zero as it leaves the source.
FILL_RULE_ZERO = "zero"
# The source is read as a picture with a boundary, so an output pixel is the mean of the part of its footprint inside it.
FILL_RULE_RENORMALIZE = "renormalize"
# What an eight-bit sample accumulated in a thirty-two-bit integer leaves for the weights carried against it.
_WEIGHT_PRECISION_BITS = 32 - 8 - 2

# How far off its structural zero a map's shear and perspective entries may sit
# and still be the axis-aligned map this resamples: a shear of a ten-thousandth
# of a pixel per pixel is what float32 round-off leaves behind, three orders of
# magnitude under the smallest rotation or perspective any caller means.
_AXIS_ALIGNED_TOLERANCE = 1e-4


class AffineResample(BaseTransform):
    """Places a raster onto the grid an axis-aligned affine map carries it to, each output pixel gathering along each axis the source samples its kernel weights."""

    def __init__(
        self,
        transform: torch.Tensor,
        output_resolution: Tuple[int, int],
        sampling_mode: str,
        fill_rule: str,
    ) -> None:
        """Configures one placement: the map source pixel coordinates travel, the window that map is rasterized over, the kernel weighting the samples each output pixel gathers, and how the source's own boundary is read.

        Args:
            transform (torch.Tensor): (3, 3) floating-point homogeneous map carrying source pixel coordinates (x, y, 1) onto the output window's own pixel coordinates, integer indices being pixel centres on both sides; its linear part must be diagonal, i.e. one scale and one translation per axis.
            output_resolution (Tuple[int, int]): (height, width), in pixels, of the window the map is rasterized over.
            sampling_mode (str): One of SAMPLING_MODE_BOX, SAMPLING_MODE_TRIANGLE, SAMPLING_MODE_CUBIC.
            fill_rule (str): One of FILL_RULE_ZERO, FILL_RULE_RENORMALIZE, fixing what the part of a footprint lying past the source contributes.

        Returns:
            None
        """
        assert isinstance(
            transform, torch.Tensor
        ), f"Expected `transform` to be a torch tensor. {type(transform)=}"
        assert transform.shape == (
            3,
            3,
        ), f"Expected `transform` to be a `(3, 3)` homogeneous map. {transform.shape=}"
        assert (
            transform.dtype.is_floating_point
        ), f"Expected `transform` to have a floating-point dtype. {transform.dtype=}"
        # Each shear entry read against its own axis's scale, the dimensionless
        # shear the linear part carries.
        assert abs(float(transform[0, 1])) <= _AXIS_ALIGNED_TOLERANCE * abs(
            float(transform[0, 0])
        ) and abs(float(transform[1, 0])) <= _AXIS_ALIGNED_TOLERANCE * abs(
            float(transform[1, 1])
        ), f"Expected each axis to carry one scale and one translation, but the linear part is not diagonal. {transform=} {_AXIS_ALIGNED_TOLERANCE=}"
        assert (
            abs(float(transform[2, 0])) <= _AXIS_ALIGNED_TOLERANCE
            and abs(float(transform[2, 1])) <= _AXIS_ALIGNED_TOLERANCE
            and abs(float(transform[2, 2]) - 1.0) <= _AXIS_ALIGNED_TOLERANCE
        ), f"Expected `transform` to be homogeneous, but its last row is not homogeneous. {transform=} {_AXIS_ALIGNED_TOLERANCE=}"
        assert (
            transform[0, 0] != 0 and transform[1, 1] != 0
        ), f"Expected each axis to carry a nonzero scale, but an axis carries no scale. {transform=}"
        assert isinstance(
            output_resolution, tuple
        ), f"Expected `output_resolution` to be a tuple. {type(output_resolution)=}"
        assert (
            len(output_resolution) == 2
        ), f"Expected `output_resolution` to be an `(H, W)` pair. {len(output_resolution)=}"
        assert all(
            isinstance(extent, int) and extent > 0 for extent in output_resolution
        ), f"Expected `output_resolution` to hold positive integer extents. {output_resolution=}"
        assert sampling_mode in {
            SAMPLING_MODE_BOX,
            SAMPLING_MODE_TRIANGLE,
            SAMPLING_MODE_CUBIC,
        }, f"Expected `sampling_mode` to be one of the three modes this module declares. {sampling_mode=}"
        assert fill_rule in {
            FILL_RULE_ZERO,
            FILL_RULE_RENORMALIZE,
        }, f"Expected `fill_rule` to be one of the two rules this module declares. {fill_rule=}"

        # Where each source coordinate lands, which fixes nothing about how large a window is rasterized.
        self.transform = transform
        # How large that window is, which the map on its own leaves open.
        self.output_resolution = output_resolution
        # Which source samples an output pixel gathers, and under what weights.
        self.sampling_mode = sampling_mode
        # What the part of that footprint lying past the source contributes, which the kernel leaves open.
        self.fill_rule = fill_rule

    def _call_single(
        self, raster: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Resamples one raster under the configured placement, so the base applies that one placement unchanged to every raster it is handed.

        Args:
            raster (torch.Tensor): (..., H, W) source raster of torch.bool, of any integer dtype or of any floating-point dtype, whose last two dimensions are the height this map's y axis addresses and the width its x axis addresses.
            generator (Optional[torch.Generator]): The randomness source the base hands every transform, which this placement is fixed by its configured map alone and draws nothing from.

        Returns:
            torch.Tensor: (..., out_h, out_w) resampled raster, in the raster's own dtype.
        """
        assert isinstance(
            raster, torch.Tensor
        ), f"Expected `raster` to be a torch tensor. {type(raster)=}"
        assert (
            raster.ndim >= 2
        ), f"Expected `raster` to carry at least the two resampled axes. {raster.shape=}"

        # Separable, so the two axes gather independently.
        scales = torch.diagonal(self.transform)[:2]
        translations = self.transform[:2, 2]

        if raster.dtype == torch.bool:
            # A boolean raster carries no weighted sum, so it is placed only under the kernel whose footprint holds one source sample and whose gather is therefore a selection.
            assert (
                self.sampling_mode == SAMPLING_MODE_BOX
            ), f"Expected a boolean raster to be placed only under the selecting kernel. {self.sampling_mode=} {raster.dtype=}"

        result = raster
        for axis in range(2):
            sample_indices, weights = self._axis_gather_weights(
                source_extent=result.shape[-1 - axis],
                output_extent=self.output_resolution[1 - axis],
                scale=float(scales[axis]),
                translation=float(translations[axis]),
                device=result.device,
            )
            # Each axis gathers what the previous pass returned, so an eight-bit raster is back on its own levels before the second axis reads it.
            result = self._axis_gather(
                raster=result,
                axis=axis,
                sample_indices=sample_indices,
                weights=weights,
            )

        return result.to(raster.dtype)

    def _axis_gather_weights(
        self,
        source_extent: int,
        output_extent: int,
        scale: float,
        translation: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds one axis's gather: which source indices each output index reads, and the weight it reads each one under.

        Args:
            source_extent (int): Number of source samples along this axis.
            output_extent (int): Number of output samples along this axis.
            scale (float): This axis's entry on the map's diagonal, carrying a source coordinate to an output coordinate.
            translation (float): This axis's entry in the map's last column, in output pixels.
            device (torch.device): The device the raster being gathered sits on, which the indices and weights are built on so the gather never crosses devices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (output_extent, K) int64 source indices each output index reads, and the (output_extent, K) float64 weights it reads them under, both on `device`.
        """
        output_index = torch.arange(output_extent, dtype=torch.float64, device=device)
        # This axis's own inverse map, formed once and applied, rather than a division taken at every output index; the pair is what inverting this axis's block of the map returns.
        inverse_scale = 1.0 / scale
        inverse_translation = -translation * inverse_scale
        # Integer indices are pixel centres, so a caller's own centre convention rides in the transform it supplies.
        source_coordinate = output_index * inverse_scale + inverse_translation
        widening, half_width = self._kernel_footprint(scale=scale)
        sample_count = int(2.0 * half_width) + 1
        sample_indices = (
            torch.ceil(source_coordinate - half_width).unsqueeze(1)
            + torch.arange(sample_count, device=device)
        ).to(torch.int64)
        # Into the kernel's own units, so a widened kernel is evaluated on its own shape rather than a stretched one.
        offsets = (sample_indices - source_coordinate.unsqueeze(1)) / widening
        weights = self._kernel_weights(offsets=offsets)
        sample_inside = (sample_indices >= 0) & (sample_indices < source_extent)
        if self.fill_rule == FILL_RULE_ZERO:
            # Summing to one over the footprint, the samples past the source extent included.
            weights = weights / weights.sum(dim=1, keepdim=True)
            # What those samples carry is the zero the source is read as surrounded by.
            weights = torch.where(sample_inside, weights, torch.zeros_like(weights))
        if self.fill_rule == FILL_RULE_RENORMALIZE:
            weights = torch.where(sample_inside, weights, torch.zeros_like(weights))
            # An output pixel inside the source is the mean of the part of its footprint it can see.
            weights = weights / weights.sum(dim=1, keepdim=True)
            # Such a pixel sees nothing to average, and this is what carries a reference's crop past its own resized frame. The source's own extent is read on the output grid the map images it onto, which is the grid that crop is taken in and is the same test in either direction of the scale.
            imaged_lower = -0.5 * scale + translation
            imaged_upper = (source_extent - 0.5) * scale + translation
            coordinate_inside = (output_index >= min(imaged_lower, imaged_upper)) & (
                output_index < max(imaged_lower, imaged_upper)
            )
            weights = torch.where(
                coordinate_inside.unsqueeze(1), weights, torch.zeros_like(weights)
            )
        return sample_indices, weights

    def _axis_gather(
        self,
        raster: torch.Tensor,
        axis: int,
        sample_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Contracts one axis of a raster against that axis's gather, an integer raster in its own fixed point and every other raster in the float64 the weights are carried in.

        Args:
            raster (torch.Tensor): (..., H, W) raster this pass reads, of torch.bool, of any integer dtype or of any floating-point dtype.
            axis (int): 0 for the width this map's x axis addresses, 1 for the height its y axis addresses.
            sample_indices (torch.Tensor): (output_extent, K) int64 source indices along this axis, on the raster's own device.
            weights (torch.Tensor): (output_extent, K) float64 weights those indices are read under, on the raster's own device.

        Returns:
            torch.Tensor: The raster with this axis replaced by its output_extent gathered samples, in torch.bool for a boolean raster, in the raster's own dtype for an integer one, and in float64 otherwise.
        """
        dim = -1 - axis
        source_extent = raster.shape[dim]
        # Out-of-extent indices already carry zero weight, so clamping them only keeps the gather itself legal.
        gathered = raster.movedim(dim, -1)[
            ..., sample_indices.clamp(min=0, max=source_extent - 1)
        ]
        if raster.dtype == torch.bool:
            # A selection, so the truth value that comes back is one the source already carried.
            selected = torch.logical_and(gathered, weights > 0.0).any(dim=-1)
            return selected.movedim(-1, dim)
        if not raster.dtype.is_floating_point:
            scaled = weights * float(1 << _WEIGHT_PRECISION_BITS)
            # Carried onto integers of two to the minus _WEIGHT_PRECISION_BITS, each rounded away from zero as an eight-bit resample rounds its own.
            quantized = torch.where(
                scaled >= 0.0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5)
            ).to(torch.int64)
            accumulated = (gathered.to(torch.int64) * quantized).sum(dim=-1) + (
                1 << (_WEIGHT_PRECISION_BITS - 1)
            )
            # This pass returns the raster's own dtype, so what the next axis gathers sits on those levels rather than on a continuum.
            levels = torch.iinfo(raster.dtype)
            contracted = (accumulated >> _WEIGHT_PRECISION_BITS).clamp(
                min=levels.min, max=levels.max
            )
            return contracted.to(raster.dtype).movedim(-1, dim)
        return (gathered * weights).sum(dim=-1).movedim(-1, dim)

    def _kernel_footprint(self, scale: float) -> Tuple[float, float]:
        """Returns this mode's footprint on one axis: the factor its support is widened by, and the half-width it gathers over once widened.

        Args:
            scale (float): This axis's entry on the map's diagonal, whose magnitude is below one under a reduction and whose sign mirrors the axis.

        Returns:
            Tuple[float, float]: The widening factor, and the half-width in source samples the gather covers.
        """
        if self.sampling_mode == SAMPLING_MODE_BOX:
            return 1.0, 0.5
        if self.sampling_mode == SAMPLING_MODE_TRIANGLE:
            return 1.0, 1.0
        if self.sampling_mode == SAMPLING_MODE_CUBIC:
            # The reduction is the magnitude of the scale, so an axis the map mirrors widens by what that axis reduces by.
            widening = max(1.0 / abs(scale), 1.0)
            return widening, 2.0 * widening
        assert 0, "Should not reach here. " f"{self.sampling_mode=}"

    def _kernel_weights(self, offsets: torch.Tensor) -> torch.Tensor:
        """Evaluates this mode's kernel at each sample's offset from its own output pixel's source coordinate, in the kernel's own units.

        Args:
            offsets (torch.Tensor): (output_extent, K) float64 offsets, in the kernel's own units.

        Returns:
            torch.Tensor: (output_extent, K) float64 unnormalized kernel weights.
        """
        if self.sampling_mode == SAMPLING_MODE_BOX:
            return torch.logical_and(offsets >= -0.5, offsets < 0.5).to(offsets.dtype)
        if self.sampling_mode == SAMPLING_MODE_TRIANGLE:
            return (1.0 - offsets.abs()).clamp(min=0.0)
        if self.sampling_mode == SAMPLING_MODE_CUBIC:
            magnitude = offsets.abs()
            # The Keys cubic at a = -0.5, written out on each half of its support of two.
            near = ((1.5 * magnitude - 2.5) * magnitude) * magnitude + 1.0
            far = ((-0.5 * magnitude + 2.5) * magnitude - 4.0) * magnitude + 2.0
            return torch.where(
                magnitude < 1.0,
                near,
                torch.where(magnitude < 2.0, far, torch.zeros_like(magnitude)),
            )
        assert 0, "Should not reach here. " f"{self.sampling_mode=}"
