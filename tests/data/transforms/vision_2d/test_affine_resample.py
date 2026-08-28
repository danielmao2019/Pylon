import pytest
import torch

from data.transforms.vision_2d.affine_resample import (
    _AXIS_ALIGNED_TOLERANCE,
    FILL_RULE_RENORMALIZE,
    FILL_RULE_ZERO,
    SAMPLING_MODE_BOX,
    SAMPLING_MODE_CUBIC,
    SAMPLING_MODE_TRIANGLE,
    AffineResample,
)


def test_each_sampling_mode_gathers_over_its_own_support() -> None:
    """The three modes differ in the kernel alone, so under one identity-scaled map each output pixel reads exactly the source samples its own kernel's support reaches and no others.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.rand(size=(3, 11), dtype=torch.float64)
    translation = 0.25
    transform = torch.tensor(
        [[1.0, 0.0, translation], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    for sampling_mode, half_width in (
        (SAMPLING_MODE_BOX, 0.5),
        (SAMPLING_MODE_TRIANGLE, 1.0),
        (SAMPLING_MODE_CUBIC, 2.0),
    ):
        resample = AffineResample(
            transform=transform,
            output_resolution=(3, 11),
            sampling_mode=sampling_mode,
            fill_rule=FILL_RULE_ZERO,
        )
        baseline = resample(raster)
        for source_index in range(raster.shape[1]):
            perturbed = raster.clone()
            perturbed[:, source_index] += 1.0
            changed = resample(perturbed) != baseline
            for output_index in range(baseline.shape[1]):
                source_coordinate = output_index - translation
                inside = abs(source_index - source_coordinate) < half_width
                assert bool(changed[:, output_index].any()) == inside, (
                    "An output pixel must change exactly when the moved source "
                    "sample lies inside its own kernel's support. "
                    f"{sampling_mode=}, {half_width=}, {source_index=}, "
                    f"{output_index=}, {source_coordinate=}, {inside=}"
                )
    return


def test_only_the_cubic_support_widens_with_the_reduction() -> None:
    """Only the kernel whose support widens with the reduction reads further than its own half-width, so a reducing map spreads the cubic's footprint while leaving the other two where they were.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.rand(size=(3, 48), dtype=torch.float64)
    scale = 0.25
    translation = 0.0625
    transform = torch.tensor(
        [[scale, 0.0, translation], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    for sampling_mode, half_width in (
        (SAMPLING_MODE_BOX, 0.5),
        (SAMPLING_MODE_TRIANGLE, 1.0),
        (SAMPLING_MODE_CUBIC, 2.0 / scale),
    ):
        resample = AffineResample(
            transform=transform,
            output_resolution=(3, 12),
            sampling_mode=sampling_mode,
            fill_rule=FILL_RULE_ZERO,
        )
        baseline = resample(raster)
        for source_index in range(raster.shape[1]):
            perturbed = raster.clone()
            perturbed[:, source_index] += 1.0
            changed = resample(perturbed) != baseline
            for output_index in range(baseline.shape[1]):
                source_coordinate = (output_index - translation) / scale
                inside = abs(source_index - source_coordinate) < half_width
                assert bool(changed[:, output_index].any()) == inside, (
                    "An output pixel must change exactly when the moved source "
                    "sample lies inside its own kernel's support. "
                    f"{sampling_mode=}, {half_width=}, {source_index=}, "
                    f"{output_index=}, {source_coordinate=}, {inside=}"
                )
    return


def test_the_output_window_is_the_configured_resolution_rather_than_the_maps_own() -> (
    None
):
    """The map fixes where content lands and the configured window fixes how much of that grid is rasterized, so a larger window carries the smaller one's result in its own corner.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.rand(size=(16, 16), dtype=torch.float64)
    transform = torch.tensor(
        [[0.5, 0.0, 0.25], [0.0, 0.5, 0.25], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    results = []
    for output_resolution in ((5, 5), (8, 8)):
        resample = AffineResample(
            transform=transform,
            output_resolution=output_resolution,
            sampling_mode=SAMPLING_MODE_CUBIC,
            fill_rule=FILL_RULE_ZERO,
        )
        results.append(resample(raster))
    assert results[0].shape == (
        5,
        5,
    ), f"The placement must span the window it was configured with. {results[0].shape=}"
    assert results[1].shape == (
        8,
        8,
    ), f"The placement must span the window it was configured with. {results[1].shape=}"
    assert torch.equal(results[0], results[1][:5, :5]), (
        "The window's size must not move where the map lands its content, so the "
        "smaller window must be the larger one's own corner. "
        f"{results[0]=}, {results[1]=}"
    )
    return


def test_the_zero_fill_rule_reads_zero_past_the_source_extent() -> None:
    """Under the zero rule the fill is this resample's own rather than the map's, so the part of a window that reaches past the source raster is fabricated zeros carrying no record of the source.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.rand(size=(10, 10), dtype=torch.float64) + 1.0
    transform = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    resample = AffineResample(
        transform=transform,
        output_resolution=(16, 16),
        sampling_mode=SAMPLING_MODE_CUBIC,
        fill_rule=FILL_RULE_ZERO,
    )
    result = resample(raster)
    # The very same window over a source that actually continues past the ten-by-ten one's edge.
    extended = torch.rand(size=(20, 20), dtype=torch.float64) + 1.0
    extended[:10, :10] = raster
    unclipped = resample(extended)

    assert torch.equal(
        result[12:, :], torch.zeros_like(result[12:, :])
    ), f"Every footprint from row 12 on falls wholly outside the source, {result[12:, :]=}"
    assert torch.equal(
        result[:, 12:], torch.zeros_like(result[:, 12:])
    ), f"Every footprint from column 12 on falls wholly outside the source, {result[:, 12:]=}"
    assert torch.equal(
        result[2:8, 2:8], unclipped[2:8, 2:8]
    ), f"A footprint wholly inside the source reads what it would read of a source that continues, {result[2:8, 2:8]=}, {unclipped[2:8, 2:8]=}"

    # A constant source, so what a straddling footprint reads is the edge value's own fraction.
    edge_value = 1.5
    straddling = resample(
        torch.full(size=(10, 10), fill_value=edge_value, dtype=torch.float64)
    )
    assert bool(
        torch.all(straddling[10, 2:8] < edge_value)
    ), f"A footprint straddling the source's edge reads below that edge's own value, {straddling[10, 2:8]=}, {edge_value=}"
    return


def test_the_renormalize_fill_rule_averages_the_part_of_the_footprint_it_can_see() -> (
    None
):
    """Under the renormalize rule the source is read as a picture with a boundary, so a pixel straddling that boundary is the mean of the samples inside it rather than a value the ones outside dimmed.

    Args:
        None.

    Returns:
        None.
    """
    constant = 1.5
    raster = torch.full(size=(10, 10), fill_value=constant, dtype=torch.float64)
    transform = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    resample = AffineResample(
        transform=transform,
        output_resolution=(16, 16),
        sampling_mode=SAMPLING_MODE_CUBIC,
        fill_rule=FILL_RULE_RENORMALIZE,
    )
    result = resample(raster)

    # The source's own extent, its ten samples each half a pixel wide, carried onto the output grid the map images it on; an output index inside that is one lying inside the picture.
    indices = torch.arange(16, dtype=torch.float64)
    centre_inside = (indices >= -0.5 * 1.0 + 0.5) & (indices < (10 - 0.5) * 1.0 + 0.5)
    inside = centre_inside.unsqueeze(1) & centre_inside.unsqueeze(0)
    assert torch.allclose(
        result[inside], torch.full_like(result[inside], constant)
    ), f"A constant source comes back at that same constant wherever the mapped centre falls inside, {result[inside]=}, {constant=}"
    assert torch.equal(
        result[~inside], torch.zeros_like(result[~inside])
    ), f"A pixel whose mapped centre falls outside the source sees nothing to average, {result[~inside]=}"
    return


def test_the_two_fill_rules_agree_wherever_every_footprint_stays_inside_the_source() -> (
    None
):
    """The rules differ only in what the part of a footprint past the source contributes, so a window and support clear of the source's boundary place a raster identically under either.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.rand(size=(20, 20), dtype=torch.float64)
    # Every output index's centre lands at least four samples inside the source, so no support of two reaches its boundary.
    transform = torch.tensor(
        [[1.0, 0.0, -4.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    results = []
    for fill_rule in (FILL_RULE_ZERO, FILL_RULE_RENORMALIZE):
        resample = AffineResample(
            transform=transform,
            output_resolution=(8, 8),
            sampling_mode=SAMPLING_MODE_CUBIC,
            fill_rule=fill_rule,
        )
        results.append(resample(raster))
    assert torch.equal(results[0], results[1]), (
        "The two fill rules must agree wherever no footprint reaches past the "
        "source, nothing being renormalized there. "
        f"{results[0]=}, {results[1]=}"
    )
    return


def test_an_integer_raster_lands_back_on_its_own_levels_between_the_two_axes() -> None:
    """An eight-bit raster resamples in eight-bit fixed point an axis at a time, so the first axis's result is itself an eight-bit raster and the second axis gathers those levels rather than a continuum.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.randint(low=0, high=256, size=(24, 32), dtype=torch.uint8)
    scale_x, scale_y = 0.5, 0.25
    translation_x, translation_y = -0.25, 0.125
    transform = torch.tensor(
        [
            [scale_x, 0.0, translation_x],
            [0.0, scale_y, translation_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    resample = AffineResample(
        transform=transform,
        output_resolution=(6, 16),
        sampling_mode=SAMPLING_MODE_CUBIC,
        fill_rule=FILL_RULE_ZERO,
    )
    result = resample(raster)

    levels = torch.iinfo(raster.dtype)
    assert (
        result.dtype == raster.dtype
    ), f"An integer raster must come back on its own dtype. {result.dtype=}, {raster.dtype=}"
    assert bool(torch.all((result >= levels.min) & (result <= levels.max))), (
        "Every resampled value must land back inside the levels the raster's own "
        "dtype carries. "
        f"{result.min()=}, {result.max()=}, {levels.min=}, {levels.max=}"
    )

    width_only = AffineResample(
        transform=torch.tensor(
            [[scale_x, 0.0, translation_x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        output_resolution=(24, 16),
        sampling_mode=SAMPLING_MODE_CUBIC,
        fill_rule=FILL_RULE_ZERO,
    )
    intermediate = width_only(raster)
    assert intermediate.dtype == raster.dtype, (
        "The first axis must hand the second an integer raster rather than an "
        "accumulator, so the two axes meet on the raster's own levels. "
        f"{intermediate.dtype=}, {raster.dtype=}"
    )
    height_only = AffineResample(
        transform=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, scale_y, translation_y], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        output_resolution=(6, 16),
        sampling_mode=SAMPLING_MODE_CUBIC,
        fill_rule=FILL_RULE_ZERO,
    )
    composed = height_only(intermediate)
    assert torch.equal(result, composed), (
        "Resampling both axes at once must give what resampling them one after the "
        "other gives. "
        f"{result=}, {composed=}"
    )
    return


def test_a_boolean_raster_is_selected_under_the_box_kernel_and_refused_under_the_others() -> (
    None
):
    """A boolean raster carries no weighted sum, so the one kernel whose footprint holds a single source sample places it as a selection.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.rand(size=(8, 8)) > 0.5
    transform = torch.tensor(
        [[0.5, 0.0, -0.25], [0.0, 0.5, -0.25], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    resample = AffineResample(
        transform=transform,
        output_resolution=(4, 4),
        sampling_mode=SAMPLING_MODE_BOX,
        fill_rule=FILL_RULE_ZERO,
    )
    assert (
        raster.dtype == torch.bool
    ), f"The raster this case is about must be the boolean one. {raster.dtype=}"
    result = resample(raster)

    assert result.dtype == torch.bool, (
        "A boolean raster must come back boolean, its gather being a selection "
        "rather than a weighted sum. "
        f"{result.dtype=}, {raster.dtype=}"
    )
    selected = raster[0:8:2, 0:8:2]
    assert torch.equal(
        result, selected
    ), f"Every returned value is the one sample its own footprint holds, {result=}, {selected=}"

    for sampling_mode in (SAMPLING_MODE_TRIANGLE, SAMPLING_MODE_CUBIC):
        refusing = AffineResample(
            transform=transform,
            output_resolution=(4, 4),
            sampling_mode=sampling_mode,
            fill_rule=FILL_RULE_ZERO,
        )
        with pytest.raises(AssertionError, match="boolean raster"):
            refusing(raster)
    return


def test_one_configured_placement_applies_to_every_raster_it_is_handed() -> None:
    """One configured placement is a placement rather than a call, so every raster it is handed comes back placed exactly as that raster placed on its own is.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    rasters = [torch.rand(size=(4, 12), dtype=torch.float64) for _ in range(3)]
    transform = torch.tensor(
        [[0.5, 0.0, 0.25], [0.0, 0.5, 0.25], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    resample = AffineResample(
        transform=transform,
        output_resolution=(2, 6),
        sampling_mode=SAMPLING_MODE_TRIANGLE,
        fill_rule=FILL_RULE_ZERO,
    )
    results = resample(*rasters)
    assert len(results) == len(rasters), (
        "One configured placement must return one result per raster it is handed. "
        f"{len(results)=}, {len(rasters)=}"
    )
    for raster, result in zip(rasters, results, strict=True):
        assert torch.equal(result, resample(raster)), (
            "Each raster in the batch must be placed exactly as that raster placed "
            "on its own is. "
            f"{result=}, {resample(raster)=}"
        )
    return


def test_a_map_carrying_rotation_is_rejected() -> None:
    """A rotation breaks the separability the gather is built on, so a map carrying one is refused at construction rather than resampled along axes it does not have.

    Args:
        None.

    Returns:
        None.
    """
    transform = torch.tensor(
        [[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    with pytest.raises(AssertionError, match="linear part is not diagonal"):
        AffineResample(
            transform=transform,
            output_resolution=(8, 8),
            sampling_mode=SAMPLING_MODE_TRIANGLE,
            fill_rule=FILL_RULE_ZERO,
        )
    return


def test_the_zero_fill_normalizes_over_the_whole_footprint() -> None:
    """The zero fill reads the source as surrounded by zeros, so a widening kernel's weights sum to one over its whole footprint before the outside is zeroed, and an interior constant comes back unchanged rather than scaled by the widening.

    Args:
        None.

    Returns:
        None.
    """
    value = 0.375
    # A reduction, so the cubic's support widens and its unnormalized weights sum
    # to the widening rather than to one.
    scale = 0.25
    raster = torch.full((1, 64, 64), value, dtype=torch.float64)
    result = AffineResample(
        transform=torch.tensor(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        output_resolution=(16, 16),
        sampling_mode=SAMPLING_MODE_CUBIC,
        fill_rule=FILL_RULE_ZERO,
    )(raster)

    # The interior alone: at the source's own edges part of the footprint lies
    # outside, which the zero fill is meant to darken.
    interior = result[:, 4:-4, 4:-4]
    assert torch.allclose(
        input=interior,
        other=torch.full_like(interior, value),
        rtol=0.0,
        atol=1e-12,
    ), (
        "A constant the whole footprint sees must come back at that constant, the "
        "weights summing to one over the footprint rather than to the widening, "
        f"{value=} {float(interior.min())=} {float(interior.max())=}"
    )
    return


def test_a_map_carrying_only_fit_residue_is_accepted() -> None:
    """A float32 matrix inverse leaves round-off where the structural zeros are, and round-off is not a rotation.

    Args:
        None.

    Returns:
        None.
    """
    forward = torch.tensor(
        [[0.7784, 0.0, -133.558], [0.0, 0.7784, -258.483], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    transform = torch.linalg.inv(forward)
    assert (
        float(transform[0, 1]) != 0.0 and float(transform[1, 0]) != 0.0
    ), f"This map must actually carry the shear round-off the test is about, {transform=}"
    assert (
        float(transform[2, 0]) != 0.0
        or float(transform[2, 1]) != 0.0
        or float(transform[2, 2]) != 1.0
    ), f"This map must actually carry the last-row round-off the test is about, {transform=}"
    resample = AffineResample(
        transform=transform,
        output_resolution=(8, 8),
        sampling_mode=SAMPLING_MODE_TRIANGLE,
        fill_rule=FILL_RULE_ZERO,
    )
    assert resample.transform is transform, (
        "A map carrying only its own fit residue must be stored as it was handed "
        "over rather than rewritten. "
        f"{resample.transform=}, {transform=}"
    )

    # The same map with a shear ten times the tolerance, which is a rotation rather
    # than round-off. Ten times, not a thousand: a rejection an order of magnitude
    # above the constant is what a loosened constant would stop rejecting.
    assert _AXIS_ALIGNED_TOLERANCE == 1e-4, (
        "This rejection is read one order of magnitude above the tolerance, so a "
        "changed tolerance is a changed test rather than a silently weaker one. "
        f"{_AXIS_ALIGNED_TOLERANCE=}"
    )
    rotated = transform.clone()
    rotated[0, 1] = 1e-3 * float(transform[0, 0])
    with pytest.raises(AssertionError):
        AffineResample(
            transform=rotated,
            output_resolution=(8, 8),
            sampling_mode=SAMPLING_MODE_TRIANGLE,
            fill_rule=FILL_RULE_ZERO,
        )
    return


def test_an_unknown_sampling_mode_or_fill_rule_is_refused_at_construction() -> None:
    """The kernel and the boundary rule are each named out of a closed set this module declares, so a name outside either set fails construction rather than reaching a dispatch with no branch for it.

    Args:
        None.

    Returns:
        None.
    """
    transform = torch.eye(3, dtype=torch.float64)
    with pytest.raises(AssertionError, match="one of the three modes"):
        AffineResample(
            transform=transform,
            output_resolution=(8, 8),
            sampling_mode="lanczos",
            fill_rule=FILL_RULE_ZERO,
        )
    with pytest.raises(AssertionError, match="one of the two rules"):
        AffineResample(
            transform=transform,
            output_resolution=(8, 8),
            sampling_mode=SAMPLING_MODE_TRIANGLE,
            fill_rule="reflect",
        )
    return


def test_the_gather_meets_the_raster_on_its_own_device() -> None:
    """The placement's own indices and weights are built where the raster sits, so a raster on the accelerator is resampled there rather than crossing devices.

    Args:
        None.

    Returns:
        None.
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device to resample a raster on.")
    torch.manual_seed(0)
    raster = torch.rand(size=(3, 12, 16), dtype=torch.float32)
    transform = torch.tensor(
        [[0.5, 0.0, 0.25], [0.0, 0.5, 0.25], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    resample = AffineResample(
        transform=transform,
        output_resolution=(6, 8),
        sampling_mode=SAMPLING_MODE_TRIANGLE,
        fill_rule=FILL_RULE_ZERO,
    )
    result = resample(raster.to("cuda"))
    assert (
        result.device.type == "cuda"
    ), f"The gather must return on the device the raster came in on. {result.device=}"
    assert torch.allclose(result.cpu(), resample(raster)), (
        "The device must not move the values, so the two placements must agree. "
        f"{(result.cpu() - resample(raster)).abs().max()=}"
    )
    return


def test_the_bases_generator_reaches_the_placement_and_moves_nothing() -> None:
    """The base hands every transform its randomness source, and this placement is fixed by its configured map alone.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.rand(size=(4, 12), dtype=torch.float64)
    transform = torch.tensor(
        [[0.5, 0.0, 0.25], [0.0, 0.5, 0.25], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    resample = AffineResample(
        transform=transform,
        output_resolution=(2, 6),
        sampling_mode=SAMPLING_MODE_TRIANGLE,
        fill_rule=FILL_RULE_ZERO,
    )
    first_generator = torch.Generator()
    first_generator.manual_seed(0)
    second_generator = torch.Generator()
    second_generator.manual_seed(1)
    under_first = resample._call_single(raster, generator=first_generator)
    under_second = resample._call_single(raster, generator=second_generator)
    assert torch.equal(under_first, under_second), (
        "This placement draws no randomness, so two different generators must "
        "leave the same result. "
        f"{(under_first - under_second).abs().max()=}"
    )
    under_none = resample._call_single(raster)
    assert torch.equal(under_first, under_none), (
        "This placement draws no randomness, so no generator at all must leave "
        "the same result again. "
        f"{(under_first - under_none).abs().max()=}"
    )
    return


def test_a_mirrored_axis_widens_by_what_that_axis_reduces_by() -> None:
    """The widening a reducing cubic takes is the magnitude of the reduction, so a mirrored map matches the unmirrored placement of its pre-flipped raster.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    raster = torch.rand(size=(1, 40, 40), dtype=torch.float64)
    output_extent = 16
    for scale in (0.5, 0.78125, 1.0, 2.0):
        forward = AffineResample(
            transform=torch.tensor(
                [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            output_resolution=(output_extent, output_extent),
            sampling_mode=SAMPLING_MODE_CUBIC,
            fill_rule=FILL_RULE_RENORMALIZE,
        )(raster)
        # The same placement with the x axis mirrored, its translation put back so the same content lands in the same window.
        mirrored = AffineResample(
            transform=torch.tensor(
                [
                    [-scale, 0.0, scale * float(raster.shape[2] - 1)],
                    [0.0, scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            output_resolution=(output_extent, output_extent),
            sampling_mode=SAMPLING_MODE_CUBIC,
            fill_rule=FILL_RULE_RENORMALIZE,
        )(torch.flip(raster, dims=[2]))
        assert torch.allclose(forward, mirrored), (
            "A mirrored axis must widen by what that axis reduces by, so the same "
            "content must land the same way under either map. "
            f"{scale=}, {float((forward - mirrored).abs().max())=}"
        )
    return
