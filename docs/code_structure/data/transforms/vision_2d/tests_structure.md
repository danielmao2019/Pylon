# data/transforms/vision_2d — tests structure

## Tests implementation structure

`tests/data/transforms/vision_2d/test_affine_resample.py`

```text
test_affine_resample.py
├── import pytest
├── import torch
├── from data.transforms.vision_2d.affine_resample import FILL_RULE_RENORMALIZE, FILL_RULE_ZERO, SAMPLING_MODE_BOX, SAMPLING_MODE_CUBIC, SAMPLING_MODE_TRIANGLE, AffineResample
├── def test_each_sampling_mode_gathers_over_its_own_support
│   ├── # The three modes differ in the kernel alone, so under one identity-scaled map each output pixel reads exactly the source samples its own kernel's support reaches.
│   ├── for each (sampling mode, half-width) over (SAMPLING_MODE_BOX, one half), (SAMPLING_MODE_TRIANGLE, one) and (SAMPLING_MODE_CUBIC, two)
│   │   ├── calls AffineResample
│   │   ├── calls resample.__call__
│   │   ├── impls assert an output pixel changes when a source sample inside its half-width changes
│   │   └── impls assert it remains equal when a source sample outside that half-width changes
│   └── return
├── def test_only_the_cubic_support_widens_with_the_reduction
│   ├── # A reduction under a non-widening kernel reads the few samples nearest each mapped coordinate, while the cubic gathers the whole footprint the reduction covers.
│   ├── for each of SAMPLING_MODE_BOX, SAMPLING_MODE_TRIANGLE and SAMPLING_MODE_CUBIC under a map reducing by a factor of four
│   │   ├── calls AffineResample
│   │   ├── calls resample.__call__
│   │   ├── impls assert the SAMPLING_MODE_BOX and SAMPLING_MODE_TRIANGLE results read only the samples within their own unwidened half-widths  # impls-node-one-step:skip
│   │   └── impls assert the SAMPLING_MODE_CUBIC result reads every source sample within two output pixels' worth of the mapped coordinate
│   └── return
├── def test_the_output_window_is_the_configured_resolution_rather_than_the_maps_own
│   ├── # The map fixes where each source coordinate lands and leaves open how large a window is rasterized, so one map at two configured resolutions gives two different rasters that agree wherever both cover.
│   ├── for each of two output resolutions under one map
│   │   ├── calls AffineResample
│   │   └── calls resample.__call__
│   ├── impls assert each result's height and width are its own configured output_resolution  # impls-node-one-step:skip
│   ├── impls assert the smaller result equals the larger one restricted to the smaller's own window
│   └── return
├── def test_the_zero_fill_rule_reads_zero_past_the_source_extent
│   ├── # Under the zero rule this resample supplies the boundary fill, so the part of a window that reaches past the source raster contributes fabricated zeros.
│   ├── calls AffineResample(fill_rule=FILL_RULE_ZERO)
│   ├── calls resample.__call__
│   ├── impls assert the output is zero wherever its whole footprint falls outside the source extent
│   ├── impls assert the output equals the unclipped resample wherever its whole footprint falls inside
│   ├── impls assert the output falls below the source's own edge value wherever its footprint straddles that edge
│   └── return
├── def test_the_renormalize_fill_rule_averages_the_part_of_the_footprint_it_can_see
│   ├── # Under the renormalize rule the source is read as a picture with a boundary, so a pixel straddling that boundary is the mean of the samples inside it.
│   ├── calls AffineResample(fill_rule=FILL_RULE_RENORMALIZE)
│   ├── calls resample.__call__
│   ├── impls assert a constant source comes back at that same constant wherever an output pixel's own mapped centre falls inside the source extent
│   ├── impls assert the output is zero wherever that mapped centre falls outside the source extent
│   └── return
├── def test_the_two_fill_rules_agree_wherever_every_footprint_stays_inside_the_source
│   ├── # The rules differ only in what the part of a footprint past the source contributes, so a window and support clear of the source's boundary place a raster identically under either.
│   ├── for each of FILL_RULE_ZERO and FILL_RULE_RENORMALIZE under one map whose every footprint stays inside the source
│   │   ├── calls AffineResample
│   │   └── calls resample.__call__
│   ├── impls assert the two results are equal
│   └── return
├── def test_an_integer_raster_lands_back_on_its_own_levels_between_the_two_axes
│   ├── # An eight-bit raster resamples in eight-bit fixed point one axis at a time, so the first axis's result is itself an eight-bit raster and the second axis gathers those levels.
│   ├── calls AffineResample
│   ├── calls resample.__call__
│   ├── impls assert every returned value is one of the raster's own integer levels
│   ├── impls first_pass = the raster gathered along one axis
│   ├── impls relanded = first_pass landed back on the raster's own integer levels
│   ├── impls reference = relanded gathered along the other axis
│   ├── impls assert the result equals reference
│   └── return
├── def test_a_boolean_raster_is_selected_under_the_box_kernel_and_refused_under_the_others
│   ├── # A boolean raster carries selections, so the one kernel whose footprint holds a single source sample places that sample directly.
│   ├── calls AffineResample(sampling_mode=SAMPLING_MODE_BOX)
│   ├── calls resample.__call__
│   ├── impls assert the returned dtype is the raster's own torch.bool
│   ├── impls assert the call consumes the boolean raster directly as torch.bool
│   ├── impls assert every returned value is one the source raster already held
│   ├── for each of SAMPLING_MODE_TRIANGLE and SAMPLING_MODE_CUBIC
│   │   └── impls assert the boolean raster is refused at the call
│   └── return
├── def test_one_configured_placement_applies_to_every_raster_it_is_handed
│   ├── # The placement is the instance's own state, so handing the base several rasters at once resamples each under that one map, window and kernel.
│   ├── calls AffineResample
│   ├── calls resample.__call__
│   ├── impls assert a multi-raster call returns one result per raster
│   ├── impls assert each equals that raster's own single-raster call
│   └── return
├── def test_a_map_carrying_rotation_is_rejected
│   ├── # A rotation breaks this separable per-axis gather, so construction fails when a map's linear part is off diagonal.
│   ├── calls AffineResample
│   ├── impls assert construction fails on a transform whose linear part is off diagonal
│   └── return
├── def test_the_zero_fill_normalizes_over_the_whole_footprint
│   ├── # The zero fill reads the source as surrounded by zeros, so a widening kernel's weights sum to one over its whole footprint before the outside is zeroed, and an interior constant comes back at its source value.
│   ├── calls AffineResample(transform=a map reducing on both axes, sampling_mode=SAMPLING_MODE_CUBIC, fill_rule=FILL_RULE_ZERO)  # so the cubic's support widens and its unnormalized weights sum to the widening
│   ├── calls resample.__call__
│   ├── impls assert the interior comes back at the constant the whole footprint sees  # the source's own edges are where the fill is meant to darken, so they stay out of it
│   └── return
├── def test_a_map_carrying_only_fit_residue_is_accepted
│   ├── # A similarity fitted between two axis-aligned squares carries its zero rotation as residue, so the diagonality the gather needs is read to _AXIS_ALIGNED_TOLERANCE and that residue constructs.
│   ├── calls AffineResample
│   ├── impls assert a map whose off-diagonal entries sit within the tolerance constructs
│   ├── impls assert the tolerance is the value this rejection is read one order of magnitude above  # so the test tracks changes to the constant's threshold
│   ├── impls assert a map whose off-diagonal entries sit past it fails construction
│   └── return
├── def test_an_unknown_sampling_mode_or_fill_rule_is_refused_at_construction
│   ├── # The kernel and boundary rule are each named out of this module's closed sets, so unknown names fail construction at validation.
│   ├── calls AffineResample
│   ├── impls assert construction fails on a sampling_mode outside SAMPLING_MODE_BOX, SAMPLING_MODE_TRIANGLE and SAMPLING_MODE_CUBIC  # impls-node-one-step:skip
│   ├── impls assert construction fails on a fill_rule outside FILL_RULE_ZERO and FILL_RULE_RENORMALIZE                               # impls-node-one-step:skip
│   └── return
├── def test_the_gather_meets_the_raster_on_its_own_device
│   ├── # The gather is built on the raster's own device, so a raster on either device resamples where its weights are created.
│   ├── calls AffineResample
│   ├── calls resample.__call__
│   ├── impls assert the result carries the raster's own device
│   ├── impls assert the two devices' results agree
│   └── return
├── def test_the_bases_generator_reaches_the_placement_and_moves_nothing
│   ├── # The base hands every transform a generator for sampled transforms, and this deterministic placement gives the same resample whichever generator it is handed.
│   ├── calls AffineResample
│   ├── calls resample.__call__
│   ├── impls assert the results under two different generators are equal
│   ├── impls assert the result under generator=None equals them
│   └── return
└── def test_a_mirrored_axis_widens_by_what_that_axis_reduces_by
    ├── # The widening a reducing cubic takes is the magnitude of the reduction, so a map that mirrors an axis places the same content the same way the unmirrored map places its pre-flipped raster.
    ├── for each of a reduction, a magnification and the identity
    │   ├── calls AffineResample(transform=that scale on both axes, sampling_mode=SAMPLING_MODE_CUBIC, fill_rule=FILL_RULE_RENORMALIZE)
    │   ├── calls resample.__call__
    │   ├── calls AffineResample(transform=that same scale with the x axis mirrored, its translation put back so the same content lands in the same window, sampling_mode=SAMPLING_MODE_CUBIC, fill_rule=FILL_RULE_RENORMALIZE)  # impls-node-one-step:skip
    │   ├── calls resample.__call__  # the raster pre-flipped along that axis
    │   └── impls assert the mirrored placement equals the unmirrored one  # proves widening is read from scale magnitude
    └── return
```
