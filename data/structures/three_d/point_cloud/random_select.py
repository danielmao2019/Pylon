from typing import Any, Optional

import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select
from utils.determinism.hash_utils import convert_to_seed


class RandomSelect:
    """Draws a random subset of a point cloud's points, sized either as a fraction of the cloud or as a fixed count."""

    def __init__(
        self, percentage: Optional[float] = None, count: Optional[int] = None
    ) -> None:
        """Holds exactly one of the two sizing modes and leaves the other empty.

        Args:
            percentage: The fraction of the cloud's points to draw, as a float in (0, 1], or None when the count mode is used.
            count: The number of points to draw, as a positive int, or None when the percentage mode is used.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            assert (percentage is not None) ^ (
                count is not None
            ), f"exactly one of the two sizing modes is given: percentage={percentage}, count={count}"
            if percentage is not None:
                assert isinstance(
                    percentage, (int, float)
                ), f"a percentage is an int or a float: type(percentage)={type(percentage)}"
                assert (
                    0 < percentage <= 1
                ), f"a percentage lies in (0, 1]: percentage={percentage}"
            else:
                assert isinstance(
                    count, int
                ), f"a count is an int: type(count)={type(count)}"
                assert count > 0, f"a count is positive: count={count}"

        _validate_inputs()

        if percentage is not None:
            self.percentage = float(percentage)
            self.count = None
        else:
            self.count = count
            self.percentage = None

    def __call__(
        self,
        pc: PointCloud,
        seed: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
    ) -> PointCloud:
        """Takes the sized random subset of pc, through a Select over the head of a random permutation of its point indices.

        Args:
            pc: The point cloud to draw from, whose fields are torch tensors of shape [N] or [N, C] on one device.
            seed: The seed the drawing is made deterministic by, as an int or as any hashable value convert_to_seed turns into one, or None when a generator is given.
            generator: The torch generator the drawing is made from, sitting on a device of the same type as pc's, or None when a seed is given.

        Returns:
            The selected point cloud, carrying every field of pc indexed down to the drawn points plus an 'indices' field naming which points of pc they were.
        """

        def _validate_inputs() -> None:
            assert isinstance(
                pc, PointCloud
            ), f"a random selection draws from a point cloud: type(pc)={type(pc)}"
            # the two randomness sources are one arg apiece, so the pair is checked once the second of them is reached
            assert (seed is not None) ^ (
                generator is not None
            ), f"exactly one of the two randomness sources is given: seed={seed}, generator={generator}"

        _validate_inputs()

        device = pc.device
        num_points = pc.num_points

        if generator is not None:
            assert (
                generator.device.type == device.type
            ), f"a generator draws on the device the point cloud sits on: generator.device.type={generator.device.type}, device.type={device.type}"
            gen = generator
        else:
            gen = torch.Generator(device=device)
            if not isinstance(seed, int):
                seed = convert_to_seed(seed)
            gen.manual_seed(seed)

        if self.percentage is not None:
            num_points_to_select = int(num_points * self.percentage)
        else:
            num_points_to_select = min(self.count, num_points)

        indices = torch.randperm(num_points, generator=gen, device=device)[
            :num_points_to_select
        ]
        # constructing the Select is not applying it, and the applied result is what this returns
        selected = Select(indices=indices)(pc)
        return selected

    def __str__(self) -> str:
        """Renders the selection under whichever of the two sizing modes it carries.

        Args:
            None.

        Returns:
            The rendering, as a str naming the sizing mode this selection carries.
        """
        if self.percentage is not None:
            return f"RandomSelect(percentage={self.percentage})"
        return f"RandomSelect(count={self.count})"
