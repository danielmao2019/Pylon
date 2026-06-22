"""2D axis-aligned-box display response APIs."""

from typing import List, Optional

from data.viewer.utils.displays.aabbs.twod.ts.backend.schemas.display_response import (
    Aabb2dDisplayResponse,
)


def create_aabb_2d_display_response(
    slot_id: str,
    title: str,
    aabbs: List[List[float]],
    scores: Optional[List[float]] = None,
) -> Aabb2dDisplayResponse:
    """Create a 2D axis-aligned-box overlay display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        aabbs: Inline axis-aligned 2D boxes, each a 4-float list
            ``[min_x, min_y, max_x, max_y]`` in image-pixel coordinates.
        scores: Optional per-box scores, one per box in ``aabbs``.

    Returns:
        2D axis-aligned-box overlay display response.
    """

    def _validate_inputs() -> None:
        assert isinstance(slot_id, str), (
            "Slot id must be a string. slot_id=%r" % slot_id
        )
        assert isinstance(title, str), "Title must be a string. title=%r" % title
        assert isinstance(aabbs, list), "Boxes must be a list. aabbs=%r" % aabbs
        for box in aabbs:
            assert isinstance(box, list), "Each box must be a list. box=%r" % box
            assert len(box) == 4, (
                "Each 2D box must have 4 floats [min_x, min_y, max_x, max_y]. box=%r"
                % box
            )
            for coordinate in box:
                assert isinstance(coordinate, float), (
                    "Each box coordinate must be a float. coordinate=%r" % coordinate
                )
        assert scores is None or isinstance(scores, list), (
            "Scores must be a list or None. scores=%r" % scores
        )
        if scores is not None:
            assert len(scores) == len(
                aabbs
            ), "Scores must have one entry per box. len(scores)=%d len(aabbs)=%d" % (
                len(scores),
                len(aabbs),
            )
            for score in scores:
                assert isinstance(score, float), (
                    "Each score must be a float. score=%r" % score
                )

    _validate_inputs()

    return Aabb2dDisplayResponse(
        slot_id=slot_id,
        title=title,
        aabbs=aabbs,
        scores=scores,
    )
