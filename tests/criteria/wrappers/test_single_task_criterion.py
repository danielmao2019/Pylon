from typing import Any, Dict, List

import pytest
import torch

from criteria.wrappers.single_task_criterion import SingleTaskCriterion


class StructuredPassthroughCriterion(SingleTaskCriterion):
    """Criterion that verifies structured dict values reach `_compute_loss`.

    Args:
        None.

    Returns:
        None.
    """

    expected_pred_metadata: List[str]
    expected_true_metadata: List[str]

    def __init__(
        self,
        expected_pred_metadata: List[str],
        expected_true_metadata: List[str],
    ) -> None:
        """Store the structured values expected at loss-computation time.

        Args:
            expected_pred_metadata: List object expected under `y_pred`.
            expected_true_metadata: List object expected under `y_true`.

        Returns:
            None.
        """
        super().__init__()
        self.expected_pred_metadata = expected_pred_metadata
        self.expected_true_metadata = expected_true_metadata

    def _compute_loss(
        self,
        y_pred: Dict[str, Any],
        y_true: Dict[str, Any],
    ) -> torch.Tensor:
        """Check structured values and return a scalar tensor loss.

        Args:
            y_pred: Prediction dict containing tensor `loss` and structured `metadata`.
            y_true: Supervision dict containing tensor `target` and structured `metadata`.

        Returns:
            Scalar tensor equal to `y_pred['loss'] + y_true['target']`.
        """
        assert y_pred["metadata"] is self.expected_pred_metadata, (
            "Expected the prediction metadata list to reach `_compute_loss` "
            "unchanged. "
            f"{id(y_pred['metadata'])=} {id(self.expected_pred_metadata)=}"
        )
        assert y_true["metadata"] is self.expected_true_metadata, (
            "Expected the supervision metadata list to reach `_compute_loss` "
            "unchanged. "
            f"{id(y_true['metadata'])=} {id(self.expected_true_metadata)=}"
        )
        loss = y_pred["loss"] + y_true["target"]
        return loss


def test_structured_dict_values_pass_through_to_the_subclass() -> None:
    """Structured dict values are not rejected by the generic wrapper.

    Args:
        None.

    Returns:
        None.
    """
    pred_metadata = ["model-side helper"]
    true_metadata = ["dataset-side helper"]
    criterion = StructuredPassthroughCriterion(
        expected_pred_metadata=pred_metadata,
        expected_true_metadata=true_metadata,
    )

    loss = criterion(
        y_pred={
            "loss": torch.tensor(2.0),
            "metadata": pred_metadata,
        },
        y_true={
            "target": torch.tensor(3.0),
            "metadata": true_metadata,
        },
    )

    assert torch.equal(loss, torch.tensor(5.0)), (
        "Expected the subclass-computed scalar loss to be returned. " f"{loss=}"
    )


def test_dict_inputs_must_carry_at_least_one_tensor_value() -> None:
    """A structured-only dict is rejected before subclass loss computation.

    Args:
        None.

    Returns:
        None.
    """
    criterion = StructuredPassthroughCriterion(
        expected_pred_metadata=["model-side helper"],
        expected_true_metadata=["dataset-side helper"],
    )

    with pytest.raises(AssertionError, match="at least one tensor prediction"):
        criterion(
            y_pred={"metadata": ["model-side helper"]},
            y_true={
                "target": torch.tensor(3.0),
                "metadata": ["dataset-side helper"],
            },
        )

    with pytest.raises(AssertionError, match="at least one tensor supervision"):
        criterion(
            y_pred={
                "loss": torch.tensor(2.0),
                "metadata": ["model-side helper"],
            },
            y_true={"metadata": ["dataset-side helper"]},
        )
