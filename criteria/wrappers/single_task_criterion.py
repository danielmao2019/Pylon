from typing import Dict, Optional, Tuple, Union

import torch

from criteria.base_criterion import BaseCriterion
from utils.input_checks import check_write_file


class SingleTaskCriterion(BaseCriterion):

    def __call__(
        self,
        y_pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        r"""This method assumes `_compute_loss(...)` is implemented.

        Args:
            y_pred: Single-task model outputs. Tensor inputs are forwarded
                directly. One-entry tensor dicts are unwrapped for backward
                compatibility. Structured values are otherwise passed through to
                `_compute_loss(...)` unchanged.
            y_true: Single-task supervision inputs. Tensor inputs are forwarded
                directly. One-entry tensor dicts are unwrapped for backward
                compatibility. Structured values are otherwise passed through to
                `_compute_loss(...)` unchanged.

        Returns:
            Scalar loss tensor returned by `_compute_loss(...)`.
        """
        assert hasattr(self, "_compute_loss") and callable(self._compute_loss), (
            "Expected subclasses of `SingleTaskCriterion` to implement "
            "`_compute_loss(...)`. "
            f"{type(self)=}"
        )

        def _validate_inputs() -> None:
            assert isinstance(y_pred, (torch.Tensor, dict)), (
                "Expected `y_pred` to be a tensor or a dict. " f"{type(y_pred)=}"
            )
            if isinstance(y_pred, dict):
                assert len(y_pred) > 0, (
                    "Expected `y_pred` dict inputs to be non-empty. "
                    f"{y_pred.keys()=}"
                )
                assert all(isinstance(key, str) for key in y_pred.keys()), (
                    "Expected every `y_pred` dict key to be a string. "
                    f"{y_pred.keys()=}"
                )
                assert all(
                    isinstance(value, torch.Tensor) for value in y_pred.values()
                ), (
                    "Expected every `y_pred` dict value to be a tensor. "
                    f"{[(key, type(value)) for key, value in y_pred.items()]=}"
                )

            assert isinstance(y_true, (torch.Tensor, dict)), (
                "Expected `y_true` to be a tensor or a dict. " f"{type(y_true)=}"
            )
            if isinstance(y_true, dict):
                assert len(y_true) > 0, (
                    "Expected `y_true` dict inputs to be non-empty. "
                    f"{y_true.keys()=}"
                )
                assert all(isinstance(key, str) for key in y_true.keys()), (
                    "Expected every `y_true` dict key to be a string. "
                    f"{y_true.keys()=}"
                )
                assert all(
                    isinstance(value, torch.Tensor) for value in y_true.values()
                ), (
                    "Expected every `y_true` dict value to be a tensor. "
                    f"{[(key, type(value)) for key, value in y_true.items()]=}"
                )

        _validate_inputs()

        def _normalize_inputs() -> Tuple[
            Union[torch.Tensor, Dict[str, torch.Tensor]],
            Union[torch.Tensor, Dict[str, torch.Tensor]],
        ]:
            if isinstance(y_pred, dict) and len(y_pred) == 1:
                y_pred_normalized = next(iter(y_pred.values()))
            else:
                y_pred_normalized = y_pred

            if isinstance(y_true, dict) and len(y_true) == 1:
                y_true_normalized = next(iter(y_true.values()))
            else:
                y_true_normalized = y_true

            return y_pred_normalized, y_true_normalized

        y_pred, y_true = _normalize_inputs()

        loss = self._compute_loss(y_pred=y_pred, y_true=y_true)
        assert isinstance(loss, torch.Tensor), (
            "Expected `_compute_loss(...)` to return a tensor. " f"{type(loss)=}"
        )
        assert loss.ndim == 0 and loss.numel() == 1, (
            "Expected `_compute_loss(...)` to return a scalar tensor. " f"{loss.shape=}"
        )
        self.add_to_buffer(loss)
        return loss

    def summarize(self, output_path: Optional[str] = None) -> torch.Tensor:
        r"""This method stacks loss trajectory across all data points in buffer.
        Thread-safe version that works with async operations.
        """
        assert self.use_buffer and hasattr(self, 'buffer') and self.buffer is not None
        self._buffer_queue.join()  # Wait for all items to be processed
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0

        # summarize losses
        result = torch.stack(self.buffer, dim=0)
        assert result.ndim == 1, f"{result.shape=}"

        # save to disk if path provided
        if output_path is not None:
            check_write_file(path=output_path)
            torch.save(obj=result, f=output_path)

        return result
