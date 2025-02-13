from typing import Tuple, Optional
from abc import ABC, abstractmethod
import torch
from criteria.wrappers import SingleTaskCriterion
from utils.input_checks import check_semantic_segmentation


class SemanticMapBaseCriterion(SingleTaskCriterion, ABC):

    REDUCTION_OPTIONS = ['mean', 'sum']

    def __init__(
        self,
        class_weights: Optional[Tuple[float, ...]] = None,
        reduction: Optional[str] = 'mean',
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        super(SemanticMapBaseCriterion, self).__init__()
        if class_weights is not None:
            assert type(class_weights) == tuple, f"{type(class_weights)=}"
            assert all([type(elem) == float for elem in class_weights])
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            class_weights = class_weights / class_weights.sum()
        self.class_weights = class_weights
        assert reduction in self.REDUCTION_OPTIONS, f"{reduction=}"
        self.reduction = reduction
        self.device = device

    @abstractmethod
    def _compute_semantic_map_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """To be implemented by child classes.
        """
        raise NotImplementedError

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Assumes y_pred is a tensor of logits.
        """
        # sanity checks
        assert hasattr(self, '_compute_semantic_map_loss')
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)
        # convert to probability distributions
        B, C, _, _ = y_pred.shape
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_true = torch.eye(C, dtype=torch.float32, device=y_true.device)[y_true].permute(0, 3, 1, 2)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert torch.all(torch.isclose(torch.sum(y_pred, dim=1, keepdim=True), torch.ones_like(y_pred)))
        assert torch.all(torch.isclose(torch.sum(y_true, dim=1, keepdim=True), torch.ones_like(y_pred)))
        # match resolution
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_true = torch.nn.functional.interpolate(y_true, size=y_pred.shape[-2:], mode='nearest')
        # compute loss
        loss = self._compute_semantic_map_loss(y_pred=y_pred, y_true=y_true)
        assert loss.shape == (B, C), f"{loss.shape=}"
        # weighted sum over classes
        class_weights = self.class_weights.view((B, C)) if self.class_weights else \
            1 / C * torch.ones(size=(C,), dtype=torch.float32, device=loss.device)
        loss = torch.sum(class_weights * loss, dim=1)
        assert loss.shape == (B,), f"{loss.shape=}"
        # reduce along batch dimension
        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
