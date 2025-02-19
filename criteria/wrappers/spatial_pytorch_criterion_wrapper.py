import torch
from criteria.wrappers import SingleTaskCriterion
from utils.builders import build_from_config
from utils.input_checks import check_semantic_segmentation


class SpatialPyTorchCriterionWrapper(SingleTaskCriterion):

    def __init__(self, criterion_cfg) -> None:
        super(SpatialPyTorchCriterionWrapper, self).__init__()
        self.criterion = build_from_config(criterion_cfg)

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert isinstance(y_pred, torch.Tensor)
        assert isinstance(y_true, torch.Tensor)
        assert y_pred.ndim >= 2
        assert y_true.ndim >= 2
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_true = torch.nn.functional.interpolate(y_true, size=y_pred.shape[-2:], mode='nearest')
        return self.criterion(input=y_pred, target=y_true)
