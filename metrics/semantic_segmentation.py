import torch


class SemanticSegmentationMetric:

    def __init__(self, num_classes: int, ignore_index: int):
        self.num_classes = num_classes
        self.ignore_index= ignore_index

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Return:
            score (torch.Tensor): 1D tensor of length num_classes representing the IoU scores for each class.
        """
        # input checks
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
        assert y_pred.shape[-2:] == y_true.shape[-2:], f"{y_pred.shape=}, {y_true.shape=}"
        # make prediction from output
        y_pred = torch.argmax(torch.nn.functional.softmax(y_pred, dim=1), dim=1)
        y_pred = y_pred.type(torch.int64)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute score
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        count = torch.bincount(
            y_true[mask] * self.num_classes + y_pred[mask], minlength=self.num_classes**2,
        ).reshape((self.num_classes,) * 2)
        numerator = count.diag()
        denominator = count.sum(dim=0, keepdim=False) + count.sum(dim=1, keepdim=False) - count.diag()
        score = numerator / denominator
        assert score.shape == (self.num_classes,), f"{score.shape=}, {self.num_classes=}"
        return score
