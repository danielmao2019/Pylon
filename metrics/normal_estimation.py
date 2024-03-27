import torch


class NormalEstimationMetric:

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): the (unnormalized) output of the model.
            y_true (torch.Tensor): the ground truth label for the normal vector.
        Returns:
            score (torch.Tensor): a single-element tensor representing the score for normal estimation.
        """
        # sanity checks
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert len(y_pred.shape) == len(y_true.shape) == 4, f"{y_pred.shape=}, {y_true.shape=}"
        assert torch.all((torch.norm(y_true, p=2, dim=1) - 1).abs() < 1e-03), f"{(torch.norm(y_true, p=2, dim=1) - 1).abs().max()=}"
        # compute score
        cosine_map = torch.nn.CosineSimilarity(dim=1)(y_pred, y_true)
        binary_mask = torch.sum(y_true, dim=1) != 0
        cosine_map = cosine_map.masked_select(binary_mask)
        score = torch.rad2deg(torch.acos(cosine_map)).mean()
        assert score.numel() == 1, f"{score.numel()=}"
        return score
