from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def circle_loss(
    pos_masks: torch.Tensor,
    neg_masks: torch.Tensor,
    feat_dists: torch.Tensor,
    pos_margin: float,
    neg_margin: float,
    pos_optimal: float,
    neg_optimal: float,
    log_scale: float,
) -> torch.Tensor:
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights).detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights).detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


def weighted_circle_loss(
    pos_masks: torch.Tensor,
    neg_masks: torch.Tensor,
    feat_dists: torch.Tensor,
    pos_margin: float,
    neg_margin: float,
    pos_optimal: float,
    neg_optimal: float,
    log_scale: float,
    pos_scales: Optional[torch.Tensor] = None,
    neg_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights)
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


class CircleLoss(nn.Module):
    def __init__(self, pos_margin: float, neg_margin: float, pos_optimal: float, neg_optimal: float, log_scale: float) -> None:
        super(CircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks: torch.Tensor, neg_masks: torch.Tensor, feat_dists: torch.Tensor) -> torch.Tensor:
        return circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
        )


class WeightedCircleLoss(nn.Module):
    def __init__(self, pos_margin: float, neg_margin: float, pos_optimal: float, neg_optimal: float, log_scale: float) -> None:
        super(WeightedCircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks: torch.Tensor, neg_masks: torch.Tensor, feat_dists: torch.Tensor, pos_scales: Optional[torch.Tensor] = None, neg_scales: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
            pos_scales=pos_scales,
            neg_scales=neg_scales,
        )
