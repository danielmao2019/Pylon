from typing import Dict
import torch
import torch.nn as nn
from easydict import EasyDict
from criteria.vision_3d.point_cloud_registration.geotransformer_criterion.circle_loss import WeightedCircleLoss
from models.point_cloud_registration.geotransformer.transformation import apply_transform
from models.point_cloud_registration.geotransformer.pairwise_distance import pairwise_distance
from criteria.wrappers.single_task_criterion import SingleTaskCriterion


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, y_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref_feats = y_pred['ref_feats_c']
        src_feats = y_pred['src_feats_c']
        gt_node_corr_indices = y_pred['gt_node_corr_indices']
        gt_node_corr_overlaps = y_pred['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True)).float()

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref_node_corr_knn_points = y_pred['ref_node_corr_knn_points']
        src_node_corr_knn_points = y_pred['src_node_corr_knn_points']
        ref_node_corr_knn_masks = y_pred['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = y_pred['src_node_corr_knn_masks']
        matching_scores = y_pred['matching_scores']
        transform = y_true['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class GeoTransformerCriterion(SingleTaskCriterion):

    def __init__(self, **cfg) -> None:
        cfg = EasyDict(cfg)
        super(GeoTransformerCriterion, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        coarse_loss = self.coarse_loss(y_pred)
        fine_loss = self.fine_loss(y_pred, y_true)
        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss
        self.add_to_buffer(loss)
        return loss
