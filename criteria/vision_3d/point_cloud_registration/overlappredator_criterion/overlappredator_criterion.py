from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.point_cloud_registration.overlappredator.utils import square_distance
from criteria.wrappers.single_task_criterion import SingleTaskCriterion


class OverlapPredatorCriterion(SingleTaskCriterion):
    """
    We evaluate both contrastive loss and circle loss
    """
    def __init__(self, **configs) -> None:
        from easydict import EasyDict
        configs = EasyDict(configs)

        super(OverlapPredatorCriterion, self).__init__()
        self.log_scale = configs.log_scale
        self.pos_optimal = configs.pos_optimal
        self.neg_optimal = configs.neg_optimal

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        self.safe_radius = configs.safe_radius
        self.matchability_radius = configs.matchability_radius
        self.pos_radius = configs.pos_radius # just to take care of the numeric precision

        self.w_circle_loss = configs.w_circle_loss
        self.w_overlap_loss = configs.w_overlap_loss
        self.w_saliency_loss = configs.w_saliency_loss

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach()

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()

        return w_class_loss

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, Any]) -> torch.Tensor:
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """
        # Input checks
        assert isinstance(y_pred, dict), f"{type(y_pred)=}"
        assert y_pred.keys() == {'feats_f', 'scores_overlap', 'scores_saliency'}, f"{y_pred.keys()=}"
        assert isinstance(y_true, dict), f"{type(y_true)=}"
        assert y_true.keys() == {'src_pc', 'tgt_pc', 'correspondences', 'rot', 'trans'}, f"{y_true.keys()=}"
        src_pcd = y_true['src_pc']
        tgt_pcd = y_true['tgt_pc']
        assert isinstance(src_pcd, torch.Tensor), f"{type(src_pcd)=}"
        assert isinstance(tgt_pcd, torch.Tensor), f"{type(tgt_pcd)=}"

        assert len(y_pred['feats_f']) == len(src_pcd) + len(tgt_pcd), \
            f"{y_pred['feats_f'].shape=}, {src_pcd.shape=}, {tgt_pcd.shape=}"
        src_feats = y_pred['feats_f'][:len(src_pcd)]
        tgt_feats = y_pred['feats_f'][len(src_pcd):]
        correspondences = y_true['correspondences']
        rot = y_true['rot']
        trans = y_true['trans']
        scores_overlap = y_pred['scores_overlap']
        scores_saliency = y_pred['scores_saliency']

        src_pcd = (torch.matmul(rot,src_pcd.transpose(0,1))+trans.reshape(-1, 1)).transpose(0,1)
        stats = dict()

        src_idx = list(set(correspondences[:,0].int().tolist()))
        tgt_idx = list(set(correspondences[:,1].int().tolist()))

        #######################
        # get BCE loss for overlap, here the ground truth label is obtained from correspondence information
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx]=1.
        tgt_gt = torch.zeros(tgt_pcd.size(0))
        tgt_gt[tgt_idx]=1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))

        class_loss = self.get_weighted_bce_loss(scores_overlap, gt_labels)
        stats['overlap_loss'] = class_loss

        #######################
        # get BCE loss for saliency part, here we only supervise points in the overlap region
        src_feats_sel, src_pcd_sel = src_feats[src_idx], src_pcd[src_idx]
        tgt_feats_sel, tgt_pcd_sel = tgt_feats[tgt_idx], tgt_pcd[tgt_idx]
        scores = torch.matmul(src_feats_sel, tgt_feats_sel.transpose(0,1))
        _, idx = scores.max(1)
        distance_1 = torch.norm(src_pcd_sel - tgt_pcd_sel[idx], p=2, dim=1)
        _, idx = scores.max(0)
        distance_2 = torch.norm(tgt_pcd_sel - src_pcd_sel[idx], p=2, dim=1)

        gt_labels = torch.cat(((distance_1<self.matchability_radius).float(), (distance_2<self.matchability_radius).float()))

        src_saliency_scores = scores_saliency[:src_pcd.size(0)][src_idx]
        tgt_saliency_scores = scores_saliency[src_pcd.size(0):][tgt_idx]
        scores_saliency = torch.cat((src_saliency_scores, tgt_saliency_scores))

        class_loss = self.get_weighted_bce_loss(scores_saliency, gt_labels)
        stats['saliency_loss'] = class_loss

        #######################################
        # filter some of correspondence as we are using different radius for "overlap" and "correspondence"
        c_dist = torch.norm(src_pcd[correspondences[:,0]] - tgt_pcd[correspondences[:,1]], dim = 1)
        c_select = c_dist < self.pos_radius - 0.001
        correspondences = correspondences[c_select]
        if(correspondences.size(0) > self.max_points):
            choice = np.random.permutation(correspondences.size(0))[:self.max_points]
            correspondences = correspondences[choice]
        src_idx = correspondences[:,0]
        tgt_idx = correspondences[:,1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]

        #######################
        # get L2 distance between source / target point cloud
        coords_dist = torch.sqrt(square_distance(src_pcd[None,:,:], tgt_pcd[None,:,:]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats[None,:,:], tgt_feats[None,:,:],normalised=True)).squeeze(0)

        ##############################
        # get FMR and circle loss
        ##############################
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)
        stats['circle_loss']= circle_loss

        assert isinstance(stats, dict), f"{type(stats)=}"
        assert stats.keys() == {'overlap_loss', 'saliency_loss', 'circle_loss'}, f"{stats.keys()=}"
        total_loss = (
            self.w_circle_loss * stats['circle_loss'] +
            self.w_overlap_loss * stats['overlap_loss'] +
            self.w_saliency_loss * stats['saliency_loss']
        )
        self.add_to_buffer(total_loss)
        return total_loss
