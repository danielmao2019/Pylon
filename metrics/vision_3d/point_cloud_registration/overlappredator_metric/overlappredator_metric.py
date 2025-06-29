from typing import Dict, Any
from easydict import EasyDict
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from models.point_cloud_registration.overlappredator.utils import square_distance
from metrics.wrappers.single_task_metric import SingleTaskMetric


class OverlapPredatorMetric(SingleTaskMetric):

    def __init__(self, **configs):
        configs = EasyDict(configs)
        super(OverlapPredatorMetric, self).__init__()
        self.max_points = configs.max_points
        self.matchability_radius = configs.matchability_radius
        self.pos_radius = configs.pos_radius # just to take care of the numeric precision

    def get_recall(self, coords_dist, feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1)>0).float().sum()+1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist,dim=-1,index=sel_idx[:,None])[pos_mask.sum(-1)>0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall

    def get_cls_precision_recall(self, prediction, gt):
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(),predicted_labels, average='binary')
        return cls_precision, cls_recall

    def _compute_overlap_metrics(self, scores_overlap: torch.Tensor, src_idx: list, tgt_idx: list, src_pcd: torch.Tensor, tgt_pcd: torch.Tensor) -> Dict[str, float]:
        """Compute overlap precision and recall metrics."""
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        tgt_gt = torch.zeros(tgt_pcd.size(0))
        tgt_gt[tgt_idx] = 1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))

        cls_precision, cls_recall = self.get_cls_precision_recall(scores_overlap, gt_labels)
        return {
            'overlap_recall': cls_recall,
            'overlap_precision': cls_precision
        }

    def _compute_saliency_metrics(self, scores_saliency: torch.Tensor, src_idx: list, tgt_idx: list,
                                src_pcd: torch.Tensor, tgt_pcd: torch.Tensor, src_feats: torch.Tensor,
                                tgt_feats: torch.Tensor) -> Dict[str, float]:
        """Compute saliency precision and recall metrics."""
        src_feats_sel, src_pcd_sel = src_feats[src_idx], src_pcd[src_idx]
        tgt_feats_sel, tgt_pcd_sel = tgt_feats[tgt_idx], tgt_pcd[tgt_idx]
        scores = torch.matmul(src_feats_sel, tgt_feats_sel.transpose(0,1))

        _, idx = scores.max(1)
        distance_1 = torch.norm(src_pcd_sel - tgt_pcd_sel[idx], p=2, dim=1)
        _, idx = scores.max(0)
        distance_2 = torch.norm(tgt_pcd_sel - src_pcd_sel[idx], p=2, dim=1)

        gt_labels = torch.cat(((distance_1<self.matchability_radius).float(),
                             (distance_2<self.matchability_radius).float()))

        src_saliency_scores = scores_saliency[:src_pcd.size(0)][src_idx]
        tgt_saliency_scores = scores_saliency[src_pcd.size(0):][tgt_idx]
        scores_saliency = torch.cat((src_saliency_scores, tgt_saliency_scores))

        cls_precision, cls_recall = self.get_cls_precision_recall(scores_saliency, gt_labels)
        return {
            'saliency_recall': cls_recall,
            'saliency_precision': cls_precision
        }

    def _compute_feature_recall(self, correspondence: torch.Tensor, src_pcd: torch.Tensor,
                              tgt_pcd: torch.Tensor, src_feats: torch.Tensor,
                              tgt_feats: torch.Tensor) -> float:
        """Compute feature matching recall."""
        # Filter correspondence based on radius
        c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim=1)
        c_select = c_dist < self.pos_radius - 0.001
        correspondence = correspondence[c_select]

        if (correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]

        src_idx = correspondence[:,0]
        tgt_idx = correspondence[:,1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]

        coords_dist = torch.sqrt(square_distance(src_pcd[None,:,:], tgt_pcd[None,:,:]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats[None,:,:], tgt_feats[None,:,:], normalised=True)).squeeze(0)

        return self.get_recall(coords_dist, feats_dist)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, Any], idx: int) -> Dict[str, torch.Tensor]:
        # Input checks
        assert isinstance(y_pred, dict), f"{type(y_pred)=}"
        assert y_pred.keys() == {'feats_f', 'scores_overlap', 'scores_saliency'}, f"{y_pred.keys()=}"
        assert isinstance(y_true, dict), f"{type(y_true)=}"
        assert y_true.keys() == {'src_pc', 'tgt_pc', 'correspondence', 'rot', 'trans'}, f"{y_true.keys()=}"

        src_pcd = y_true['src_pc']
        tgt_pcd = y_true['tgt_pc']
        assert isinstance(src_pcd, torch.Tensor), f"{type(src_pcd)=}"
        assert isinstance(tgt_pcd, torch.Tensor), f"{type(tgt_pcd)=}"
        assert len(y_pred['feats_f']) == len(src_pcd) + len(tgt_pcd), \
            f"{y_pred['feats_f'].shape=}, {src_pcd.shape=}, {tgt_pcd.shape=}"
        src_feats = y_pred['feats_f'][:len(src_pcd)]
        tgt_feats = y_pred['feats_f'][len(src_pcd):]
        correspondence = y_true['correspondence']
        rot = y_true['rot']
        trans = y_true['trans']
        scores_overlap = y_pred['scores_overlap']
        scores_saliency = y_pred['scores_saliency']

        src_pcd = (torch.matmul(rot, src_pcd.transpose(0,1)) + trans).transpose(0,1)

        src_idx = list(set(correspondence[:,0].int().tolist()))
        tgt_idx = list(set(correspondence[:,1].int().tolist()))

        # Compute all metrics
        scores = {}
        scores.update(self._compute_overlap_metrics(
            scores_overlap, src_idx, tgt_idx, src_pcd, tgt_pcd,
        ))
        scores.update(self._compute_saliency_metrics(
            scores_saliency, src_idx, tgt_idx, src_pcd, tgt_pcd, src_feats, tgt_feats,
        ))
        scores['recall'] = self._compute_feature_recall(
            correspondence, src_pcd, tgt_pcd, src_feats, tgt_feats,
        )

        self.add_to_buffer(scores, idx)
        return scores
