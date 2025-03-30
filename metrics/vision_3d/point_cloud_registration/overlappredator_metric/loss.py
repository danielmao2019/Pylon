import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lib.utils import square_distance
from sklearn.metrics import precision_recall_fscore_support


class OverlapPredatorMetric(nn.Module):

    def __init__(self, configs):
        super(OverlapPredatorMetric,self).__init__()

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        self.safe_radius = configs.safe_radius 
        self.matchability_radius = configs.matchability_radius
        self.pos_radius = configs.pos_radius # just to take care of the numeric precision

    def get_recall(self,coords_dist,feats_dist):
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

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, Any]) -> torch.Tensor:
        # Input checks
        assert isinstance(y_pred, dict), f"{type(y_pred)=}"
        assert y_pred.keys() == {'scores_overlap', 'scores_saliency'}, f"{y_pred.keys()=}"
        assert isinstance(y_true, dict), f"{type(y_true)=}"
        assert y_true.keys() == {'src_pc', 'tgt_pc', 'correspondence', 'rot', 'trans'}, f"{y_true.keys()=}"
        src_pc = y_true['src_pc']
        tgt_pc = y_true['tgt_pc']
        assert isinstance(src_pc, dict), f"{type(src_pc)=}"
        assert src_pc.keys() == {'pos', 'feat'}, f"{src_pc.keys()=}"
        assert isinstance(tgt_pc, dict), f"{type(tgt_pc)=}"
        assert tgt_pc.keys() == {'pos', 'feat'}, f"{tgt_pc.keys()=}"

        src_pcd = src_pcd['pos']
        tgt_pcd = tgt_pcd['pos']
        src_feats = src_pcd['feat']
        tgt_feats = tgt_pcd['feat']
        correspondence = y_true['correspondence']
        rot = y_true['rot']
        trans = y_true['trans']
        scores_overlap = y_pred['scores_overlap']
        scores_saliency = y_pred['scores_saliency']

        src_pcd = (torch.matmul(rot,src_pcd.transpose(0,1))+trans).transpose(0,1)
        stats = dict()

        src_idx = list(set(correspondence[:,0].int().tolist()))
        tgt_idx = list(set(correspondence[:,1].int().tolist()))

        #######################
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx]=1.
        tgt_gt = torch.zeros(tgt_pcd.size(0))
        tgt_gt[tgt_idx]=1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))

        cls_precision, cls_recall = self.get_cls_precision_recall(scores_overlap, gt_labels)
        stats['overlap_recall'] = cls_recall
        stats['overlap_precision'] = cls_precision

        #######################
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

        cls_precision, cls_recall = self.get_cls_precision_recall(scores_saliency, gt_labels)
        stats['saliency_recall'] = cls_recall
        stats['saliency_precision'] = cls_precision

        #######################################
        # filter some of correspondence as we are using different radius for "overlap" and "correspondence"
        c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        c_select = c_dist < self.pos_radius - 0.001
        correspondence = correspondence[c_select]
        if(correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]
        src_idx = correspondence[:,0]
        tgt_idx = correspondence[:,1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]

        #######################
        # get L2 distance between source / target point cloud
        coords_dist = torch.sqrt(square_distance(src_pcd[None,:,:], tgt_pcd[None,:,:]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats[None,:,:], tgt_feats[None,:,:],normalised=True)).squeeze(0)

        #######################
        recall = self.get_recall(coords_dist, feats_dist)
        stats['recall']=recall

        return stats
