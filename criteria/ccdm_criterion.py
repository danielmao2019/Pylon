"""Reference: https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation.
"""
from typing import Dict
import torch
from .base_criterion import BaseCriterion
from utils.semantic_segmentation import to_one_hot_encoding


class CCDMCriterion(BaseCriterion):

    def __init__(self, num_classes: int, ignore_index: int, num_steps: int):
        super(CCDMCriterion, self).__init__()
        assert type(num_classes) == int
        self.num_classes = num_classes
        assert type(ignore_index) == int, f"{type(ignore_index)=}"
        self.ignore_index = ignore_index
        assert type(num_steps) == int, f"{type(num_steps)=}"
        self.num_steps = num_steps
        from datasets.diffusers import BaseDiffuser
        BaseDiffuser._init_noise_schedule_(self)

    def theta_post(self, diffused_mask: torch.Tensor, original_mask: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # input checks
        assert len(diffused_mask) == len(original_mask) == len(time), f"{diffused_mask.shape=}, {original_mask=}, {time=}"
        assert diffused_mask.dim() == original_mask.dim() + 1, f"{diffused_mask.shape=}, {original_mask.shape=}"
        assert diffused_mask.dtype == original_mask.dtype == time.dtype == torch.int64, f"{diffused_mask.dtype=}, {original_mask.dtype=}, {time.dtype=}"
        assert 0 <= time.min() <= time.max() < self.num_steps, f"{time=}, {self.num_steps=}"
        # transform original mask into one-hot encoding
        original_mask = to_one_hot_encoding(original_mask, num_classes=self.num_classes, ignore_index=self.ignore_index)
        assert diffused_mask.shape == original_mask.shape, f"{diffused_mask.shape=}, {original_mask.shape=}"
        assert diffused_mask.shape[1] == original_mask.shape[1] == self.num_classes, f"{diffused_mask.shape=}, {original_mask.shape=}"
        # compute theta post
        alphas_t = self.alphas[time].view((len(time), 1, 1, 1))
        alphas_cumprod_tm1 = self.alphas_cumprod[time - 1].view((len(time), 1, 1, 1))
        alphas_t[time == 0] = 0.0
        alphas_cumprod_tm1[time == 0] = 1.0
        theta = (
            (alphas_t * diffused_mask + (1 - alphas_t) / self.num_classes) *
            (alphas_cumprod_tm1 * original_mask + (1 - alphas_cumprod_tm1) / self.num_classes)
        )
        theta_post = theta / theta.sum(dim=1, keepdim=True)
        return theta_post

    def __call__(self, y_pred: torch.Tensor, y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert type(y_pred) == torch.Tensor, f"{type(y_pred)=}"
        assert type(y_true) == dict, f"{type(y_true)=}"
        assert set(['original_mask', 'diffused_mask', 'time']).issubset(set(y_true.keys())), f"{y_true.keys()=}"
        prob_xtm1_given_xt_x0 = self.theta_post(diffused_mask=y_true['diffused_mask'], original_mask=y_true['original_mask'], time=y_true['time'])
        mask = y_true['original_mask'] != self.ignore_index
        mask = mask.unsqueeze(-3).expand(-1, self.num_classes, -1, -1)
        assert mask.shape == prob_xtm1_given_xt_x0.shape == y_pred.shape
        prob_xtm1_given_xt_x0 = prob_xtm1_given_xt_x0[mask]
        y_pred = y_pred[mask]
        loss = torch.nn.functional.kl_div(
            input=torch.log(y_pred),
            target=prob_xtm1_given_xt_x0, log_target=False,
            reduction='mean',
        )
        assert not loss.isnan(), f"{y_pred.min()=}, {y_pred.max()=}, {prob_xtm1_given_xt_x0.min()=}, {prob_xtm1_given_xt_x0.max()=}"
        assert loss.numel() == 1, f"{loss.shape=}"
        self.buffer.append(loss)
        return loss
