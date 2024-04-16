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
        alphas_t = self.alphas[time][..., None, None, None]
        alphas_cumprod_tm1 = self.alphas_cumprod[time - 1][..., None, None, None]
        alphas_t[time == 0] = 0.0
        alphas_cumprod_tm1[time == 0] = 1.0
        theta = (
            (alphas_t * diffused_mask + (1 - alphas_t) / self.num_classes) *
            (alphas_cumprod_tm1 * original_mask + (1 - alphas_cumprod_tm1) / self.num_classes)
        )
        theta_post = theta / theta.sum(dim=1, keepdim=True)
        assert not theta_post.isnan().any(), f"alphas_t in [{alphas_t.min()}, {alphas_t.max()}], alphas_cumprod_tm1 in {alphas_cumprod_tm1.min()}, {alphas_cumprod_tm1.max()}]."
        return theta_post

    def theta_post_prob(self, diffused_mask: torch.Tensor, pred_mask: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        This is equivalent to calling theta_post with all possible values of x0
        from 0 to C-1 and multiplying each answer times theta_x0[:, c].

        This should be used when x0 is unknown and what you have is a probability
        distribution over x0. If x0 is one-hot encoded (i.e., only 0's and 1's),
        use theta_post instead.
        """
        assert diffused_mask.shape == pred_mask.shape, f"{diffused_mask.shape=}, {pred_mask.shape=}"
        assert 0 <= time.min() <= time.max() < self.num_steps, f"{time=}, {self.num_steps=}"
        alphas_t = self.alphas[time][..., None, None, None]
        cumalphas_tm1 = self.alphas_cumprod[time - 1][..., None, None, None, None]
        alphas_t[time == 0] = 0.0
        cumalphas_tm1[time == 0] = 1.0

        # We need to evaluate theta_post for all values of x0
        x0 = torch.eye(self.num_classes, device=diffused_mask.device)[None, :, :, None, None]
        # theta_xt_xtm1.shape == [B, C, H, W]
        theta_xt_xtm1 = alphas_t * diffused_mask + (1 - alphas_t) / self.num_classes
        # theta_xtm1_x0.shape == [B, C1, C2, H, W]
        theta_xtm1_x0 = cumalphas_tm1 * x0 + (1 - cumalphas_tm1) / self.num_classes

        aux = theta_xt_xtm1[:, :, None] * theta_xtm1_x0
        # theta_xtm1_xtx0 == [B, C1, C2, H, W]
        theta_xtm1_xtx0 = aux / aux.sum(dim=1, keepdim=True)

        # theta_x0.shape = [B, C, H, W]

        return torch.einsum("bcdhw,bdhw->bchw", theta_xtm1_xtx0, pred_mask)

    def __call__(self, y_pred: torch.Tensor, y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert type(y_pred) == torch.Tensor, f"{type(y_pred)=}"
        assert type(y_true) == dict, f"{type(y_true)=}"
        assert set(['original_mask', 'diffused_mask', 'time']).issubset(set(y_true.keys())), f"{y_true.keys()=}"
        prob_xtm1_given_xt_x0 = self.theta_post(diffused_mask=y_true['diffused_mask'], original_mask=y_true['original_mask'], time=y_true['time'])
        prob_xtm1_given_xt_x0pred = self.theta_post_prob(diffused_mask=y_true['diffused_mask'], pred_mask=y_pred, time=y_true['time'])
        loss = torch.nn.functional.kl_div(
            torch.log(torch.clamp(prob_xtm1_given_xt_x0pred, min=1e-12)),
            prob_xtm1_given_xt_x0,
            reduction='mean',
        )
        assert not loss.isnan(), f"{prob_xtm1_given_xt_x0pred.min()=}, {prob_xtm1_given_xt_x0pred.max()=}, {prob_xtm1_given_xt_x0.min()=}, {prob_xtm1_given_xt_x0.max()=}"
        assert loss.numel() == 1, f"{loss.shape=}"
        self.buffer.append(loss)
        return loss
