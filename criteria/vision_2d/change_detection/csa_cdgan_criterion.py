from typing import Dict
import torch
from criteria.wrappers import SingleTaskCriterion, MultiTaskCriterion


class CSA_CDGAN_GeneratorCriterion(SingleTaskCriterion):

    def __init__(self, g_weight: float, d_weight: float) -> None:
        super(CSA_CDGAN_GeneratorCriterion, self).__init__()
        _g_weight = g_weight / (g_weight + d_weight)
        _d_weight = d_weight / (g_weight + d_weight)
        self.g_weight = _g_weight
        self.d_weight = _d_weight
        self.l_bce = torch.nn.BCELoss()
        self.l_con = torch.nn.L1Loss()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        err_d_fake = self.l_bce(y_pred['pred_fake_g'], y_true['fake_label'])
        err_g = self.l_con(torch.nn.functional.softmax(y_pred['gen_image'], dim=1), y_true['change_map'])
        err_g_total = self.g_weight * err_g + self.d_weight * err_d_fake
        assert err_g_total.ndim == 0, f"{err_g_total.shape=}"
        # log loss
        self.buffer.append(err_g_total.detach().cpu())
        return err_g_total


class CSA_CDGAN_DiscriminatorCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        super(CSA_CDGAN_DiscriminatorCriterion, self).__init__()
        self.l_bce = torch.nn.BCELoss()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        err_d_real = self.l_bce(y_pred['pred_real'], y_true['real_label'])
        err_d_fake = self.l_bce(y_pred['pred_fake_d'], y_true['fake_label'])
        err_d_total = (err_d_real + err_d_fake) * 0.5
        assert err_d_total.ndim == 0, f"{err_d_total.shape=}"
        # log loss
        self.buffer.append(err_d_total.detach().cpu())
        return err_d_total


class CSA_CDGAN_Criterion(MultiTaskCriterion, torch.nn.Module):

    def __init__(self, g_weight: float, d_weight: float) -> None:
        self.task_criteria = {
            'generator': CSA_CDGAN_GeneratorCriterion(g_weight, d_weight),
            'discriminator': CSA_CDGAN_DiscriminatorCriterion(),
        }
        self.task_names = set(self.task_criteria.keys())
        self.reset_buffer()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input checks
        assert isinstance(y_pred, dict) and set(y_pred.keys()) == {'gen_image', 'pred_real', 'pred_fake_g', 'pred_fake_d'}
        assert isinstance(y_true, dict) and set(y_true.keys()) == {'change_map', 'real_label', 'fake_label'}
        losses: Dict[str, torch.Tensor] = dict(
            (task, self.task_criteria[task](y_pred=y_pred, y_true=y_true))
            for task in self.task_names
        )
        return losses

    def to(self, *args, **kwargs) -> None:
        return self
