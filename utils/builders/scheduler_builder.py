import torch
import schedulers
from runners import BaseTrainer
from utils.builders import build_from_config


def build_scheduler(trainer: BaseTrainer, cfg: dict) -> torch.optim.lr_scheduler._LRScheduler:
    assert isinstance(cfg, dict) and set(cfg.keys()) == {'class', 'args'}
    cfg['args']['optimizer'] = trainer.optimizer
    if cfg['class'] == torch.optim.lr_scheduler.LambdaLR:
        assert set(cfg['args'].keys()).issubset({'optimizer', 'lr_lambda'}), f"{cfg['args'].keys()=}"
        lr_lambda_cfg = cfg['args']['lr_lambda']
        if lr_lambda_cfg['class'] == schedulers.lr_lambdas.ConstantLambda:
            pass
        elif lr_lambda_cfg['class'] == schedulers.lr_lambdas.WarmupLambda:
            if lr_lambda_cfg['args'].get('steps', None) is None:
                lr_lambda_cfg['args']['steps'] = len(trainer.train_dataloader)
        else:
            raise NotImplementedError
    elif cfg['class'] == torch.optim.lr_scheduler.PolynomialLR:
        assert set(cfg['args'].keys()).issubset({'optimizer', 'total_steps', 'power'}), f"{cfg['args'].keys()=}"
        cfg['args']['total_steps'] = len(trainer.train_dataloader) * trainer.tot_epochs
    else:
        raise NotImplementedError
    return build_from_config(cfg)
