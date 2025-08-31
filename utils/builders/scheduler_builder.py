import torch
import schedulers
from utils.builders import build_from_config


def build_scheduler(trainer: "runners.BaseTrainer", cfg: dict) -> torch.optim.lr_scheduler._LRScheduler:
    assert isinstance(cfg, dict) and set(cfg.keys()) == {'class', 'args'}
    cfg['args']['optimizer'] = trainer.optimizer.optimizer
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
    elif cfg['class'] == torch.optim.lr_scheduler.StepLR:
        if 'step_size' not in cfg['args'] or cfg['args']['step_size'] is None:
            cfg['args']['step_size'] = len(trainer.train_dataloader)
        else:
            pass
    elif cfg['class'] == torch.optim.lr_scheduler.ConstantLR:
        pass
    elif cfg['class'] == torch.optim.lr_scheduler.PolynomialLR:
        assert set(cfg['args'].keys()).issubset({'optimizer', 'total_iters', 'power'}), f"{cfg['args'].keys()=}"
        cfg['args']['total_iters'] = len(trainer.train_dataloader) * trainer.tot_epochs
    else:
        raise NotImplementedError
    return build_from_config(cfg)
