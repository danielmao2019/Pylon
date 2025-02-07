import torch


sgd_optimizer_cfg = {
    'class': torch.optim.SGD,
    'args': {
        'lr': 1.0e-03,
        'momentum': 0.9,
        'weight_decay': 1.0e-04,
    },
}
