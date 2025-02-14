import torch


sgd_optimizer_config = {
    'class': torch.optim.SGD,
    'args': {
        'lr': 1.0e-04,
        'momentum': 0.9,
    },
}

rmsprop_optimizer_config = {
    'class': torch.optim.RMSprop,
    'args': {
        'lr': 1.0e-04,
    },
}

adam_optimizer_config = {
    'class': torch.optim.Adam,
    'args': {
        'lr': 1.0e-04,
    },
}
