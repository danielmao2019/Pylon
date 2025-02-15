import torch
import schedulers


scheduler_cfg = {
    'class': schedulers.wrappers.MultiPartScheduler,
    'args': {
        'scheduler_cfgs': {
            'generator': {
                'class': torch.optim.lr_scheduler.LambdaLR,
                'args': {
                    'lr_lambda': {
                        'class': schedulers.lr_lambdas.ConstantLambda,
                        'args': {},
                    },
                },
            },
            'discriminator': {
                'class': torch.optim.lr_scheduler.LambdaLR,
                'args': {
                    'lr_lambda': {
                        'class': schedulers.lr_lambdas.ConstantLambda,
                        'args': {},
                    },
                },
            },
        },
    }
}
