import optimizers
from configs.common.optimizers.standard import rmsprop_optimizer_config


optimizer_config = {
    'class': optimizers.wrappers.MultiPartOptimizer,
    'args': {
        'optimizer_cfgs': {
            'generator': {
                'class': optimizers.SingleTaskOptimizer,
                'args': {
                    'optimizer_config': rmsprop_optimizer_config,
                },
            },
            'generator':  {
                'class': optimizers.SingleTaskOptimizer,
                'args': {
                    'optimizer_config': rmsprop_optimizer_config,
                },
            },
        },
    },
}
