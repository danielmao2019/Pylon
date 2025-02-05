import optimizers
from configs.common.optimizers.standard import rmsprop_optimizer_config


optimizer_config = {
    'class': optimizers.wrappers.MultiPartOptimizer,
    'args': {
        'optimizer_cfgs': {
            'generator': rmsprop_optimizer_config,
            'generator': rmsprop_optimizer_config,
        },
    },
}
