import optimizers
from configs.common.optimizers.standard import rmsprop_optimizer_config


optimizer_config = {
    'class': optimizers.wrappers.MultiPartOptimizer,
    'args': {
        'optimizer_cfgs': {
            'G_optimizer': rmsprop_optimizer_config,
            'D_optimizer': rmsprop_optimizer_config,
        },
    },
}
