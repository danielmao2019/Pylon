import optimizers
from configs.common.optimizers.standard import adam_optimizer_config


optimizer_config = {
    'class': optimizers.wrappers.MultiPartOptimizer,
    'args': {
        'optimizer_cfgs': {
            'generator': {
                'class': optimizers.SingleTaskOptimizer,
                'args': {
                    'optimizer_config': adam_optimizer_config,
                },
            },
            'discriminator':  {
                'class': optimizers.SingleTaskOptimizer,
                'args': {
                    'optimizer_config': adam_optimizer_config,
                },
            },
        },
    },
}
