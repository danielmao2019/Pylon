import optimizers


optimizer_config = {
    'class': optimizers.GradNormOptimizer,
    'args': {
        'wrt_rep': False,
        'alpha': 1.5,
    },
}
