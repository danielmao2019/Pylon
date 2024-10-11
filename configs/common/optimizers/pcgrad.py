import optimizers


optimizer_config = {
    'class': optimizers.PCGradOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
    },
}
