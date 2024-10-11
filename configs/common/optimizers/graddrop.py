import optimizers


optimizer_config = {
    'class': optimizers.GradDropOptimizer,
    'args': {
        'wrt_rep': True,
        'per_layer': False,
    },
}
