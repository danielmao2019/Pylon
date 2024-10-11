import optimizers


optimizer_config = {
    'class': optimizers.MGDAOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
        'max_iters': 25,
    },
}
