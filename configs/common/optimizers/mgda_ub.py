import optimizers


optimizer_config = {
    'class': optimizers.MGDAOptimizer,
    'args': {
        'wrt_rep': True,
        'per_layer': False,
        'max_iters': 25,
    },
}
