import optimizers


optimizer_config = {
    'class': optimizers.IMTLOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
    },
}
