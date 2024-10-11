import optimizers


optimizer_config = {
    'class': optimizers.RGWOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
    },
}
