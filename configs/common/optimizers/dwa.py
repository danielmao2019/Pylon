import optimizers


optimizer_config = {
    'class': optimizers.DWAOptimizer,
    'args': {
        'window_size': 32,
    },
}
