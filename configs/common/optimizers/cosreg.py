import optimizers


optimizer_config = {
    'class': optimizers.CosRegOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
        'penalty': 10.0,
    },
}
