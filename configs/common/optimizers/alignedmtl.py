import optimizers


optimizer_config = {
    'class': optimizers.AlignedMTLOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
    },
}
