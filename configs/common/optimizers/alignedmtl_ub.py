import optimizers


optimizer_config = {
    'class': optimizers.AlignedMTLOptimizer,
    'args': {
        'wrt_rep': True,
        'per_layer': False,
    },
}
