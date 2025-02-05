import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.AlignedMTLOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
    },
}
