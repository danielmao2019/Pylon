import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.AlignedMTLOptimizer,
    'args': {
        'wrt_rep': True,
        'per_layer': False,
    },
}
