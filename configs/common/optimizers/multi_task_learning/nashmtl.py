import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.NashMTLOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
        'max_iter': 20,
    },
}
