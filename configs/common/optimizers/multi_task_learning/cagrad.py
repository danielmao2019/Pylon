import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.CAGradOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
        'c': 0.5,
    },
}
