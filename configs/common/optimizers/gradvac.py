import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.GradVacOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
        'beta': 0.01,
    },
}
