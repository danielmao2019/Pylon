import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.GradDropOptimizer,
    'args': {
        'wrt_rep': True,
        'per_layer': False,
    },
}
