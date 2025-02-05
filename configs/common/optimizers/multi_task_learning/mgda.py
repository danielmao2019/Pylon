import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.MGDAOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
        'max_iters': 25,
    },
}
