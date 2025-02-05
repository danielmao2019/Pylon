import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.MGDAOptimizer,
    'args': {
        'wrt_rep': True,
        'per_layer': False,
        'max_iters': 25,
    },
}
