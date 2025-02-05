import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.DWAOptimizer,
    'args': {
        'window_size': 32,
    },
}
