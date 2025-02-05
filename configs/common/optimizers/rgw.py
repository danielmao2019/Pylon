import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.RGWOptimizer,
    'args': {
        'wrt_rep': False,
        'per_layer': False,
    },
}
