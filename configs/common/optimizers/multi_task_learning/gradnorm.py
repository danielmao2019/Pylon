import optimizers


optimizer_config = {
    'class': optimizers.multi_task_optimizers.GradNormOptimizer,
    'args': {
        'wrt_rep': False,
        'alpha': 1.5,
    },
}
