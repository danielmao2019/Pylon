from runners import BaseEvaluator


config = {
    'runner': BaseEvaluator,
    'work_dir': None,
    # seeds
    'seed': None,
    # dataset config
    'eval_dataset': None,
    'eval_dataloader': None,
    'metric': None,
    # model config
    'model': None,
}
