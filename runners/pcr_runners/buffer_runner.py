from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer


class BufferRunner(SupervisedSingleTaskTrainer):
    
    def _init_model_(self) -> None:
        super(BufferRunner, self)._init_model_()
        freeze_stages = ['Ref', 'Desc', 'Keypt', 'Inlier'].remove(self.config['stage'])
        for stage in freeze_stages:
            for param in getattr(self.model, stage).parameters():
                param.requires_grad = False
