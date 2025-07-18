from typing import List, Dict, Any, Optional
import copy
import torch
from .base_trainer import BaseTrainer
import utils


class MultiStageTrainer(BaseTrainer):
    """Trainer class for multi-stage training with continuous epoch numbering.

    This trainer takes a list of configs for each stage. The number of epochs for each stage
    is read from the config files. It maintains continuous epoch numbering across stages and
    reinitializes components (except model) when entering a new stage.
    """

    def __init__(
        self,
        stage_configs: List[Dict[str, Any]],
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        """Initialize the multi-stage trainer.

        Args:
            stage_configs: List of config dictionaries for each stage. Each config must contain
                          an 'epochs' key specifying the number of epochs for that stage.
            device: Device to use for training
        """
        assert len(stage_configs) > 0, "Must provide at least one stage config"
        assert all('epochs' in config for config in stage_configs), "Each stage config must contain 'epochs' key"
        assert all(config['epochs'] > 0 for config in stage_configs), "All stage epoch counts must be positive"

        # Store stage info
        self.stage_configs = stage_configs
        self.stage_epochs = [config['epochs'] for config in stage_configs]

        # Initialize with first stage config
        super(MultiStageTrainer, self).__init__(config=stage_configs[0], device=device)

        # Track current stage
        self.current_stage = 0

    def _init_tot_epochs(self) -> None:
        """Override to use total epochs across all stages."""
        self.tot_epochs = sum(self.stage_epochs)

    def _init_determinism(self):
        # check dependencies
        assert hasattr(self, 'logger') and self.logger is not None, "logger="

        self.logger.info("Initializing determinism...")
        utils.determinism.set_determinism()

        # Set init seed (use init_seed from first stage)
        assert 'init_seed' in self.stage_configs[0].keys()
        init_seed = self.stage_configs[0]['init_seed']
        assert type(init_seed) == int, f"{type(init_seed)=}"
        utils.determinism.set_seed(seed=init_seed)

        # Get training seeds
        self.train_seeds = []
        for stage_config in self.stage_configs:
            assert 'train_seeds' in stage_config.keys()
            train_seeds = stage_config['train_seeds']
            assert type(train_seeds) == list, f"{type(train_seeds)=}"
            assert all(type(seed) == int for seed in train_seeds), f"{train_seeds=}"
            assert len(train_seeds) == stage_config['epochs'], f"{len(train_seeds)=}, {stage_config['epochs']=}"
            self.train_seeds.extend(train_seeds)
        assert len(self.train_seeds) == self.tot_epochs, f"{len(self.train_seeds)=}, {self.tot_epochs=}"

        # Get validation seeds
        self.val_seeds = []
        for stage_config in self.stage_configs:
            assert 'val_seeds' in stage_config.keys()
            val_seeds = stage_config['val_seeds']
            assert type(val_seeds) == list, f"{type(val_seeds)=}"
            assert all(type(seed) == int for seed in val_seeds), f"{val_seeds=}"
            assert len(val_seeds) == stage_config['epochs'], f"{len(val_seeds)=}, {stage_config['epochs']=}"
            self.val_seeds.extend(val_seeds)
        assert len(self.val_seeds) == self.tot_epochs, f"{len(self.val_seeds)=}, {self.tot_epochs=}"

        # Get test seed (use test_seed from first stage)
        assert 'test_seed' in self.stage_configs[0].keys()
        test_seed = self.stage_configs[0]['test_seed']
        assert type(test_seed) == int, f"{type(test_seed)=}"
        self.test_seed = test_seed

    def _get_stage_for_epoch(self, epoch: int) -> int:
        """Get the stage index for a given epoch number."""
        running_sum = 0
        for stage_idx, stage_epochs in enumerate(self.stage_epochs):
            running_sum += stage_epochs
            if epoch < running_sum:
                return stage_idx
        return len(self.stage_epochs) - 1

    def _switch_to_stage(self, stage_idx: int) -> None:
        """Switch to a new stage by reinitializing components with the stage's config."""
        self.logger.info(f"Switching to stage {stage_idx}")
        self.current_stage = stage_idx
        self.config = copy.deepcopy(self.stage_configs[stage_idx])

        # Set init seed
        assert 'init_seed' in self.config.keys()
        init_seed = self.config['init_seed']
        assert type(init_seed) == int, f"{type(init_seed)=}"
        utils.determinism.set_seed(seed=init_seed)

    def _reinitialize(self) -> None:
        # Reinitialize components except model
        self._init_dataloaders()
        self._init_criterion()
        self._init_metric()
        self._init_optimizer()
        self._init_scheduler()

    def _init_state(self) -> None:
        """Initialize state and handle resumption."""
        super(MultiStageTrainer, self)._init_state()
        # Determine which stage to resume from
        self.current_stage = self._get_stage_for_epoch(self.cum_epochs-1)
        self._switch_to_stage(self.current_stage)

    def run(self):
        """Run the multi-stage training process."""
        # Initialize components
        self._init_components_()
        start_epoch = self.cum_epochs
        self.logger.page_break()

        # Training and validation epochs
        for idx in range(start_epoch, self.tot_epochs):
            # Switch stage if needed
            stage_idx = self._get_stage_for_epoch(idx)
            if stage_idx != self.current_stage:
                self._switch_to_stage(stage_idx)
                self._reinitialize()

            # Set seed for this epoch
            utils.determinism.set_seed(seed=self.train_seeds[idx])

            # Run training and validation
            self._train_epoch_()
            self._val_epoch_()
            self.logger.page_break()
            self.cum_epochs = idx + 1

        # Test epoch
        self._test_epoch_()
