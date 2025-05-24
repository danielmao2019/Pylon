from typing import List, Dict, Any, Optional
import torch
from .base_trainer import BaseTrainer
import utils
from utils.builders import build_from_config
import os


class MultiStageTrainer(BaseTrainer):
    """Trainer class for multi-stage training with continuous epoch numbering.
    
    This trainer takes a list of configs and corresponding epoch counts for each stage.
    It maintains continuous epoch numbering across stages and reinitializes components
    (except model) when entering a new stage.
    """

    def __init__(
        self,
        stage_configs: List[Dict[str, Any]],
        stage_epochs: List[int],
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        """Initialize the multi-stage trainer.
        
        Args:
            stage_configs: List of config dictionaries for each stage
            stage_epochs: List of number of epochs for each stage
            device: Device to use for training
        """
        assert len(stage_configs) == len(stage_epochs), "Number of configs must match number of epoch counts"
        assert all(epochs > 0 for epochs in stage_epochs), "All stage epoch counts must be positive"
        
        # Store stage info
        self.stage_configs = stage_configs
        self.stage_epochs = stage_epochs
        self.tot_epochs = sum(stage_epochs)
        
        # Initialize with first stage config
        super().__init__(config=stage_configs[0], device=device)
        
        # Track current stage
        self.current_stage = 0
        self.stage_start_epoch = 0

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
        if stage_idx == self.current_stage:
            return
            
        self.logger.info(f"Switching to stage {stage_idx}")
        self.current_stage = stage_idx
        self.config = self.stage_configs[stage_idx]
        
        # Calculate stage start epoch
        self.stage_start_epoch = sum(self.stage_epochs[:stage_idx])
        
        # Reinitialize components except model
        self._init_dataloaders_()
        self._init_criterion_()
        self._init_metric_()
        self._init_optimizer_()
        self._init_scheduler_()

    def _init_state_(self) -> None:
        """Initialize state and handle resumption."""
        self.logger.info("Initializing state...")
        # init epoch numbers
        self.cum_epochs = 0
        if self.work_dir is None:
            return
            
        # determine where to resume from
        load_idx: Optional[int] = None
        for idx in range(self.tot_epochs):
            if not utils.automation.run_status.check_epoch_finished(
                epoch_dir=os.path.join(self.work_dir, f"epoch_{idx}"),
                expected_files=self.expected_files,
            ):
                break
            if os.path.isfile(os.path.join(self.work_dir, f"epoch_{idx}", "checkpoint.pt")):
                load_idx = idx
                
        # resume state
        if load_idx is None:
            self.logger.info("Training from scratch.")
            return
            
        # Determine which stage to resume from
        stage_idx = self._get_stage_for_epoch(load_idx)
        self._switch_to_stage(stage_idx)
        
        # Load checkpoint
        checkpoint_filepath = os.path.join(self.work_dir, f"epoch_{load_idx}", "checkpoint.pt")
        try:
            self.logger.info(f"Loading checkpoint from {checkpoint_filepath}...")
            checkpoint = torch.load(checkpoint_filepath)
            self._load_checkpoint_(checkpoint)
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load checkpoint at {checkpoint_filepath}: {e}")

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
                
            # Set seed for this epoch
            utils.determinism.set_seed(seed=self.train_seeds[idx])
            
            # Run training and validation
            self._train_epoch_()
            self._val_epoch_()
            self.logger.page_break()
            self.cum_epochs = idx + 1
            
        # Test epoch
        self._test_epoch_() 