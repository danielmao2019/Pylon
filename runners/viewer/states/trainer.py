import numpy as np
from pathlib import Path
import torch
import importlib.util
from utils.automation.cfg_log_conversion import get_work_dir, get_repo_root
from runners.viewer.utils import get_default_colors
import utils.determinism


class TrainingState:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.current_iteration = 0
        self.current_sample_idx = 0
        self.current_epoch = 0
        self.class_colors = get_default_colors()
        self.device = torch.device('cuda')

        # Load config and initialize trainer
        self.config = self._load_config()
        
        # Store batch size from config
        self.batch_size = self.config['train_dataloader']['args']['batch_size']

        # Initialize trainer using the runner class specified in config
        self.trainer = self.config['runner'](config=self.config)
        self.trainer._init_components_()

        # Store references to trainer components we need
        self.model = self.trainer.model
        self.train_dataloader = self.trainer.train_dataloader
        self.criterion = self.trainer.criterion
        self.optimizer = self.trainer.optimizer
        self.scheduler = self.trainer.scheduler
        
        # Initialize dataloader iterator and process first batch
        self.dataloader_iter = iter(self.train_dataloader)
        self.current_batch = next(self.dataloader_iter)
        dp = {
            'inputs': self.current_batch['inputs'],
            'labels': self.current_batch['labels']
        }
        self.trainer._train_step_(dp)
        self.current_outputs = dp['outputs']

    def _load_config(self):
        """Load config from Python file and modify work_dir for viewer."""
        # Load the config module
        spec = importlib.util.spec_from_file_location("config_file", self.config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the config dictionary
        config = module.config

        # Get the corresponding work directory using repo root
        work_dir = Path(get_work_dir(str(self.config_path)))
        # Get relative path from repo root
        rel_path = work_dir.relative_to(get_repo_root() / 'logs')
        # Construct new path with 'viewer' as second directory
        viewer_work_dir = get_repo_root() / 'logs' / 'viewer' / rel_path

        # Update config with viewer work directory
        config['work_dir'] = str(viewer_work_dir)

        # Append repo root to dataset paths
        repo_root = get_repo_root()
        for dataset_type in ['train_dataset', 'val_dataset', 'test_dataset']:
            if config.get(dataset_type, None):
                assert isinstance(config[dataset_type], dict)
                assert 'args' in config[dataset_type]
                assert 'data_root' in config[dataset_type]['args']
                rel_path = config[dataset_type]['args']['data_root']
                config[dataset_type]['args']['data_root'] = str(repo_root / rel_path)

        return config

    def next_epoch(self):
        """Move to next training epoch using trainer's functionality."""
        # Set seed for deterministic behavior
        utils.determinism.set_seed(seed=self.trainer.train_seeds[self.current_epoch])
        
        # Call trainer's train epoch method
        self.trainer._train_epoch_()
        
        # Update trainer's cumulative epoch counter
        self.trainer.cum_epochs = self.trainer.cum_epochs + 1
        
        # Update our state
        self.current_epoch += 1
        self.current_iteration = 0
        self.current_sample_idx = 0
        self.dataloader_iter = iter(self.train_dataloader)
        
        return self.current_epoch

    def next_iteration(self):
        """Move to next training iteration."""
        self.current_iteration += 1
        self.current_sample_idx = 0  # Reset sample index for new batch
        
        # Handle end of iterator
        if self.current_iteration >= len(self.train_dataloader):
            self.current_iteration = 0
            return self.next_epoch()
            
        # Load and process next batch
        self.current_batch = next(self.dataloader_iter)
        dp = {
            'inputs': self.current_batch['inputs'],
            'labels': self.current_batch['labels']
        }
        self.trainer._train_step_(dp)
        self.current_outputs = dp['outputs']
        
        return self.current_iteration

    def next_sample(self):
        """Move to next sample in current batch if available."""
        self.current_sample_idx = (self.current_sample_idx + 1) % self.batch_size
        return self.current_sample_idx

    def get_current_data(self):
        """Get data for current sample in current batch."""
        return {
            'input1': self.current_batch['inputs']['img_1'][self.current_sample_idx].cpu().numpy(),  # [C,H,W]
            'input2': self.current_batch['inputs']['img_2'][self.current_sample_idx].cpu().numpy(),  # [C,H,W]
            'pred': self.current_outputs[self.current_sample_idx].argmax(dim=0).cpu().numpy(),  # [H,W]
            'gt': self.current_batch['labels']['change_map'][self.current_sample_idx].cpu().numpy(),  # [H,W]
        }

    def get_navigation_info(self):
        """Get current navigation state."""
        return {
            'current_iteration': self.current_iteration,
            'total_iterations': len(self.train_dataloader),
            'current_sample': self.current_sample_idx,
            'batch_size': self.batch_size,
            'current_epoch': self.current_epoch
        }
