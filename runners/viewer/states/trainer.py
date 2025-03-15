import numpy as np
from pathlib import Path
import torch
import importlib.util
from utils.automation.cfg_log_conversion import get_work_dir, get_repo_root


class TrainingState:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.current_batch_idx = 0
        self.current_sample_idx = 0
        self.class_colors = self._get_default_colors()
        self.device = torch.device('cuda')

        # Load config and initialize trainer
        self.config = self._load_config()

        # Initialize trainer using the runner class specified in config
        self.trainer = self.config['runner'](config=self.config)
        self.trainer._init_components_()

        # Store references to trainer components we need
        self.model = self.trainer.model
        self.train_dataloader = self.trainer.train_dataloader
        self.criterion = self.trainer.criterion
        self.optimizer = self.trainer.optimizer
        self.scheduler = self.trainer.scheduler
        
        # Initialize first batch
        self._load_current_batch()

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

    def _load_current_batch(self):
        """Load the current batch based on current_batch_idx."""
        # Convert dataloader to list for random access
        if not hasattr(self, '_batches'):
            self._batches = list(self.train_dataloader)
        
        # Get current batch
        self.current_batch = self._batches[self.current_batch_idx]
        
        # Create data point dictionary
        dp = {
            'inputs': self.current_batch['inputs'],
            'labels': self.current_batch['labels']
        }

        # Use trainer's train step
        self.trainer._train_step_(dp)
        
        # Store processed outputs
        self.current_outputs = dp['outputs']

    def get_current_data(self):
        """Get data for current sample in current batch."""
        # Convert tensors to numpy for visualization
        return {
            'input1': self.current_batch['inputs']['img_1'][self.current_sample_idx].cpu().numpy(),
            'input2': self.current_batch['inputs']['img_2'][self.current_sample_idx].cpu().numpy(),
            'pred': self.current_outputs['logits'][self.current_sample_idx].argmax(dim=0).cpu().numpy(),
            'gt': self.current_batch['labels']['change_map'][self.current_sample_idx].cpu().numpy()
        }

    def next_batch(self):
        """Move to next batch if available."""
        if not hasattr(self, '_batches'):
            self._load_current_batch()
            return 0

        next_idx = (self.current_batch_idx + 1) % len(self._batches)
        if next_idx != self.current_batch_idx:
            self.current_batch_idx = next_idx
            self.current_sample_idx = 0  # Reset sample index for new batch
            self._load_current_batch()
        return self.current_batch_idx

    def prev_batch(self):
        """Move to previous batch if available."""
        if not hasattr(self, '_batches'):
            self._load_current_batch()
            return 0

        next_idx = (self.current_batch_idx - 1) % len(self._batches)
        if next_idx != self.current_batch_idx:
            self.current_batch_idx = next_idx
            self.current_sample_idx = 0  # Reset sample index for new batch
            self._load_current_batch()
        return self.current_batch_idx

    def next_sample(self):
        """Move to next sample in current batch if available."""
        batch_size = len(self.current_batch['inputs']['img_1'])
        next_idx = (self.current_sample_idx + 1) % batch_size
        if next_idx != self.current_sample_idx:
            self.current_sample_idx = next_idx
        return self.current_sample_idx

    def prev_sample(self):
        """Move to previous sample in current batch if available."""
        batch_size = len(self.current_batch['inputs']['img_1'])
        next_idx = (self.current_sample_idx - 1) % batch_size
        if next_idx != self.current_sample_idx:
            self.current_sample_idx = next_idx
        return self.current_sample_idx

    def get_navigation_info(self):
        """Get current navigation state."""
        batch_size = len(self.current_batch['inputs']['img_1'])
        return {
            'current_batch': self.current_batch_idx,
            'total_batches': len(self._batches) if hasattr(self, '_batches') else 1,
            'current_sample': self.current_sample_idx,
            'batch_size': batch_size
        }

    def _get_default_colors(self):
        """Return default color mapping for visualization."""
        return {
            0: [0, 0, 0],      # Background (black)
            1: [255, 0, 0],    # Change class 1 (red)
            2: [0, 255, 0],    # Change class 2 (green)
            3: [0, 0, 255],    # Change class 3 (blue)
        }

    def class_to_rgb(self, class_indices):
        """Convert class indices to RGB values."""
        rgb = np.zeros((*class_indices.shape, 3), dtype=np.uint8)
        for class_idx, color in self.class_colors.items():
            mask = class_indices == class_idx
            rgb[mask] = color
        return rgb
