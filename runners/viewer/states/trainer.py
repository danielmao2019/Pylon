import numpy as np
from pathlib import Path
import torch
from utils.builders import build_from_config
import json
import utils.determinism
import importlib.util
import sys
from utils.automation.cfg_log_conversion import get_work_dir, get_repo_root


class TrainingState:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.current_iteration = 0
        self.class_colors = self._get_default_colors()
        self.device = torch.device('cuda')
        
        # Load config and initialize trainer components
        self.config = self._load_config()
        self._init_determinism()
        self._init_trainer()
        
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
    
    def _init_determinism(self):
        """Initialize determinism settings."""
        utils.determinism.set_determinism()
        if 'init_seed' in self.config:
            utils.determinism.set_seed(seed=self.config['init_seed'])
    
    def _init_trainer(self):
        """Initialize trainer components."""
        # Initialize model
        if self.config.get('model', None):
            self.model = build_from_config(self.config['model'])
            self.model = self.model.to(self.device)
            self.model.train()  # Set to training mode
        else:
            self.model = None
            
        # Initialize training dataloader
        if self.config.get('train_dataset', None) and self.config.get('train_dataloader', None):
            train_dataset = build_from_config(self.config['train_dataset'])
            self.train_dataloader = build_from_config(
                dataset=train_dataset,
                shuffle=True,
                config=self.config['train_dataloader']
            )
        else:
            self.train_dataloader = None
            
        # Initialize criterion
        if self.config.get('criterion', None):
            self.criterion = build_from_config(self.config['criterion'])
            self.criterion = self.criterion.to(self.device)
        else:
            self.criterion = None
            
        # Initialize optimizer
        if self.config.get('optimizer', None) and self.model:
            self.optimizer = build_from_config(
                params=self.model.parameters(),
                config=self.config['optimizer']
            )
        else:
            self.optimizer = None
            
        # Initialize scheduler
        if self.config.get('scheduler', None) and self.optimizer:
            scheduler_cfg = self.config['scheduler']
            scheduler_cfg['args']['optimizer'] = self.optimizer
            self.scheduler = build_from_config(scheduler_cfg)
        else:
            self.scheduler = None
    
    def _get_default_colors(self):
        """Return default color mapping for visualization."""
        return {
            0: [0, 0, 0],      # Background (black)
            1: [255, 0, 0],    # Change class 1 (red)
            2: [0, 255, 0],    # Change class 2 (green)
            3: [0, 0, 255],    # Change class 3 (blue)
        }
    
    def get_current_data(self):
        """Get data for current iteration."""
        if not hasattr(self, 'current_batch'):
            # Get first batch
            self.current_batch = next(iter(self.train_dataloader))
            
        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in self.current_batch['inputs'].items()}
        labels = {k: v.to(self.device) for k, v in self.current_batch['labels'].items()}
            
        # Forward pass
        self.optimizer.zero_grad()  # Zero gradients
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(inputs)
            losses = self.criterion(y_pred=outputs, y_true=labels)
            
        # Backward pass
        losses['total'].backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
            
        # Convert tensors to numpy for visualization
        return {
            'input1': self.current_batch['inputs']['image1'].cpu().numpy(),
            'input2': self.current_batch['inputs']['image2'].cpu().numpy(),
            'pred': outputs['logits'].argmax(dim=1).cpu().numpy(),
            'gt': self.current_batch['labels']['change_map'].cpu().numpy()
        }
    
    def next_iteration(self):
        """Move to next iteration if available."""
        try:
            self.current_batch = next(iter(self.train_dataloader))
            self.current_iteration += 1
            return self.current_iteration
        except StopIteration:
            # Reset dataloader iterator
            self.train_dataloader = iter(self.train_dataloader)
            return self.current_iteration
    
    def class_to_rgb(self, class_indices):
        """Convert class indices to RGB values."""
        rgb = np.zeros((*class_indices.shape, 3), dtype=np.uint8)
        for class_idx, color in self.class_colors.items():
            mask = class_indices == class_idx
            rgb[mask] = color
        return rgb
