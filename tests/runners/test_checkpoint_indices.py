import pytest
import torch
from unittest.mock import Mock, patch
from runners.base_trainer import BaseTrainer


class TestableBaseTrainer(BaseTrainer):
    """Testable version of BaseTrainer that implements abstract methods."""
    
    def __init__(self, tot_epochs=10, checkpoint_method='latest'):
        # Create minimal config
        config = {
            'work_dir': './logs/tests',
            'epochs': tot_epochs,  # BaseTrainer expects 'epochs' not 'tot_epochs'
            'checkpoint_method': checkpoint_method
        }
        
        # Mock device
        device = torch.device('cpu')
        
        # Initialize BaseTrainer
        super().__init__(config, device)
        
        # Mock required attributes that would normally be set by _init_components_
        self.logger = Mock()
        self.logger.info = Mock()
    
    # Implement abstract methods (minimal implementations for testing)
    def _init_components_(self):
        pass
    
    def _load_checkpoint_(self):
        pass
    
    def _train_epoch_(self):
        pass
    
    def _eval_epoch_(self):
        pass
    
    def _init_optimizer_(self):
        pass
    
    def _init_scheduler_(self):
        pass
    
    def _set_gradients_(self, dp):
        pass


@pytest.mark.parametrize("tot_epochs,checkpoint_method,expected_indices", [
    (10, 'latest', [9]),
    (10, 'all', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (10, 2, [1, 3, 5, 7, 9]),  # Every 2 epochs + last
    (10, 3, [2, 5, 8, 9]),     # Every 3 epochs + last
    (10, 5, [4, 9]),           # Every 5 epochs + last
    (5, 2, [1, 3, 4]),         # Every 2 epochs + last
    (3, 5, [2]),               # Interval > tot_epochs, only last
])
def test_checkpoint_indices_calculation(tot_epochs, checkpoint_method, expected_indices):
    """Test that checkpoint indices are calculated correctly for different methods."""
    trainer = TestableBaseTrainer(tot_epochs=tot_epochs, checkpoint_method=checkpoint_method)
    trainer._init_checkpoint_indices()
    
    assert trainer.checkpoint_indices == expected_indices


def test_checkpoint_indices_edge_cases():
    """Test edge cases for checkpoint indices calculation."""
    # Test with 1 epoch
    trainer = TestableBaseTrainer(tot_epochs=1, checkpoint_method='latest')
    trainer._init_checkpoint_indices()
    assert trainer.checkpoint_indices == [0]  # Only epoch 0
    
    # Test with interval larger than total epochs
    trainer = TestableBaseTrainer(tot_epochs=3, checkpoint_method=10)
    trainer._init_checkpoint_indices()
    assert trainer.checkpoint_indices == [2]  # Only last epoch
    
    # Test with interval = 1 (every epoch)
    trainer = TestableBaseTrainer(tot_epochs=3, checkpoint_method=1)
    trainer._init_checkpoint_indices()
    assert trainer.checkpoint_indices == [0, 1, 2]  # Every epoch


def test_checkpoint_indices_config_default():
    """Test that default checkpoint_method works correctly."""
    # Test without checkpoint_method in config (should default to 'latest')
    config = {
        'work_dir': './logs/tests',
        'epochs': 5,  # BaseTrainer expects 'epochs'
        # No checkpoint_method specified
    }
    
    trainer = TestableBaseTrainer.__new__(TestableBaseTrainer)
    BaseTrainer.__init__(trainer, config, torch.device('cpu'))
    trainer.logger = Mock()
    trainer.logger.info = Mock()
    
    trainer._init_checkpoint_indices()
    assert trainer.checkpoint_indices == [4]  # Latest only (last epoch)
