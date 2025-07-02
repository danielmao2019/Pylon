from typing import Dict, Any
import torch
import torch.nn as nn
import pytest
from optimizers.wrappers.multi_part_optimizer import MultiPartOptimizer
from optimizers.single_task_optimizer import SingleTaskOptimizer


def test_multi_part_optimizer_initialization():
    """Test that MultiPartOptimizer correctly initializes with optimizer configs."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    optimizer_cfgs = {
        'encoder': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': list(model[0].parameters()),
                        'lr': 0.01,
                        'momentum': 0.9
                    }
                }
            }
        },
        'decoder': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.Adam,
                    'args': {
                        'params': list(model[2].parameters()),
                        'lr': 0.001
                    }
                }
            }
        }
    }
    
    optimizer = MultiPartOptimizer(optimizer_cfgs)
    
    assert hasattr(optimizer, 'optimizers')
    assert 'encoder' in optimizer.optimizers
    assert 'decoder' in optimizer.optimizers
    assert isinstance(optimizer.optimizers['encoder'], SingleTaskOptimizer)
    assert isinstance(optimizer.optimizers['decoder'], SingleTaskOptimizer)


def test_multi_part_optimizer_state_dict_save_load():
    """Test that MultiPartOptimizer correctly saves and loads state dicts."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    optimizer_cfgs = {
        'encoder': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': list(model[0].parameters()),
                        'lr': 0.01,
                        'momentum': 0.9
                    }
                }
            }
        },
        'decoder': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': list(model[2].parameters()),
                        'lr': 0.001,
                        'momentum': 0.95
                    }
                }
            }
        }
    }
    
    # Create MultiPartOptimizer
    optimizer = MultiPartOptimizer(optimizer_cfgs)
    
    # Run a few optimization steps to build up state
    for _ in range(3):
        for opt in optimizer.optimizers.values():
            opt.zero_grad()
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for opt in optimizer.optimizers.values():
            opt.step()
    
    # Save state dict
    state_dict = optimizer.state_dict()
    
    # Verify state dict structure
    assert isinstance(state_dict, dict)
    assert 'encoder' in state_dict
    assert 'decoder' in state_dict
    assert 'state' in state_dict['encoder']
    assert 'param_groups' in state_dict['encoder']
    assert 'state' in state_dict['decoder']
    assert 'param_groups' in state_dict['decoder']
    
    # Create a new optimizer with the same config
    new_optimizer = MultiPartOptimizer(optimizer_cfgs)
    
    # Load state dict
    new_optimizer.load_state_dict(state_dict)
    
    # Verify that the state was loaded correctly
    new_state_dict = new_optimizer.state_dict()
    
    # Check that the loaded state matches the original
    for name in ['encoder', 'decoder']:
        assert name in new_state_dict
        
        # Check param_groups
        orig_pg = state_dict[name]['param_groups']
        new_pg = new_state_dict[name]['param_groups']
        assert len(orig_pg) == len(new_pg)
        
        for i, (orig, new) in enumerate(zip(orig_pg, new_pg)):
            for key in ['lr', 'momentum']:
                if key in orig:
                    assert orig[key] == new[key]
        
        # Check state (momentum buffers)
        orig_state = state_dict[name]['state']
        new_state = new_state_dict[name]['state']
        assert len(orig_state) == len(new_state)
        
        for param_id in orig_state:
            if 'momentum_buffer' in orig_state[param_id]:
                assert torch.allclose(
                    orig_state[param_id]['momentum_buffer'],
                    new_state[param_id]['momentum_buffer']
                )


def test_multi_part_optimizer_reset_buffer():
    """Test that reset_buffer works for all optimizers."""
    optimizer_cfgs = {
        'part1': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': [torch.nn.Parameter(torch.randn(2, 2))],
                        'lr': 0.01
                    }
                }
            }
        },
        'part2': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': [torch.nn.Parameter(torch.randn(2, 2))],
                        'lr': 0.01
                    }
                }
            }
        }
    }
    
    optimizer = MultiPartOptimizer(optimizer_cfgs)
    
    # Reset buffers should not raise an error
    optimizer.reset_buffer()
    
    # Verify buffers are reset
    for opt in optimizer.optimizers.values():
        assert hasattr(opt, 'buffer')
        assert opt.buffer == []


def test_multi_part_optimizer_summarize():
    """Test that summarize works correctly."""
    optimizer_cfgs = {
        'part1': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': [torch.nn.Parameter(torch.randn(2, 2))],
                        'lr': 0.01
                    }
                }
            }
        },
        'part2': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': [torch.nn.Parameter(torch.randn(2, 2))],
                        'lr': 0.01
                    }
                }
            }
        }
    }
    
    optimizer = MultiPartOptimizer(optimizer_cfgs)
    
    # Summarize should return a dictionary
    summary = optimizer.summarize()
    assert isinstance(summary, dict)
    assert 'part1' in summary
    assert 'part2' in summary
