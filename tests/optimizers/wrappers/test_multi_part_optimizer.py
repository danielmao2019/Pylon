from typing import Dict, Any
import torch
import torch.nn as nn
import pytest
from optimizers.wrappers.multi_part_optimizer import MultiPartOptimizer
from optimizers.single_task_optimizer import SingleTaskOptimizer


def test_multi_part_optimizer_initialization(multi_part_optimizer_configs):
    """Test that MultiPartOptimizer correctly initializes with optimizer configs."""
    optimizer = MultiPartOptimizer(multi_part_optimizer_configs)

    assert hasattr(optimizer, 'optimizers')
    assert 'encoder' in optimizer.optimizers
    assert 'decoder' in optimizer.optimizers
    assert isinstance(optimizer.optimizers['encoder'], SingleTaskOptimizer)
    assert isinstance(optimizer.optimizers['decoder'], SingleTaskOptimizer)


def test_multi_part_optimizer_state_dict_save_load(simple_model, multi_part_optimizer_configs):
    """Test that MultiPartOptimizer correctly saves and loads state dicts."""
    # Create MultiPartOptimizer
    optimizer = MultiPartOptimizer(multi_part_optimizer_configs)

    # Run a few optimization steps to build up state
    for _ in range(3):
        for opt in optimizer.optimizers.values():
            opt.zero_grad()
        x = torch.randn(4, 10)
        y = simple_model(x)
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
    new_optimizer = MultiPartOptimizer(multi_part_optimizer_configs)

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

        for i, (orig, new) in enumerate(zip(orig_pg, new_pg, strict=True)):
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


def test_multi_part_optimizer_reset_buffer(basic_optimizer_configs):
    """Test that reset_buffer works for all optimizers."""
    optimizer = MultiPartOptimizer(basic_optimizer_configs)

    # Reset buffers should not raise an error
    optimizer.reset_buffer()

    # Verify buffers are reset
    for opt in optimizer.optimizers.values():
        assert hasattr(opt, 'buffer')
        assert opt.buffer == []


def test_multi_part_optimizer_summarize(basic_optimizer_configs):
    """Test that summarize works correctly."""
    optimizer = MultiPartOptimizer(basic_optimizer_configs)

    # Summarize should return a dictionary
    summary = optimizer.summarize()
    assert isinstance(summary, dict)
    assert 'part1' in summary
    assert 'part2' in summary
