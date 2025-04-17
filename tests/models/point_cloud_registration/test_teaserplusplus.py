import torch
import numpy as np
import pytest
from models.point_cloud_registration.classic.teaserplusplus import TeaserPlusPlus


def test_teaserplusplus_forward_pass():
    """Test that TeaserPlusPlus can perform forward pass for 10 iterations."""
    # Create model
    model = TeaserPlusPlus()
    
    # Run for 10 iterations
    for i in range(10):
        # Create dummy inputs
        batch_size = 2
        num_source_points = 100
        num_target_points = 120
        
        # Create random point clouds
        src_pc = torch.rand(batch_size, num_source_points, 3)
        tgt_pc = torch.rand(batch_size, num_target_points, 3)
        
        # Create input dictionary
        inputs = {
            'src_pc': {'pos': src_pc},
            'tgt_pc': {'pos': tgt_pc}
        }
        
        # Run forward pass
        output = model(inputs)
        
        # Check output shape
        assert output.shape == (batch_size, 4, 4), f"Iteration {i}: Expected output shape (batch_size, 4, 4), got {output.shape}"
        
        print(f"Iteration {i+1}/10 completed successfully")
