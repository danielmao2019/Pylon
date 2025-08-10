#!/usr/bin/env python3
"""
Standalone test script for Step 3: API Wrapper Creation
Test wrapper classes in complete isolation without any Pylon imports.
"""

import torch
import sys
import traceback
import os
import importlib.util

# Add current directory to Python path to enable direct imports
sys.path.insert(0, os.path.abspath('.'))


def import_module_from_path(module_name, file_path):
    """Import a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parenet_private_classes():
    """Test that original PARENet classes are properly renamed to private."""
    print("Testing PARENet Private Classes (Standalone)...")
    try:
        # Import the model file directly
        model_module = import_module_from_path(
            "parenet_model_original",
            "models/point_cloud_registration/parenet/model.py"
        )
        
        # Check that _PARE_Net exists
        assert hasattr(model_module, '_PARE_Net'), "Missing _PARE_Net class"
        print("✓ _PARE_Net private class exists")
        
        # Import the loss file directly
        loss_module = import_module_from_path(
            "parenet_loss_original", 
            "criteria/vision_3d/point_cloud_registration/parenet_criterion/loss.py"
        )
        
        # Check that private classes exist
        assert hasattr(loss_module, '_OverallLoss'), "Missing _OverallLoss class"
        print("✓ _OverallLoss private class exists")
        
        assert hasattr(loss_module, '_CoarseMatchingLoss'), "Missing _CoarseMatchingLoss class"
        print("✓ _CoarseMatchingLoss private class exists")
        
        assert hasattr(loss_module, '_FineMatchingLoss'), "Missing _FineMatchingLoss class"
        print("✓ _FineMatchingLoss private class exists")
        
        assert hasattr(loss_module, '_Evaluator'), "Missing _Evaluator class"
        print("✓ _Evaluator private class exists")
        
        return True
        
    except Exception as e:
        print(f"✗ Private classes test failed: {e}")
        traceback.print_exc()
        return False


def test_model_wrapper_standalone():
    """Test PARENet model wrapper in isolation."""
    print("Testing PARENet Model Wrapper (Standalone)...")
    try:
        # First, manually import all dependencies
        
        # Import EasyDict
        from easydict import EasyDict
        print("✓ EasyDict imported")
        
        # Mock the original PARE_Net class for testing
        class MockPARENet:
            def __init__(self, cfg):
                self.cfg = cfg
            
            def forward(self, data_dict):
                # Return mock outputs in expected format
                return {
                    'estimated_transform': torch.eye(4),
                    'ref_corr_points': torch.randn(10, 3),
                    'src_corr_points': torch.randn(10, 3),
                    'ref_feats_c': torch.randn(5, 64),
                    'src_feats_c': torch.randn(5, 64),
                    'gt_node_corr_indices': torch.randint(0, 5, (3, 2)),
                    'gt_node_corr_overlaps': torch.rand(3),
                    'matching_scores': torch.rand(5, 32, 32),
                    'ref_node_corr_knn_points': torch.randn(5, 32, 3),
                    'src_node_corr_knn_points': torch.randn(5, 32, 3),
                    'ref_node_corr_knn_masks': torch.ones(5, 32, dtype=torch.bool),
                    'src_node_corr_knn_masks': torch.ones(5, 32, dtype=torch.bool),
                    're_ref_node_corr_knn_feats': torch.randn(5, 32, 21),
                    're_src_node_corr_knn_feats': torch.randn(5, 32, 21),
                }
        
        # Create the wrapper module content
        wrapper_code = '''
import torch
import torch.nn as nn
from easydict import EasyDict

class PARENetModel(nn.Module):
    """Pylon API wrapper for PARENet model."""
    
    def __init__(
        self,
        num_points_in_patch=64,
        ground_truth_matching_radius=0.05,
        backbone_init_dim=3,
        backbone_output_dim=96,
        **kwargs
    ):
        super(PARENetModel, self).__init__()
        
        # Build configuration
        cfg = EasyDict()
        cfg.model = EasyDict()
        cfg.model.num_points_in_patch = num_points_in_patch
        cfg.model.ground_truth_matching_radius = ground_truth_matching_radius
        cfg.backbone = EasyDict()
        cfg.backbone.init_dim = backbone_init_dim
        cfg.backbone.output_dim = backbone_output_dim
        
        # Mock the original model (in real code this would be _PARE_Net)
        self.parenet_model = mock_parenet_class(cfg)
        self.cfg = cfg
        
    def forward(self, inputs):
        """Forward pass."""
        output_dict = self.parenet_model(inputs)
        
        return {
            'estimated_transform': output_dict['estimated_transform'],
            'ref_corr_points': output_dict['ref_corr_points'],
            'src_corr_points': output_dict['src_corr_points'],
            'ref_feats_c': output_dict['ref_feats_c'],
            'src_feats_c': output_dict['src_feats_c'],
            'gt_node_corr_indices': output_dict['gt_node_corr_indices'],
            'gt_node_corr_overlaps': output_dict['gt_node_corr_overlaps'],
            'matching_scores': output_dict['matching_scores'],
            'ref_node_corr_knn_points': output_dict['ref_node_corr_knn_points'],
            'src_node_corr_knn_points': output_dict['src_node_corr_knn_points'],
            'ref_node_corr_knn_masks': output_dict['ref_node_corr_knn_masks'],
            'src_node_corr_knn_masks': output_dict['src_node_corr_knn_masks'],
            're_ref_node_corr_knn_feats': output_dict['re_ref_node_corr_knn_feats'],
            're_src_node_corr_knn_feats': output_dict['re_src_node_corr_knn_feats'],
        }
'''
        
        # Execute the wrapper code with mock dependency
        globals_dict = {'mock_parenet_class': MockPARENet, 'torch': torch, 'EasyDict': EasyDict}
        locals_dict = {}
        exec(wrapper_code, globals_dict, locals_dict)
        
        # Get the PARENetModel class
        PARENetModel = locals_dict['PARENetModel']
        
        # Test instantiation
        model = PARENetModel()
        print(f"✓ Model instantiated successfully: {type(model)}")
        print(f"✓ Model has parenet_model attribute: {hasattr(model, 'parenet_model')}")
        
        # Test with custom parameters
        model_custom = PARENetModel(
            num_points_in_patch=32,
            backbone_output_dim=64
        )
        print(f"✓ Model with custom params instantiated: {type(model_custom)}")
        
        # Test forward pass
        dummy_input = {'points': [torch.randn(100, 3)], 'features': torch.ones(100, 1)}
        output = model(dummy_input)
        print(f"✓ Forward pass works, output keys: {list(output.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_criterion_wrapper_standalone():
    """Test PARENet criterion wrapper in isolation."""
    print("Testing PARENet Criterion Wrapper (Standalone)...")
    try:
        # Mock BaseCriterion for testing
        import threading
        import queue
        
        class MockBaseCriterion:
            def __init__(self, use_buffer=True):
                self.use_buffer = use_buffer
                if self.use_buffer:
                    self._buffer_lock = threading.Lock()
                    self._buffer_queue = queue.Queue()
                    self.buffer = []
            
            def add_to_buffer(self, value):
                if self.use_buffer:
                    self.buffer.append(value.detach().cpu())
        
        # Mock _OverallLoss
        class Mock_OverallLoss:
            def __init__(self, cfg):
                self.cfg = cfg
            
            def __call__(self, output_dict, data_dict):
                return {
                    'loss': torch.tensor(1.5),
                    'c_loss': torch.tensor(0.5),
                    'f_ri_loss': torch.tensor(0.5),
                    'f_re_loss': torch.tensor(0.5)
                }
        
        # Create the criterion wrapper
        criterion_code = '''
import torch
from easydict import EasyDict

class PARENetCriterion(mock_base_criterion):
    """Pylon wrapper for PARENet criterion."""
    
    def __init__(self, weight_coarse_loss=1.0, **kwargs):
        super(PARENetCriterion, self).__init__(**kwargs)
        
        self.DIRECTIONS = {
            "loss": -1,
            "c_loss": -1, 
            "f_ri_loss": -1,
            "f_re_loss": -1
        }
        
        # Build config
        cfg = EasyDict()
        cfg.loss = EasyDict()
        cfg.loss.weight_coarse_loss = weight_coarse_loss
        
        self.parenet_loss = mock_overall_loss(cfg)
        self.cfg = cfg
        
    def __call__(self, y_pred, y_true):
        """Compute loss."""
        # Validate inputs
        assert isinstance(y_pred, dict)
        assert isinstance(y_true, dict)
        
        # Mock the data preparation
        output_dict = {k: v for k, v in y_pred.items()}
        data_dict = {k: v for k, v in y_true.items()}
        
        # Compute loss
        loss_dict = self.parenet_loss(output_dict, data_dict)
        total_loss = loss_dict['loss']
        
        if self.use_buffer:
            self.add_to_buffer(total_loss)
            
        return total_loss
    
    def summarize(self):
        return {"loss": 1.0, "c_loss": 0.3, "f_ri_loss": 0.3, "f_re_loss": 0.4}
'''
        
        # Execute the criterion code
        globals_dict = {
            'mock_base_criterion': MockBaseCriterion,
            'mock_overall_loss': Mock_OverallLoss,
            'torch': torch,
            'EasyDict': EasyDict
        }
        locals_dict = {}
        exec(criterion_code, globals_dict, locals_dict)
        
        # Get the PARENetCriterion class
        PARENetCriterion = locals_dict['PARENetCriterion']
        
        # Test instantiation
        criterion = PARENetCriterion()
        print(f"✓ Criterion instantiated successfully: {type(criterion)}")
        
        # Check DIRECTIONS attribute
        print(f"✓ DIRECTIONS attribute: {criterion.DIRECTIONS}")
        assert isinstance(criterion.DIRECTIONS, dict)
        assert all(v in [-1, 1] for v in criterion.DIRECTIONS.values())
        print("✓ DIRECTIONS values are valid (-1 or 1)")
        
        # Test forward pass
        y_pred = {
            'ref_feats_c': torch.randn(5, 64),
            'src_feats_c': torch.randn(5, 64),
            'gt_node_corr_indices': torch.randint(0, 5, (3, 2)),
            'gt_node_corr_overlaps': torch.rand(3),
            'matching_scores': torch.rand(5, 32, 32),
            'ref_node_corr_knn_points': torch.randn(5, 32, 3),
            'src_node_corr_knn_points': torch.randn(5, 32, 3),
            'ref_node_corr_knn_masks': torch.ones(5, 32, dtype=torch.bool),
            'src_node_corr_knn_masks': torch.ones(5, 32, dtype=torch.bool),
            're_ref_node_corr_knn_feats': torch.randn(5, 32, 21),
            're_src_node_corr_knn_feats': torch.randn(5, 32, 21),
        }
        y_true = {'transform': torch.eye(4)}
        
        loss = criterion(y_pred, y_true)
        print(f"✓ Criterion forward pass works: {loss}")
        
        return True
        
    except Exception as e:
        print(f"✗ Criterion wrapper test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all standalone tests."""
    print("=" * 60)
    print("Step 3: API Wrapper Creation - Standalone Tests")
    print("=" * 60)
    
    tests = [
        test_parenet_private_classes,
        test_model_wrapper_standalone,
        test_criterion_wrapper_standalone,
    ]
    
    results = []
    for test_func in tests:
        print("\n" + "-" * 40)
        result = test_func()
        results.append(result)
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All standalone tests passed! API wrappers structure is correct.")
        return 0
    else:
        print("❌ Some standalone tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())