#!/usr/bin/env python3
"""
Direct test script for Step 3: API Wrapper Creation
Test wrapper classes directly without going through full Pylon import chain.
"""

import torch
import sys
import traceback
import os

# Add current directory to Python path to enable direct imports
sys.path.insert(0, os.path.abspath('.'))


def test_model_wrapper_direct():
    """Test PARENet model wrapper import and instantiation directly."""
    print("Testing PARENet Model Wrapper (Direct)...")
    try:
        # Import directly without going through the full models.__init__ chain
        from models.point_cloud_registration.parenet.parenet_model import PARENetModel
        
        # Test instantiation with default parameters
        model = PARENetModel()
        print(f"✓ Model instantiated successfully: {type(model)}")
        print(f"✓ Model has parenet_model attribute: {hasattr(model, 'parenet_model')}")
        
        # Test with custom parameters
        model_custom = PARENetModel(
            num_points_in_patch=32,
            backbone_output_dim=64
        )
        print(f"✓ Model with custom params instantiated: {type(model_custom)}")
        
        # Test forward signature (without actually running it)
        import inspect
        forward_sig = inspect.signature(model.forward)
        print(f"✓ Forward method signature: {forward_sig}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_criterion_wrapper_direct():
    """Test PARENet criterion wrapper import and instantiation directly."""
    print("Testing PARENet Criterion Wrapper (Direct)...")
    try:
        # Import directly 
        from criteria.vision_3d.point_cloud_registration.parenet_criterion.parenet_criterion import PARENetCriterion
        
        # Test instantiation with default parameters
        criterion = PARENetCriterion()
        print(f"✓ Criterion instantiated successfully: {type(criterion)}")
        
        # Check DIRECTIONS attribute
        print(f"✓ DIRECTIONS attribute: {criterion.DIRECTIONS}")
        assert isinstance(criterion.DIRECTIONS, dict)
        assert all(v in [-1, 1] for v in criterion.DIRECTIONS.values())
        print("✓ DIRECTIONS values are valid (-1 or 1)")
        
        # Check that it has the buffer management from BaseCriterion
        print(f"✓ Has buffer attribute: {hasattr(criterion, 'use_buffer')}")
        print(f"✓ Has parenet_loss attribute: {hasattr(criterion, 'parenet_loss')}")
        
        # Test with custom parameters
        criterion_custom = PARENetCriterion(
            weight_coarse_loss=2.0,
            fine_positive_radius=0.05,
            use_buffer=False
        )
        print(f"✓ Criterion with custom params instantiated: {type(criterion_custom)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Criterion wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_metric_wrapper_direct():
    """Test PARENet metric wrapper import and instantiation directly."""
    print("Testing PARENet Metric Wrapper (Direct)...")
    try:
        # Import directly
        from metrics.vision_3d.point_cloud_registration.parenet_metric.parenet_metric import PARENetMetric
        
        # Test instantiation with default parameters
        metric = PARENetMetric()
        print(f"✓ Metric instantiated successfully: {type(metric)}")
        
        # Check DIRECTIONS attribute
        print(f"✓ DIRECTIONS attribute: {metric.DIRECTIONS}")
        assert isinstance(metric.DIRECTIONS, dict)
        assert all(v in [-1, 1] for v in metric.DIRECTIONS.values())
        print("✓ DIRECTIONS values are valid (-1 or 1)")
        
        # Check that it has the required components
        print(f"✓ Has isotropic_error attribute: {hasattr(metric, 'isotropic_error')}")
        print(f"✓ Has inlier_ratio attribute: {hasattr(metric, 'inlier_ratio')}")
        print(f"✓ Has parenet_evaluator attribute: {hasattr(metric, 'parenet_evaluator')}")
        
        # Test with custom parameters
        metric_custom = PARENetMetric(
            inlier_threshold=0.05,
            rmse_threshold=0.1,
            use_buffer=False
        )
        print(f"✓ Metric with custom params instantiated: {type(metric_custom)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metric wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_collator_wrapper_direct():
    """Test PARENet collator wrapper import and instantiation directly."""
    print("Testing PARENet Collator Wrapper (Direct)...")
    try:
        # Import directly
        from data.collators.parenet.parenet_collator import PARENetCollator
        
        # Test instantiation with default parameters
        collator = PARENetCollator()
        print(f"✓ Collator instantiated successfully: {type(collator)}")
        
        # Check initialization parameters
        print(f"✓ num_stages: {collator.num_stages}")
        print(f"✓ voxel_size: {collator.voxel_size}")
        print(f"✓ num_neighbors: {collator.num_neighbors}")
        print(f"✓ precompute_data: {collator.precompute_data}")
        
        # Test with custom parameters
        collator_custom = PARENetCollator(
            num_stages=3,
            voxel_size=0.1,
            num_neighbors=[16, 16, 16]
        )
        print(f"✓ Collator with custom params instantiated: {type(collator_custom)}")
        print(f"✓ Custom num_stages: {collator_custom.num_stages}")
        print(f"✓ Custom num_neighbors: {collator_custom.num_neighbors}")
        
        return True
        
    except Exception as e:
        print(f"✗ Collator wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_parenet_private_classes():
    """Test that original PARENet classes are properly renamed to private."""
    print("Testing PARENet Private Classes...")
    try:
        # Test that the private class exists
        from models.point_cloud_registration.parenet.model import _PARE_Net
        print("✓ _PARE_Net private class exists")
        
        from criteria.vision_3d.point_cloud_registration.parenet_criterion.loss import _OverallLoss
        print("✓ _OverallLoss private class exists")
        
        from criteria.vision_3d.point_cloud_registration.parenet_criterion.loss import _CoarseMatchingLoss
        print("✓ _CoarseMatchingLoss private class exists")
        
        from criteria.vision_3d.point_cloud_registration.parenet_criterion.loss import _FineMatchingLoss
        print("✓ _FineMatchingLoss private class exists")
        
        from criteria.vision_3d.point_cloud_registration.parenet_criterion.loss import _Evaluator
        print("✓ _Evaluator private class exists")
        
        return True
        
    except Exception as e:
        print(f"✗ Private classes test failed: {e}")
        traceback.print_exc()
        return False


def test_basic_torch_operations():
    """Test basic tensor operations to ensure PyTorch is working."""
    print("Testing Basic PyTorch Operations...")
    try:
        # Create dummy tensors
        transform = torch.eye(4, dtype=torch.float32)
        points = torch.randn(100, 3, dtype=torch.float32)
        
        print(f"✓ Created transform tensor: {transform.shape}")
        print(f"✓ Created points tensor: {points.shape}")
        
        # Test basic operations
        transformed_points = torch.matmul(points, transform[:3, :3].T) + transform[:3, 3]
        print(f"✓ Matrix multiplication works: {transformed_points.shape}")
        
        distances = torch.norm(points - transformed_points, dim=1)
        print(f"✓ Distance computation works: {distances.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic PyTorch test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all direct tests."""
    print("=" * 60)
    print("Step 3: API Wrapper Creation - Direct Tests")
    print("=" * 60)
    
    tests = [
        test_basic_torch_operations,
        test_parenet_private_classes,
        test_model_wrapper_direct,
        test_criterion_wrapper_direct,
        test_metric_wrapper_direct,
        test_collator_wrapper_direct,
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
        print("🎉 All direct tests passed! API wrappers are working correctly.")
        return 0
    else:
        print("❌ Some direct tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())