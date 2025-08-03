#!/usr/bin/env python3
"""
Basic test script for Step 3: API Wrapper Creation
Test if all wrapper classes can be imported and instantiated correctly.
"""

import torch
import sys
import traceback


def test_model_wrapper():
    """Test PARENet model wrapper import and instantiation."""
    print("Testing PARENet Model Wrapper...")
    try:
        from models.point_cloud_registration.parenet.parenet_model import PARENetModel
        
        # Test instantiation with default parameters
        model = PARENetModel()
        print(f"✓ Model instantiated successfully: {type(model)}")
        
        # Test with custom parameters
        model_custom = PARENetModel(
            num_points_in_patch=32,
            backbone_output_dim=64
        )
        print(f"✓ Model with custom params instantiated: {type(model_custom)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_criterion_wrapper():
    """Test PARENet criterion wrapper import and instantiation."""
    print("Testing PARENet Criterion Wrapper...")
    try:
        from criteria.vision_3d.point_cloud_registration.parenet_criterion.parenet_criterion import PARENetCriterion
        
        # Test instantiation with default parameters
        criterion = PARENetCriterion()
        print(f"✓ Criterion instantiated successfully: {type(criterion)}")
        
        # Check DIRECTIONS attribute
        print(f"✓ DIRECTIONS attribute: {criterion.DIRECTIONS}")
        assert isinstance(criterion.DIRECTIONS, dict)
        assert all(v in [-1, 1] for v in criterion.DIRECTIONS.values())
        
        # Test with custom parameters
        criterion_custom = PARENetCriterion(
            weight_coarse_loss=2.0,
            fine_positive_radius=0.05
        )
        print(f"✓ Criterion with custom params instantiated: {type(criterion_custom)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Criterion wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_metric_wrapper():
    """Test PARENet metric wrapper import and instantiation."""
    print("Testing PARENet Metric Wrapper...")
    try:
        from metrics.vision_3d.point_cloud_registration.parenet_metric.parenet_metric import PARENetMetric
        
        # Test instantiation with default parameters
        metric = PARENetMetric()
        print(f"✓ Metric instantiated successfully: {type(metric)}")
        
        # Check DIRECTIONS attribute
        print(f"✓ DIRECTIONS attribute: {metric.DIRECTIONS}")
        assert isinstance(metric.DIRECTIONS, dict)
        assert all(v in [-1, 1] for v in metric.DIRECTIONS.values())
        
        # Test with custom parameters
        metric_custom = PARENetMetric(
            inlier_threshold=0.05,
            rmse_threshold=0.1
        )
        print(f"✓ Metric with custom params instantiated: {type(metric_custom)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metric wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_collator_wrapper():
    """Test PARENet collator wrapper import and instantiation."""
    print("Testing PARENet Collator Wrapper...")
    try:
        from data.collators.parenet.parenet_collator import PARENetCollator
        
        # Test instantiation with default parameters
        collator = PARENetCollator()
        print(f"✓ Collator instantiated successfully: {type(collator)}")
        
        # Test with custom parameters
        collator_custom = PARENetCollator(
            num_stages=3,
            voxel_size=0.1,
            num_neighbors=[16, 16, 16]
        )
        print(f"✓ Collator with custom params instantiated: {type(collator_custom)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Collator wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_api_registration():
    """Test that all wrappers are properly registered in __init__.py files."""
    print("Testing API Registration...")
    try:
        # Test model registration
        from models.point_cloud_registration import PARENetModel
        print("✓ PARENetModel registered in models.point_cloud_registration")
        
        # Test criterion registration
        from criteria.vision_3d.point_cloud_registration import PARENetCriterion
        print("✓ PARENetCriterion registered in criteria.vision_3d.point_cloud_registration")
        
        # Test metric registration
        from metrics.vision_3d.point_cloud_registration import PARENetMetric
        print("✓ PARENetMetric registered in metrics.vision_3d.point_cloud_registration")
        
        # Test collator registration
        from data.collators import PARENetCollator
        print("✓ PARENetCollator registered in data.collators")
        
        return True
        
    except Exception as e:
        print(f"✗ API registration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all basic tests."""
    print("=" * 60)
    print("Step 3: API Wrapper Creation - Basic Tests")
    print("=" * 60)
    
    tests = [
        test_model_wrapper,
        test_criterion_wrapper,
        test_metric_wrapper,
        test_collator_wrapper,
        test_api_registration
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
        print("🎉 All basic tests passed! API wrappers are working correctly.")
        return 0
    else:
        print("❌ Some basic tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())