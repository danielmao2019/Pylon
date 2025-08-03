#!/usr/bin/env python3
"""
Test script for Step 2: Import Path Updates
This script tests the import fixes without creating any wrapper classes.
"""

def test_basic_imports():
    """Test basic imports from the updated paths."""
    print("Testing basic imports...")
    
    try:
        # Test main model import
        from models.point_cloud_registration.parenet.model import PARE_Net
        print("✓ Main model import successful")
        
        # Test backbone import  
        from models.point_cloud_registration.parenet.backbone import PAREConvFPN
        print("✓ Backbone import successful")
        
        # Test collator import
        from data.collators.parenet.data import precompute_subsample
        print("✓ Collator import successful")
        
        # Test criteria import
        from criteria.vision_3d.point_cloud_registration.parenet_criterion.loss import CoarseMatchingLoss
        print("✓ Criteria import successful")
        
        print("All basic imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_pareconv_module_imports():
    """Test internal pareconv module imports."""
    print("\nTesting pareconv module imports...")
    
    try:
        # Test ops imports
        from models.point_cloud_registration.parenet.pareconv.modules.ops import grid_subsample, radius_search
        print("✓ Ops module import successful")
        
        # Test registration imports
        from models.point_cloud_registration.parenet.pareconv.modules.registration import HypothesisProposer
        print("✓ Registration module import successful")
        
        # Test layers imports
        from models.point_cloud_registration.parenet.pareconv.modules.layers import ConvBlock
        print("✓ Layers module import successful")
        
        # Test transformer imports
        from models.point_cloud_registration.parenet.pareconv.modules.transformer import RPEConditionalTransformer
        print("✓ Transformer module import successful")
        
        print("All pareconv module imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Pareconv module import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all import tests."""
    print("="*50)
    print("PARENet Step 2: Import Path Updates Test")
    print("="*50)
    
    success = True
    success &= test_basic_imports()
    success &= test_pareconv_module_imports()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All import tests passed! Step 2 completed successfully.")
    else:
        print("❌ Some import tests failed. Check errors above.")
    print("="*50)
    
    return success

if __name__ == "__main__":
    main()