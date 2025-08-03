#!/usr/bin/env python3
"""
Direct test for PARENet import path updates without full package imports.
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/daniel/repos/Pylon-parenet')

def test_direct_parenet_imports():
    """Test PARENet imports directly."""
    print("Testing direct PARENet imports...")
    
    try:
        # Test pareconv extension import first
        import models.point_cloud_registration.parenet.pareconv.modules.ops.grid_subsample
        print("✓ Grid subsample import successful")
        
        # Test pareconv ops import
        from models.point_cloud_registration.parenet.pareconv.modules.ops import grid_subsample
        print("✓ Ops module import successful")
        
        # Test backbone import directly
        import models.point_cloud_registration.parenet.backbone
        print("✓ Backbone module import successful")
        
        # Test data collator import directly
        import data.collators.parenet.data
        print("✓ Data collator import successful")
        
        # Test criterion import directly
        import criteria.vision_3d.point_cloud_registration.parenet_criterion.loss
        print("✓ Criterion import successful")
        
        print("All direct PARENet imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Direct import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pareconv_internal_imports():
    """Test internal pareconv imports."""
    print("\nTesting internal pareconv imports...")
    
    try:
        # Test registration module
        import models.point_cloud_registration.parenet.pareconv.modules.registration.registration
        print("✓ Registration registration import successful")
        
        # Test transformer module
        import models.point_cloud_registration.parenet.pareconv.modules.transformer.conditional_transformer
        print("✓ Transformer conditional import successful")
        
        # Test layers module
        import models.point_cloud_registration.parenet.pareconv.modules.layers.conv_block
        print("✓ Layers conv_block import successful")
        
        print("All internal pareconv imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Internal pareconv import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run direct import tests."""
    print("="*50)
    print("PARENet Step 2: Direct Import Path Test")
    print("="*50)
    
    success = True
    success &= test_direct_parenet_imports()
    success &= test_pareconv_internal_imports()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All direct import tests passed! Step 2 import paths working.")
    else:
        print("❌ Some direct import tests failed. Need to fix remaining issues.")
    print("="*50)
    
    return success

if __name__ == "__main__":
    main()