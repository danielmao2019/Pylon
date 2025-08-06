#!/usr/bin/env python3
"""
Summary test for Step 3: API Wrapper Creation
Validate all the wrapper implementations and verify completeness.
"""

import os
import ast
import re
from typing import Dict, List, Tuple


def analyze_file_structure() -> Dict[str, str]:
    """Analyze the structure of files we've created."""
    files_created = {}
    
    # Check model wrapper
    model_file = "models/point_cloud_registration/parenet/parenet_model.py"
    if os.path.exists(model_file):
        files_created["Model Wrapper"] = model_file
    
    # Check criterion wrapper
    criterion_file = "criteria/vision_3d/point_cloud_registration/parenet_criterion/parenet_criterion.py"
    if os.path.exists(criterion_file):
        files_created["Criterion Wrapper"] = criterion_file
    
    # Check metric wrapper
    metric_file = "metrics/vision_3d/point_cloud_registration/parenet_metric/parenet_metric.py"
    if os.path.exists(metric_file):
        files_created["Metric Wrapper"] = metric_file
    
    # Check collator wrapper
    collator_file = "data/collators/parenet/parenet_collator.py"
    if os.path.exists(collator_file):
        files_created["Collator Wrapper"] = collator_file
    
    return files_created


def check_class_definitions(file_path: str) -> List[str]:
    """Extract class definitions from a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return classes
    except:
        return []


def check_directions_attribute(file_path: str) -> bool:
    """Check if a file defines DIRECTIONS attribute."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return 'DIRECTIONS' in content
    except:
        return False


def check_init_file_registration(init_file: str, class_name: str) -> bool:
    """Check if a class is properly registered in __init__.py file."""
    try:
        with open(init_file, 'r') as f:
            content = f.read()
        return class_name in content
    except:
        return False


def check_private_class_renaming() -> Dict[str, bool]:
    """Check if original classes were properly renamed to private."""
    results = {}
    
    # Check model file
    model_file = "models/point_cloud_registration/parenet/model.py"
    try:
        with open(model_file, 'r') as f:
            content = f.read()
        results["_PARE_Net"] = 'class _PARE_Net' in content and 'super(_PARE_Net' in content
    except:
        results["_PARE_Net"] = False
    
    # Check loss file
    loss_file = "criteria/vision_3d/point_cloud_registration/parenet_criterion/loss.py"
    try:
        with open(loss_file, 'r') as f:
            content = f.read()
        results["_OverallLoss"] = 'class _OverallLoss' in content
        results["_CoarseMatchingLoss"] = 'class _CoarseMatchingLoss' in content
        results["_FineMatchingLoss"] = 'class _FineMatchingLoss' in content
        results["_Evaluator"] = 'class _Evaluator' in content
    except:
        results.update({
            "_OverallLoss": False,
            "_CoarseMatchingLoss": False,
            "_FineMatchingLoss": False,
            "_Evaluator": False
        })
    
    return results


def check_wrapper_api_compliance() -> Dict[str, Dict[str, bool]]:
    """Check if wrapper classes follow Pylon API conventions."""
    results = {}
    
    # Model wrapper checks
    model_file = "models/point_cloud_registration/parenet/parenet_model.py"
    if os.path.exists(model_file):
        with open(model_file, 'r') as f:
            content = f.read()
        
        results["PARENetModel"] = {
            "inherits_nn_Module": "nn.Module" in content,
            "has_forward_method": "def forward(self, inputs:" in content,
            "returns_dict": "Dict[str, torch.Tensor]" in content,
            "has_build_from_config_params": "__init__(" in content and "**kwargs" in content
        }
    
    # Criterion wrapper checks
    criterion_file = "criteria/vision_3d/point_cloud_registration/parenet_criterion/parenet_criterion.py"
    if os.path.exists(criterion_file):
        with open(criterion_file, 'r') as f:
            content = f.read()
        
        results["PARENetCriterion"] = {
            "inherits_BaseCriterion": "BaseCriterion" in content,
            "has_call_method": "def __call__(" in content,
            "has_directions": "self.DIRECTIONS" in content,
            "has_summarize_method": "def summarize(" in content
        }
    
    # Metric wrapper checks
    metric_file = "metrics/vision_3d/point_cloud_registration/parenet_metric/parenet_metric.py"
    if os.path.exists(metric_file):
        with open(metric_file, 'r') as f:
            content = f.read()
        
        results["PARENetMetric"] = {
            "inherits_BaseMetric": "BaseMetric" in content,
            "has_call_method": "def __call__(" in content,
            "has_directions": "self.DIRECTIONS" in content,
            "has_summarize_method": "def summarize(" in content
        }
    
    # Collator wrapper checks
    collator_file = "data/collators/parenet/parenet_collator.py"
    if os.path.exists(collator_file):
        with open(collator_file, 'r') as f:
            content = f.read()
        
        results["PARENetCollator"] = {
            "inherits_BaseCollator": "BaseCollator" in content,
            "has_call_method": "def __call__(" in content,
            "handles_datapoints": "datapoints: List[Dict" in content,
            "has_create_dataloader": "def create_dataloader(" in content
        }
    
    return results


def main():
    """Run comprehensive analysis of Step 3 implementation."""
    print("=" * 70)
    print("Step 3: API Wrapper Creation - Implementation Summary")
    print("=" * 70)
    
    # 1. Check file structure
    print("\n1. FILE STRUCTURE ANALYSIS")
    print("-" * 40)
    files_created = analyze_file_structure()
    for component, file_path in files_created.items():
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        print(f"✓ {component:20} | {file_path} ({file_size} bytes)")
    
    print(f"\nTotal files created: {len(files_created)}/4")
    
    # 2. Check class definitions
    print("\n2. CLASS DEFINITIONS ANALYSIS")
    print("-" * 40)
    for component, file_path in files_created.items():
        classes = check_class_definitions(file_path)
        print(f"{component:20} | Classes: {', '.join(classes)}")
    
    # 3. Check private class renaming
    print("\n3. PRIVATE CLASS RENAMING")
    print("-" * 40)
    private_classes = check_private_class_renaming()
    for class_name, renamed in private_classes.items():
        status = "✓" if renamed else "✗"
        print(f"{status} {class_name:20} | Properly renamed: {renamed}")
    
    # 4. Check DIRECTIONS attributes
    print("\n4. DIRECTIONS ATTRIBUTE CHECK")
    print("-" * 40)
    criterion_has_directions = check_directions_attribute(
        "criteria/vision_3d/point_cloud_registration/parenet_criterion/parenet_criterion.py"
    )
    metric_has_directions = check_directions_attribute(
        "metrics/vision_3d/point_cloud_registration/parenet_metric/parenet_metric.py"
    )
    
    print(f"✓ PARENetCriterion DIRECTIONS: {criterion_has_directions}")
    print(f"✓ PARENetMetric DIRECTIONS: {metric_has_directions}")
    
    # 5. Check API registration
    print("\n5. API REGISTRATION CHECK")
    print("-" * 40)
    registrations = [
        ("PARENetModel", "models/point_cloud_registration/__init__.py"),
        ("PARENetCriterion", "criteria/vision_3d/point_cloud_registration/__init__.py"),
        ("PARENetMetric", "metrics/vision_3d/point_cloud_registration/__init__.py"),
        ("PARENetCollator", "data/collators/__init__.py")
    ]
    
    all_registered = True
    for class_name, init_file in registrations:
        registered = check_init_file_registration(init_file, class_name)
        status = "✓" if registered else "✗"
        print(f"{status} {class_name:20} | Registered in {init_file}")
        if not registered:
            all_registered = False
    
    # 6. Check API compliance
    print("\n6. PYLON API COMPLIANCE CHECK")
    print("-" * 40)
    api_compliance = check_wrapper_api_compliance()
    
    for class_name, checks in api_compliance.items():
        print(f"\n{class_name}:")
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("IMPLEMENTATION SUMMARY")
    print("=" * 70)
    
    total_checks = 0
    passed_checks = 0
    
    # Count file creation
    total_checks += 4
    passed_checks += len(files_created)
    
    # Count private class renaming
    total_checks += len(private_classes)
    passed_checks += sum(private_classes.values())
    
    # Count DIRECTIONS
    total_checks += 2
    passed_checks += criterion_has_directions + metric_has_directions
    
    # Count registrations
    total_checks += len(registrations)
    passed_checks += sum(check_init_file_registration(init_file, class_name) 
                        for class_name, init_file in registrations)
    
    # Count API compliance
    for checks in api_compliance.values():
        total_checks += len(checks)
        passed_checks += sum(checks.values())
    
    print(f"Overall Progress: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")
    
    if passed_checks == total_checks:
        print("\n🎉 STEP 3 COMPLETED SUCCESSFULLY!")
        print("✓ All API wrappers created and properly integrated")
        print("✓ Original classes renamed to private")
        print("✓ Pylon API conventions followed")
        print("✓ All components registered in __init__.py files")
        print("\nNext: Proceed to testing and validation")
        return 0
    else:
        print(f"\n⚠️  STEP 3 PARTIALLY COMPLETED")
        print(f"✓ {passed_checks} out of {total_checks} implementation checks passed")
        print("❌ Some components may need additional work")
        return 1


if __name__ == "__main__":
    exit(main())