from typing import Dict, Any, Tuple, List
import os
import sys
import importlib.util
import concurrent.futures
import traceback
from pathlib import Path
import json

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """Load a config module and return its config dict."""
    # Save original sys.path
    original_path = sys.path.copy()

    try:
        # Add project root and parent directories to sys.path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)

        # Add the directory containing the config file
        config_dir = os.path.dirname(filepath)
        sys.path.insert(0, config_dir)

        # Add parent directories of config_dir to handle relative imports
        parent_dir = os.path.dirname(config_dir)
        while parent_dir and parent_dir != '/':
            sys.path.insert(0, parent_dir)
            parent_dir = os.path.dirname(parent_dir)

        spec = importlib.util.spec_from_file_location("config_file", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config
    except Exception as e:
        raise Exception(f"Failed to load config from {filepath}: {str(e)}")
    finally:
        # Restore original sys.path
        sys.path = original_path


def deep_compare_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = "") -> List[str]:
    """Recursively compare two dictionaries and return list of differences."""
    differences = []

    # Check keys
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    for key in keys1 - keys2:
        differences.append(f"{path}.{key}: exists in old config but not in new config")

    for key in keys2 - keys1:
        differences.append(f"{path}.{key}: exists in new config but not in old config")

    # Check values for common keys
    for key in keys1 & keys2:
        current_path = f"{path}.{key}" if path else key
        val1 = dict1[key]
        val2 = dict2[key]

        if type(val1) != type(val2):
            differences.append(f"{current_path}: type mismatch - old: {type(val1).__name__}, new: {type(val2).__name__}")
        elif isinstance(val1, dict):
            differences.extend(deep_compare_dicts(val1, val2, current_path))
        elif isinstance(val1, (list, tuple)):
            if len(val1) != len(val2):
                differences.append(f"{current_path}: length mismatch - old: {len(val1)}, new: {len(val2)}")
            else:
                for i, (item1, item2) in enumerate(zip(val1, val2)):
                    if isinstance(item1, dict) and isinstance(item2, dict):
                        differences.extend(deep_compare_dicts(item1, item2, f"{current_path}[{i}]"))
                    elif item1 != item2:
                        # Special handling for class references
                        if hasattr(item1, '__module__') and hasattr(item1, '__name__') and \
                           hasattr(item2, '__module__') and hasattr(item2, '__name__'):
                            if item1.__module__ != item2.__module__ or item1.__name__ != item2.__name__:
                                differences.append(f"{current_path}[{i}]: class mismatch - old: {item1.__module__}.{item1.__name__}, new: {item2.__module__}.{item2.__name__}")
                        else:
                            differences.append(f"{current_path}[{i}]: value mismatch - old: {item1}, new: {item2}")
        else:
            if val1 != val2:
                # Special handling for class references
                if hasattr(val1, '__module__') and hasattr(val1, '__name__') and \
                   hasattr(val2, '__module__') and hasattr(val2, '__name__'):
                    if val1.__module__ != val2.__module__ or val1.__name__ != val2.__name__:
                        differences.append(f"{current_path}: class mismatch - old: {val1.__module__}.{val1.__name__}, new: {val2.__module__}.{val2.__name__}")
                else:
                    differences.append(f"{current_path}: value mismatch - old: {val1}, new: {val2}")

    return differences


def compare_config_pair(old_path: str, new_path: str) -> Tuple[str, List[str], str]:
    """Compare a pair of config files and return (relative_path, differences, error)."""
    try:
        old_config = load_config_from_file(old_path)
        new_config = load_config_from_file(new_path)

        differences = deep_compare_dicts(old_config, new_config)

        return (new_path, differences, None)
    except Exception as e:
        return (new_path, [], f"Error: {str(e)}\n{traceback.format_exc()}")


def find_config_pairs(old_dir: str, new_dir: str) -> List[Tuple[str, str]]:
    """Find all matching config file pairs between old and new directories."""
    assert os.path.isdir(old_dir), f"{old_dir=}"
    assert os.path.isdir(new_dir), f"{new_dir=}"

    pairs = []

    # Walk through the new directory structure
    for root, dirs, files in os.walk(new_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('gen_') and not ('template' in file):
                new_path = os.path.join(root, file)
                # Construct the corresponding old path
                relative_path = os.path.relpath(new_path, new_dir)
                old_path = os.path.join(old_dir, relative_path)

                if os.path.exists(old_path):
                    pairs.append((old_path, new_path))
                else:
                    print(f"Warning: No matching old file for {new_path}")

    return pairs


def main():
    old_dir = "/home/daniel/repos/Pylon-cd-configs/cd_cfg_backup"
    new_dir = "/home/daniel/repos/Pylon-cd-configs/configs/benchmarks/change_detection"

    print(f"Finding config file pairs...")
    pairs = find_config_pairs(old_dir, new_dir)
    print(f"Found {len(pairs)} config file pairs to compare")

    if not pairs:
        print("No config pairs found!")
        return

    print("\nComparing configs using concurrent futures...")

    # Use concurrent futures to speed up comparison
    all_differences = []
    errors = []

    # Process sequentially to avoid import contamination
    for old_path, new_path in pairs:
        try:
            new_path_result, differences, error = compare_config_pair(old_path, new_path)

            if error:
                errors.append((new_path_result, error))
            elif differences:
                all_differences.append((new_path_result, differences))

        except Exception as exc:
            errors.append((new_path, f"Exception: {str(exc)}"))

    # Report results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    if not all_differences and not errors:
        print("\n✅ SUCCESS: All config pairs are EXACTLY the same!")
        print(f"   Compared {len(pairs)} config file pairs")
    else:
        if all_differences:
            print(f"\n❌ DIFFERENCES FOUND: {len(all_differences)} config pairs have differences")
            print("\nDetailed differences:")
            for config_path, differences in all_differences:
                print(f"\n{config_path}:")
                for diff in differences:
                    print(f"  - {diff}")

        if errors:
            print(f"\n❌ ERRORS: {len(errors)} config pairs could not be compared")
            print("\nDetailed errors:")
            for config_path, error in errors:
                print(f"\n{config_path}:")
                print(f"  {error}")

    # Summary
    print(f"\n" + "="*80)
    print(f"Total pairs checked: {len(pairs)}")
    print(f"Identical pairs: {len(pairs) - len(all_differences) - len(errors)}")
    print(f"Different pairs: {len(all_differences)}")
    print(f"Error pairs: {len(errors)}")
    print("="*80)


if __name__ == "__main__":
    main()
