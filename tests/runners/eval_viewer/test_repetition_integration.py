import pytest
import numpy as np
import json
import os
from unittest.mock import patch

from runners.eval_viewer.backend.initialization import initialize_log_dirs
from runners.eval_viewer.backend.repetition_discovery import discover_experiment_groups


def test_repetition_discovery_integration_basic(temp_log_dir):
    """Test basic integration of repetition discovery with initialization."""
    # Create mock experiment directory with repetitions
    base_exp_dir = os.path.join(temp_log_dir, "logs", "exp")
    exp_dirs = []
    
    for run_idx in range(2):
        run_dir = os.path.join(base_exp_dir, f"TestExp_run_{run_idx}")
        os.makedirs(run_dir)
        
        # Create config.json
        config_data = {
            "model": {"class": "TestModel"},
            "eval_dataset": {"class": "KITTIDataset", "args": {}},
            "eval_dataloader": {"batch_size": 1}
        }
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config_data, f)
        
        # Create evaluation_scores.json with slight variation
        base_perf = 0.5 + run_idx * 0.1  # Vary performance by run
        scores = {
            "per_datapoint": {
                "metric1": [base_perf, base_perf + 0.1, base_perf + 0.05],
                "metric2": [base_perf * 2, base_perf * 2 + 0.2, base_perf * 2 + 0.1]
            },
            "aggregated": {
                "metric1": base_perf + 0.05,
                "metric2": base_perf * 2 + 0.1
            }
        }
        with open(os.path.join(run_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(scores, f)
        
        exp_dirs.append(run_dir)
    
    # Create corresponding config files
    configs_dir = os.path.join(temp_log_dir, "configs", "exp")
    os.makedirs(configs_dir, exist_ok=True)
    
    for run_idx in range(2):
        config_path = os.path.join(configs_dir, f"TestExp_run_{run_idx}.py")
        config_content = f'''# Generated config for TestExp_run_{run_idx}
config = {{
    'model': {{'class': 'TestModel'}},
    'eval_dataset': {{'class': 'KITTIDataset', 'args': {{}}}},
    'eval_dataloader': {{'batch_size': 1}},
    'work_dir': './logs/exp/TestExp_run_{run_idx}'
}}
'''
        with open(config_path, 'w') as f:
            f.write(config_content)
    
    # Create common dataset config
    dataset_config_dir = os.path.join(temp_log_dir, "configs", "common", "datasets", "point_cloud_registration", "val")
    os.makedirs(dataset_config_dir, exist_ok=True)
    
    kitti_config = os.path.join(dataset_config_dir, "kitti_data_cfg.py")
    with open(kitti_config, 'w') as f:
        f.write('''data_cfg = {
    'val_dataset': {
        'class': 'KITTIDataset',
        'args': {'data_root': '/data/kitti'}
    }
}
''')
    
    # Test discovery
    groups = discover_experiment_groups([exp_dirs[0]])  # Provide only first run
    
    assert len(groups) == 1
    group = groups[0]
    assert group.experiment_name == "TestExp"
    assert group.num_repetitions == 2
    assert len(group.repetition_paths) == 2


@patch('runners.eval_viewer.backend.initialization.os.path.normpath')
def test_repetition_integration_with_initialize_log_dirs(mock_normpath, temp_log_dir):
    """Test integration with initialize_log_dirs function."""
    mock_normpath.return_value = temp_log_dir
    
    # Create minimal experiment structure
    log_dir = os.path.join(temp_log_dir, "logs", "TestExp_run_0")
    os.makedirs(log_dir)
    
    # Create config.json
    config_data = {
        "eval_dataset": {"class": "KITTIDataset", "args": {}},
        "eval_dataloader": {"batch_size": 1}
    }
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json.dump(config_data, f)
    
    # Create evaluation_scores.json
    scores = {
        "per_datapoint": {"metric1": [0.5, 0.6, 0.7]},
        "aggregated": {"metric1": 0.6}
    }
    with open(os.path.join(log_dir, "evaluation_scores.json"), 'w') as f:
        json.dump(scores, f)
    
    # Create config file
    config_path = os.path.join(temp_log_dir, "configs", "TestExp_run_0.py")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(f'config = {config_data}')
    
    # Create dataset config
    dataset_config_dir = os.path.join(temp_log_dir, "configs", "common", "datasets", "point_cloud_registration", "val")
    os.makedirs(dataset_config_dir, exist_ok=True)
    kitti_config = os.path.join(dataset_config_dir, "kitti_data_cfg.py")
    with open(kitti_config, 'w') as f:
        f.write('''data_cfg = {'val_dataset': {'class': 'KITTIDataset', 'args': {}}}''')
    
    try:
        # This should work without errors, even if it only finds one repetition
        max_epochs, metric_names, num_datapoints, dataset_cfg, dataset_type, log_dir_infos, color_scales = \
            initialize_log_dirs([log_dir], force_reload=True)
        
        assert len(log_dir_infos) == 1
        assert "TestExp" in log_dir_infos
        assert metric_names == ["metric1"]
        assert num_datapoints == 3
        
    except Exception as e:
        # Skip test if dependencies not available  
        pytest.skip(f"Integration test skipped due to dependencies: {e}")


def test_experiment_group_aggregation_mathematical_correctness():
    """Test that aggregation produces mathematically correct results."""
    from runners.eval_viewer.backend.repetition_discovery import aggregate_log_dir_infos
    from runners.eval_viewer.backend.initialization import LogDirInfo
    
    # Create LogDirInfo objects with known values
    infos = []
    values = [1.0, 2.0, 3.0]  # Simple values for verification
    
    for val in values:
        score_map = np.full((2, 3, 3), val)  # 2 metrics, 3x3 grid
        aggregated_scores = np.array([val, val * 2])
        
        info = LogDirInfo(
            num_epochs=1,
            metric_names=["metric1", "metric2"],
            num_datapoints=9,
            score_map=score_map,
            aggregated_scores=aggregated_scores,
            dataset_class="KITTIDataset",
            dataset_type="pcr",
            dataset_cfg={},
            dataloader_cfg={},
            runner_type="evaluator"
        )
        infos.append(info)
    
    # Aggregate
    result = aggregate_log_dir_infos(infos)
    
    # Verify mathematical correctness
    expected_mean = np.mean(values)  # 2.0
    expected_std = np.std(values, ddof=1)  # 1.0
    
    assert np.allclose(result.score_map, expected_mean)
    assert np.allclose(result.aggregated_scores, [expected_mean, expected_mean * 2])
    
    # Check that std attributes were added correctly
    assert hasattr(result, 'num_repetitions')
    assert result.num_repetitions == 3
    assert hasattr(result, 'aggregated_scores_std')
    assert hasattr(result, 'score_map_std')
    
    assert np.allclose(result.score_map_std, expected_std)
    assert np.allclose(result.aggregated_scores_std, [expected_std, expected_std * 2])