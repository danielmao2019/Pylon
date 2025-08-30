import pytest
import numpy as np
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

from runners.viewers.eval_viewer.backend.repetition_discovery import (
    discover_experiment_groups,
    aggregate_log_dir_infos,
    _extract_base_path,
    _discover_repetitions,
    ExperimentGroup
)
from runners.viewers.eval_viewer.backend.initialization import LogDirInfo


# ==========================================
# Tests for _extract_base_path
# ==========================================

def test_extract_base_path_with_run_suffix():
    """Test _extract_base_path with _run_x suffix."""
    base_path, run_number = _extract_base_path("/logs/experiment/ICP_run_0")
    assert base_path == "/logs/experiment/ICP"
    assert run_number == 0
    
    base_path, run_number = _extract_base_path("/logs/experiment/Model_A_run_5")
    assert base_path == "/logs/experiment/Model_A"
    assert run_number == 5


def test_extract_base_path_without_run_suffix():
    """Test _extract_base_path without _run_x suffix."""
    base_path, run_number = _extract_base_path("/logs/experiment/ICP")
    assert base_path == "/logs/experiment/ICP"
    assert run_number is None
    
    base_path, run_number = _extract_base_path("/logs/experiment/Model_A")
    assert base_path == "/logs/experiment/Model_A"
    assert run_number is None


def test_extract_base_path_edge_cases():
    """Test _extract_base_path edge cases."""
    # Multiple underscores
    base_path, run_number = _extract_base_path("/logs/my_exp_model_run_2")
    assert base_path == "/logs/my_exp_model"
    assert run_number == 2
    
    # Path normalization
    base_path, run_number = _extract_base_path("./logs/experiment/ICP_run_1/")
    assert base_path == "logs/experiment/ICP"
    assert run_number == 1


# ==========================================
# Tests for aggregate_log_dir_infos
# ==========================================

def test_aggregate_log_dir_infos_single_repetition():
    """Test aggregate_log_dir_infos with single repetition returns original."""
    info = LogDirInfo(
        num_epochs=1,
        metric_names=["metric1", "metric2"],
        num_datapoints=4,
        score_map=np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(2, 2, 1),
        aggregated_scores=np.array([1.5, 3.5]),
        dataset_class="TestDataset",
        dataset_type="pcr",
        dataset_cfg={},
        dataloader_cfg={},
        runner_type="evaluator"
    )
    
    result = aggregate_log_dir_infos([info])
    
    # Should return the same object without modification
    assert np.array_equal(result.score_map, info.score_map)
    assert np.array_equal(result.aggregated_scores, info.aggregated_scores)


def test_aggregate_log_dir_infos_multiple_repetitions():
    """Test aggregate_log_dir_infos with multiple repetitions."""
    # Create test data with known values for easy verification
    infos = []
    base_aggregated = np.array([1.0, 2.0])
    base_score_map = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])  # Shape: (2, 2, 1)
    
    for i in range(3):
        # Add variation: multiply by (1 + i*0.1)
        factor = 1.0 + i * 0.1
        aggregated_scores = base_aggregated * factor
        score_map = base_score_map * factor
        
        info = LogDirInfo(
            num_epochs=1,
            metric_names=["metric1", "metric2"],
            num_datapoints=4,
            score_map=score_map,
            aggregated_scores=aggregated_scores,
            dataset_class="TestDataset",
            dataset_type="pcr",
            dataset_cfg={},
            dataloader_cfg={},
            runner_type="evaluator"
        )
        infos.append(info)
    
    result = aggregate_log_dir_infos(infos)
    
    # Verify aggregation metadata
    assert hasattr(result, 'num_repetitions')
    assert result.num_repetitions == 3
    assert hasattr(result, 'score_map_std')
    assert hasattr(result, 'aggregated_scores_std')
    
    # Verify mean calculation
    # factors are [1.0, 1.1, 1.2], mean = 1.1
    expected_mean_aggregated = base_aggregated * 1.1
    expected_mean_score_map = base_score_map * 1.1
    
    assert np.allclose(result.aggregated_scores, expected_mean_aggregated, rtol=1e-10)
    assert np.allclose(result.score_map, expected_mean_score_map, rtol=1e-10)
    
    # Verify std calculation
    # std of [1.0, 1.1, 1.2] is approximately 0.1
    factors = [1.0, 1.1, 1.2]
    expected_std = np.std(factors, ddof=1)
    expected_std_aggregated = base_aggregated * expected_std
    expected_std_score_map = base_score_map * expected_std
    
    assert np.allclose(result.aggregated_scores_std, expected_std_aggregated, rtol=1e-10)
    assert np.allclose(result.score_map_std, expected_std_score_map, rtol=1e-10)


def test_aggregate_log_dir_infos_input_validation():
    """Test aggregate_log_dir_infos input validation."""
    with pytest.raises(AssertionError, match="log_dir_infos must not be None"):
        aggregate_log_dir_infos(None)
    
    with pytest.raises(AssertionError, match="log_dir_infos must be list"):
        aggregate_log_dir_infos("not_a_list")
    
    with pytest.raises(AssertionError, match="log_dir_infos must not be empty"):
        aggregate_log_dir_infos([])


def test_aggregate_log_dir_infos_consistency_validation():
    """Test aggregate_log_dir_infos consistency validation."""
    info1 = LogDirInfo(
        num_epochs=1,
        metric_names=["metric1"],
        num_datapoints=4,
        score_map=np.zeros((1, 2, 2)),
        aggregated_scores=np.array([1.0]),
        dataset_class="TestDataset",
        dataset_type="pcr",
        dataset_cfg={},
        dataloader_cfg={},
        runner_type="evaluator"
    )
    
    info2 = LogDirInfo(
        num_epochs=1,
        metric_names=["different_metric"],  # Different metrics
        num_datapoints=4,
        score_map=np.zeros((1, 2, 2)),
        aggregated_scores=np.array([1.0]),
        dataset_class="TestDataset", 
        dataset_type="pcr",
        dataset_cfg={},
        dataloader_cfg={},
        runner_type="evaluator"
    )
    
    with pytest.raises(AssertionError, match="Inconsistent metric_names"):
        aggregate_log_dir_infos([info1, info2])


# ==========================================
# Tests for discover_experiment_groups
# ==========================================

@pytest.fixture
def mock_repetition_setup(temp_log_dir):
    """Create mock repetition directory structure."""
    # Create ICP experiment with 3 runs
    icp_dirs = []
    for run_idx in range(3):
        run_dir = os.path.join(temp_log_dir, f"ICP_run_{run_idx}")
        os.makedirs(run_dir)
        
        # Create config.json
        config_data = {"model": {"class": "ICP"}, "eval_dataset": {"class": "KITTIDataset"}}
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config_data, f)
            
        # Create evaluation_scores.json
        scores = {
            "per_datapoint": {"metric1": [0.5, 0.6, 0.7]},
            "aggregated": {"metric1": 0.6}
        }
        with open(os.path.join(run_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(scores, f)
            
        icp_dirs.append(run_dir)
    
    return icp_dirs


@patch('runners.viewers.eval_viewer.backend.repetition_discovery.extract_log_dir_info')
def test_discover_experiment_groups_single_experiment(mock_extract, mock_repetition_setup):
    """Test discover_experiment_groups with single experiment."""
    # Mock extract_log_dir_info to return valid LogDirInfo
    mock_info = MagicMock()
    mock_extract.return_value = mock_info
    
    # Provide only first run
    input_dirs = [mock_repetition_setup[0]]
    
    experiment_groups = discover_experiment_groups(input_dirs)
    
    assert len(experiment_groups) == 1
    group = experiment_groups[0]
    assert group.experiment_name == "ICP"
    assert group.num_repetitions == 3
    assert len(group.repetition_paths) == 3


@patch('runners.viewers.eval_viewer.backend.repetition_discovery.extract_log_dir_info')
def test_discover_experiment_groups_input_validation(mock_extract):
    """Test discover_experiment_groups input validation."""
    with pytest.raises(AssertionError, match="log_dirs must not be None"):
        discover_experiment_groups(None)
    
    with pytest.raises(AssertionError, match="log_dirs must be list"):
        discover_experiment_groups("not_a_list")
    
    with pytest.raises(AssertionError, match="log_dirs must not be empty"):
        discover_experiment_groups([])


# ==========================================
# Tests for ExperimentGroup
# ==========================================

@pytest.fixture
def sample_experiment_group(temp_log_dir):
    """Create sample ExperimentGroup for testing."""
    base_path = os.path.join(temp_log_dir, "experiment")
    repetition_paths = []
    
    for i in range(2):
        path = os.path.join(temp_log_dir, f"experiment_run_{i}")
        os.makedirs(path)
        
        # Create minimal valid structure
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump({"test": True}, f)
            
        scores = {"per_datapoint": {"m1": [1, 2]}, "aggregated": {"m1": 1.5}}
        with open(os.path.join(path, "evaluation_scores.json"), 'w') as f:
            json.dump(scores, f)
            
        repetition_paths.append(path)
    
    return ExperimentGroup(
        base_path=base_path,
        experiment_name="TestExp",
        repetition_paths=repetition_paths,
        num_repetitions=2
    )


@patch('runners.viewers.eval_viewer.backend.repetition_discovery.extract_log_dir_info')
def test_experiment_group_get_log_dir_infos(mock_extract, sample_experiment_group):
    """Test ExperimentGroup.get_log_dir_infos."""
    # Mock extract_log_dir_info
    mock_infos = [MagicMock(), MagicMock()]
    mock_extract.side_effect = mock_infos
    
    result = sample_experiment_group.get_log_dir_infos(force_reload=True)
    
    assert len(result) == 2
    assert result == mock_infos
    
    # Verify extract_log_dir_info was called correctly
    assert mock_extract.call_count == 2
    for i, call in enumerate(mock_extract.call_args_list):
        assert call[0][0] == sample_experiment_group.repetition_paths[i]
        assert call[1]['force_reload'] is True


@patch('runners.viewers.eval_viewer.backend.repetition_discovery.extract_log_dir_info')
def test_experiment_group_handles_extraction_errors(mock_extract, sample_experiment_group):
    """Test ExperimentGroup handles extraction errors gracefully."""
    # Make second extraction fail
    mock_extract.side_effect = [MagicMock(), Exception("Test error")]
    
    result = sample_experiment_group.get_log_dir_infos()
    
    # Should return only successful extractions
    assert len(result) == 1