"""
Test get_session_progress functionality with programmatic simulation of training runs.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs  
- Edge case testing
- Parametrized testing for multiple scenarios
- Determinism testing
"""
from typing import Dict, Any, Optional
import os
import tempfile
import json
import pytest
import torch
from utils.automation.run_status import get_session_progress
from utils.io.json import save_json
from utils.io.config import load_config
from utils.automation.cfg_log_conversion import get_config


def create_epoch_files(work_dir: str, epoch_idx: int, expected_files: list) -> None:
    """Create all expected files for a completed epoch."""
    epoch_dir = os.path.join(work_dir, f"epoch_{epoch_idx}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Create training_losses.pt
    training_losses_path = os.path.join(epoch_dir, "training_losses.pt")
    torch.save({"loss": torch.tensor(0.5)}, training_losses_path)
    
    # Create optimizer_buffer.json
    optimizer_buffer_path = os.path.join(epoch_dir, "optimizer_buffer.json")
    with open(optimizer_buffer_path, 'w') as f:
        json.dump({"lr": 0.001}, f)
    
    # Create validation_scores.json
    validation_scores_path = os.path.join(epoch_dir, "validation_scores.json")
    validation_scores = {
        "aggregated": {"loss": 0.4 - epoch_idx * 0.01},  # Improving scores
        "per_datapoint": {"loss": [0.4 - epoch_idx * 0.01]}
    }
    with open(validation_scores_path, 'w') as f:
        json.dump(validation_scores, f)


def create_progress_json(work_dir: str, completed_epochs: int, early_stopped: bool = False, 
                        early_stopped_at_epoch: Optional[int] = None, tot_epochs: int = 100) -> None:
    """Create progress.json file for fast path testing."""
    progress_data = {
        "completed_epochs": completed_epochs,
        "progress_percentage": 100.0 if early_stopped else (completed_epochs / tot_epochs * 100.0),
        "early_stopped": early_stopped,
        "early_stopped_at_epoch": early_stopped_at_epoch
    }
    progress_file = os.path.join(work_dir, "progress.json")
    save_json(progress_data, progress_file)


def create_mock_config(config_path: str, epochs: int = 100, early_stopping_enabled: bool = False, patience: int = 5) -> None:
    """Create a mock config file for testing."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    config_content = f'''
import torch
from metrics.wrappers import PyTorchMetricWrapper
from runners.early_stopping import EarlyStopping

config = {{
    'epochs': {epochs},
    'work_dir': './logs/test_run',
    'metric': {{
        'class': PyTorchMetricWrapper,
        'args': {{
            'metric': torch.nn.MSELoss(reduction='mean'),
        }},
    }},
'''
    
    if early_stopping_enabled:
        config_content += f'''
    'early_stopping': {{
        'class': EarlyStopping,
        'args': {{
            'enabled': True,
            'epochs': {patience},
        }},
    }},
'''
    
    config_content += '''
}
'''
    
    with open(config_path, 'w') as f:
        f.write(config_content)


def test_progress_fast_path_normal_run():
    """Test fast path with progress.json for normal (non-early stopped) run."""
    with tempfile.TemporaryDirectory() as work_dir:
        expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        
        # Create progress.json for normal run (57/100 epochs)
        create_progress_json(work_dir, completed_epochs=57, early_stopped=False, tot_epochs=100)
        
        progress = get_session_progress(work_dir, expected_files)
        
        assert progress == 57, f"Expected 57 completed epochs, got {progress}"


def test_progress_fast_path_early_stopped_run(monkeypatch):
    """Test fast path with progress.json for early stopped run."""
    with tempfile.TemporaryDirectory() as work_dir:
        with tempfile.TemporaryDirectory() as config_dir:
            expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
            
            # Create progress.json for early stopped run (57/100 epochs, early stopped)
            create_progress_json(work_dir, completed_epochs=57, early_stopped=True, 
                               early_stopped_at_epoch=57, tot_epochs=100)
            
            # Create mock config
            config_path = os.path.join(config_dir, "test_config.py")
            create_mock_config(config_path, epochs=100, early_stopping_enabled=True)
            
            # Mock get_config to return our test config
            def mock_get_config(work_dir_param):
                return config_path
            monkeypatch.setattr('utils.automation.run_status.get_config', mock_get_config)
            
            progress = get_session_progress(work_dir, expected_files)
            
            # Should return total epochs (100) for early stopped run
            assert progress == 100, f"Expected 100 (total epochs) for early stopped run, got {progress}"


@pytest.mark.parametrize("completed_epochs,expected_progress", [
    (0, 0),      # No epochs completed
    (1, 1),      # One epoch completed  
    (50, 50),    # Half completed
    (99, 99),    # Almost complete
    (100, 100),  # Fully complete
])
def test_progress_slow_path_normal_runs(completed_epochs, expected_progress):
    """Test slow path (no progress.json) for various normal completion levels."""
    with tempfile.TemporaryDirectory() as work_dir:
        with tempfile.TemporaryDirectory() as config_dir:
            expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
            
            # Create epoch files for completed epochs
            for epoch_idx in range(completed_epochs):
                create_epoch_files(work_dir, epoch_idx, expected_files)
            
            # Create mock config (no early stopping)
            config_path = os.path.join(config_dir, "test_config.py")
            create_mock_config(config_path, epochs=100, early_stopping_enabled=False)
            
            # Mock get_config
            import utils.automation.run_status
            original_get_config = utils.automation.run_status.get_config
            utils.automation.run_status.get_config = lambda work_dir_param: config_path
            
            try:
                progress = get_session_progress(work_dir, expected_files)
                assert progress == expected_progress, f"Expected {expected_progress}, got {progress}"
                
                # Verify progress.json was created
                progress_file = os.path.join(work_dir, "progress.json")
                assert os.path.exists(progress_file), "progress.json should have been created"
                
                # Verify progress.json content
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                assert progress_data['completed_epochs'] == completed_epochs
                assert progress_data['early_stopped'] == False
                assert progress_data['early_stopped_at_epoch'] is None
                
            finally:
                utils.automation.run_status.get_config = original_get_config


def test_progress_slow_path_early_stopped_run():
    """Test slow path detection of early stopped run."""
    with tempfile.TemporaryDirectory() as work_dir:
        with tempfile.TemporaryDirectory() as config_dir:
            expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
            
            # Create 10 epochs (less than total 100)
            completed_epochs = 10
            for epoch_idx in range(completed_epochs):
                create_epoch_files(work_dir, epoch_idx, expected_files)
            
            # Create mock config with early stopping enabled
            config_path = os.path.join(config_dir, "test_config.py")
            create_mock_config(config_path, epochs=100, early_stopping_enabled=True, patience=3)
            
            # Mock get_config and build_from_config to simulate early stopping detection
            import utils.automation.run_status
            original_get_config = utils.automation.run_status.get_config
            original_build_from_config = utils.automation.run_status.build_from_config
            
            utils.automation.run_status.get_config = lambda work_dir_param: config_path
            
            # Mock early stopping detection to return True (early stopped at epoch 10)
            def mock_detect_early_stopping(work_dir_param, expected_files_param, config_param, completed_epochs_param):
                return True, 10
            
            utils.automation.run_status._detect_early_stopping = mock_detect_early_stopping
            
            try:
                progress = get_session_progress(work_dir, expected_files)
                
                # Should return total epochs (100) for early stopped run
                assert progress == 100, f"Expected 100 (total epochs) for early stopped run, got {progress}"
                
                # Verify progress.json was created with early stopping info
                progress_file = os.path.join(work_dir, "progress.json")
                assert os.path.exists(progress_file), "progress.json should have been created"
                
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                assert progress_data['completed_epochs'] == 10
                assert progress_data['early_stopped'] == True
                assert progress_data['early_stopped_at_epoch'] == 10
                assert progress_data['progress_percentage'] == 100.0
                
            finally:
                utils.automation.run_status.get_config = original_get_config
                utils.automation.run_status.build_from_config = original_build_from_config


def test_progress_incomplete_epochs():
    """Test progress calculation with incomplete epochs (missing files)."""
    with tempfile.TemporaryDirectory() as work_dir:
        with tempfile.TemporaryDirectory() as config_dir:
            expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
            
            # Create 3 complete epochs
            for epoch_idx in range(3):
                create_epoch_files(work_dir, epoch_idx, expected_files)
            
            # Create incomplete epoch 3 (missing validation_scores.json)
            epoch_dir = os.path.join(work_dir, "epoch_3")
            os.makedirs(epoch_dir, exist_ok=True)
            torch.save({"loss": torch.tensor(0.5)}, os.path.join(epoch_dir, "training_losses.pt"))
            with open(os.path.join(epoch_dir, "optimizer_buffer.json"), 'w') as f:
                json.dump({"lr": 0.001}, f)
            # Missing validation_scores.json
            
            # Create mock config
            config_path = os.path.join(config_dir, "test_config.py")
            create_mock_config(config_path, epochs=100, early_stopping_enabled=False)
            
            # Mock get_config
            import utils.automation.run_status
            original_get_config = utils.automation.run_status.get_config
            utils.automation.run_status.get_config = lambda work_dir_param: config_path
            
            try:
                progress = get_session_progress(work_dir, expected_files)
                
                # Should only count complete epochs (3)
                assert progress == 3, f"Expected 3 complete epochs, got {progress}"
                
            finally:
                utils.automation.run_status.get_config = original_get_config


def test_progress_deterministic():
    """Test that progress calculation is deterministic across multiple calls."""
    with tempfile.TemporaryDirectory() as work_dir:
        expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        
        # Create progress.json
        create_progress_json(work_dir, completed_epochs=42, early_stopped=False)
        
        # Run multiple times and verify same result
        results = []
        for _ in range(3):
            progress = get_session_progress(work_dir, expected_files)
            results.append(progress)
        
        assert all(r == results[0] for r in results), f"Results not deterministic: {results}"
        assert results[0] == 42, f"Expected 42, got {results[0]}"


@pytest.mark.parametrize("early_stopped,completed,total,expected", [
    (False, 25, 100, 25),    # Normal run: return completed epochs
    (False, 100, 100, 100),  # Complete normal run  
    (True, 30, 100, 100),    # Early stopped: return total epochs
    (True, 75, 200, 200),    # Early stopped with different total
])
def test_progress_parametrized_scenarios(early_stopped, completed, total, expected, monkeypatch):
    """Parametrized test for various progress scenarios."""
    with tempfile.TemporaryDirectory() as work_dir:
        with tempfile.TemporaryDirectory() as config_dir:
            expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
            
            # Create progress.json for fast path
            create_progress_json(work_dir, completed_epochs=completed, early_stopped=early_stopped, 
                               early_stopped_at_epoch=completed if early_stopped else None, tot_epochs=total)
            
            if early_stopped:
                # For early stopped runs, need to mock config loading
                config_path = os.path.join(config_dir, "test_config.py")
                create_mock_config(config_path, epochs=total, early_stopping_enabled=True)
                
                def mock_get_config(work_dir_param):
                    return config_path
                monkeypatch.setattr('utils.automation.run_status.get_config', mock_get_config)
            
            progress = get_session_progress(work_dir, expected_files)
            assert progress == expected, f"Expected {expected}, got {progress} for scenario: early_stopped={early_stopped}, completed={completed}, total={total}"


def test_progress_edge_case_empty_work_dir():
    """Test progress calculation with empty work directory."""
    with tempfile.TemporaryDirectory() as work_dir:
        with tempfile.TemporaryDirectory() as config_dir:
            expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
            
            # Create mock config
            config_path = os.path.join(config_dir, "test_config.py")
            create_mock_config(config_path, epochs=100, early_stopping_enabled=False)
            
            # Mock get_config
            import utils.automation.run_status
            original_get_config = utils.automation.run_status.get_config
            utils.automation.run_status.get_config = lambda work_dir_param: config_path
            
            try:
                progress = get_session_progress(work_dir, expected_files)
                assert progress == 0, f"Expected 0 for empty work dir, got {progress}"
                
            finally:
                utils.automation.run_status.get_config = original_get_config
