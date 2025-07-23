"""
Integration tests for run_status functionality with enhanced ProgressInfo and ProcessInfo.
Focus on realistic testing with minimal mocking for end-to-end scenarios.

Following CLAUDE.md testing patterns:
- Integration testing for complete pipelines
- Realistic multi-experiment scenarios
- End-to-end workflow validation
"""
from typing import Any
import os
import tempfile
import pytest
from utils.automation.run_status import (
    get_all_run_status,
    RunStatus
)
from utils.automation.progress_tracking import ProgressInfo


# ============================================================================
# INTEGRATION TESTS (REALISTIC WITH MINIMAL MOCK)
# ============================================================================

def test_integration_full_pipeline(setup_realistic_experiment_structure):
    """Integration test for the complete enhanced run_status pipeline with minimal mocking."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create multiple experiments with different states
        experiments = [
            ("running_exp", "running", 5, True),     # Running on GPU, recent logs
            ("stuck_exp", "stuck", 3, False),        # Running on GPU, no recent logs  
            ("finished_exp", "finished", 100, False), # All epochs completed
            ("failed_exp", "failed", 2, False),       # Few epochs, not running
        ]
        
        config_files, work_dirs, system_monitor = setup_realistic_experiment_structure(
            temp_root, experiments
        )
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # Test the complete pipeline with minimal SystemMonitor mock
            all_statuses = get_all_run_status(
                config_files=config_files,
                epochs=100,
                system_monitor=system_monitor
            )
            
            # Verify results
            assert isinstance(all_statuses, dict)
            assert len(all_statuses) == len(experiments)
            
            for config_path in config_files:
                run_status = all_statuses[config_path]
                exp_name = os.path.basename(config_path).replace('.py', '')
                
                # Find expected data for this experiment
                exp_data = next(exp for exp in experiments if exp[0] == exp_name)
                target_status, epochs_completed = exp_data[1], exp_data[2]
                
                # Verify enhanced RunStatus fields
                assert isinstance(run_status.progress, ProgressInfo)
                assert run_status.progress.completed_epochs == epochs_completed
                assert isinstance(run_status.progress.early_stopped, bool)
                
                # For running/stuck experiments, should have ProcessInfo
                if target_status in ["running", "stuck"]:
                    assert run_status.process_info is not None
                    assert run_status.process_info.cmd.endswith(config_path)
                else:
                    assert run_status.process_info is None
                    
        finally:
            os.chdir(original_cwd)


def test_integration_mixed_experiment_states(setup_realistic_experiment_structure):
    """Integration test with a variety of experiment states and configurations."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create experiments with more diverse scenarios
        experiments = [
            ("early_stopped_exp", "finished", 25, False),  # Early stopped experiment
            ("long_running_exp", "running", 80, True),     # Long-running experiment
            ("quick_failure_exp", "failed", 1, False),     # Quick failure
            ("stuck_halfway_exp", "stuck", 50, False),     # Stuck at halfway point
            ("completed_exp", "finished", 100, False),     # Fully completed
        ]
        
        config_files, work_dirs, system_monitor = setup_realistic_experiment_structure(
            temp_root, experiments
        )
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # Test comprehensive status detection
            all_statuses = get_all_run_status(
                config_files=config_files,
                epochs=100,
                system_monitor=system_monitor
            )
            
            # Verify comprehensive results
            assert len(all_statuses) == len(experiments)
            
            # Check specific experiment outcomes
            for config_path in config_files:
                run_status = all_statuses[config_path]
                exp_name = os.path.basename(config_path).replace('.py', '')
                
                # Verify progress tracking is working correctly
                assert hasattr(run_status.progress, 'completed_epochs')
                assert hasattr(run_status.progress, 'early_stopped')
                assert hasattr(run_status.progress, 'progress_percentage')
                
                # Verify status determination is working
                assert run_status.status in ["running", "finished", "failed", "stuck", "outdated"]
                
                # Verify config and work_dir are correctly populated
                assert run_status.config == config_path
                assert isinstance(run_status.work_dir, str)
                
        finally:
            os.chdir(original_cwd)


def test_integration_no_running_experiments(setup_realistic_experiment_structure):
    """Integration test when no experiments are running on GPU."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create only failed/finished experiments (no GPU processes)
        experiments = [
            ("old_finished_exp", "finished", 100, False),
            ("failed_exp1", "failed", 5, False),
            ("failed_exp2", "failed", 0, False),
        ]
        
        config_files, work_dirs, system_monitor = setup_realistic_experiment_structure(
            temp_root, experiments
        )
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            all_statuses = get_all_run_status(
                config_files=config_files,
                epochs=100,
                system_monitor=system_monitor
            )
            
            # All experiments should have no process_info
            for config_path in config_files:
                run_status = all_statuses[config_path]
                assert run_status.process_info is None
                assert run_status.status in ["finished", "failed"]
                
        finally:
            os.chdir(original_cwd)


def test_integration_all_running_experiments(setup_realistic_experiment_structure):
    """Integration test when all experiments are running on GPU."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create only running/stuck experiments (all have GPU processes)
        experiments = [
            ("running_exp1", "running", 10, True),
            ("running_exp2", "running", 25, True),
            ("stuck_exp1", "stuck", 40, False),
            ("stuck_exp2", "stuck", 60, False),
        ]
        
        config_files, work_dirs, system_monitor = setup_realistic_experiment_structure(
            temp_root, experiments
        )
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            all_statuses = get_all_run_status(
                config_files=config_files,
                epochs=100,
                system_monitor=system_monitor
            )
            
            # All experiments should have process_info
            for config_path in config_files:
                run_status = all_statuses[config_path]
                assert run_status.process_info is not None
                assert run_status.status in ["running", "stuck"]
                assert hasattr(run_status.process_info, 'pid')
                assert hasattr(run_status.process_info, 'cmd')
                assert run_status.process_info.cmd.endswith(config_path)
                
        finally:
            os.chdir(original_cwd)


def test_integration_large_scale_experiments(setup_realistic_experiment_structure):
    """Integration test with many experiments to verify scalability."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create many experiments to test scalability
        experiments = [
            (f"exp_{i}", "failed", i % 10, False)  # Various completion levels
            for i in range(20)
        ]
        
        config_files, work_dirs, system_monitor = setup_realistic_experiment_structure(
            temp_root, experiments
        )
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            all_statuses = get_all_run_status(
                config_files=config_files,
                epochs=100,
                system_monitor=system_monitor
            )
            
            # Verify all experiments are processed
            assert len(all_statuses) == 20
            
            # Verify each experiment has correct structure
            for i, config_path in enumerate(config_files):
                run_status = all_statuses[config_path]
                expected_epochs = i % 10
                
                assert run_status.progress.completed_epochs == expected_epochs
                assert hasattr(run_status.progress, 'completed_epochs')
                assert run_status.status == "failed"  # All are failed in this test
                
        finally:
            os.chdir(original_cwd)
