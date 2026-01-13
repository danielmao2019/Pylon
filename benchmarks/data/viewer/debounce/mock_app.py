"""Mock Dash app with viewer callbacks for benchmarking."""

import time
from typing import Dict, List, Optional, Union, Any
from unittest.mock import Mock
from dash.exceptions import PreventUpdate

from data.viewer.utils.debounce import debounce
from .mock_data import get_mock_datapoint, generate_synthetic_pcr_dataset


class MockViewerApp:
    """Mock Dash app that simulates viewer callbacks without actual UI."""

    def __init__(self, use_debounce: bool = True, num_datapoints: int = 100, num_points: int = 5000):
        """Initialize mock app with synthetic data.

        Args:
            use_debounce: Whether to use debouncing decorators
            num_datapoints: Number of datapoints in synthetic dataset
            num_points: Number of points per point cloud
        """
        self.use_debounce = use_debounce
        self.dataset = generate_synthetic_pcr_dataset(num_datapoints, num_points)

        # Mock UI state
        self.ui_state = {
            'current_datapoint_index': 0,
            'selected_transforms': [0],  # Start with identity transform
            '3d_settings': {
                'point_size': 3.0,
                'point_opacity': 0.8,
                'sym_diff_radius': 0.1,
                'corr_radius': 0.05,
                'lod_type': 'continuous',
                'density_percentage': 80
            },
            'camera_state': {
                'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
                'center': {'x': 0, 'y': 0, 'z': 0},
                'up': {'x': 0, 'y': 0, 'z': 1}
            }
        }

        # Mock expensive operations (simulate real callback work)
        self.mock_backend_processing = Mock()
        self.mock_display_creation = Mock()
        self.mock_3d_rendering = Mock()
        self.mock_camera_sync = Mock()

        # Apply debouncing conditionally
        if use_debounce:
            self.update_datapoint_from_navigation = debounce(self._update_datapoint_from_navigation)
            self.update_index_from_buttons = debounce(self._update_index_from_buttons)
            self.update_3d_settings = debounce(self._update_3d_settings)
            self.update_datapoint_from_transforms = debounce(self._update_datapoint_from_transforms)
            self.sync_camera_state = debounce(self._sync_camera_state)
        else:
            self.update_datapoint_from_navigation = self._update_datapoint_from_navigation
            self.update_index_from_buttons = self._update_index_from_buttons
            self.update_3d_settings = self._update_3d_settings
            self.update_datapoint_from_transforms = self._update_datapoint_from_transforms
            self.sync_camera_state = self._sync_camera_state

    def _simulate_expensive_work(self, operation_name: str, duration: float = 0.1):
        """Simulate expensive callback work."""
        time.sleep(duration)  # Simulate processing time
        if operation_name == 'backend_processing':
            self.mock_backend_processing()
        elif operation_name == 'display_creation':
            self.mock_display_creation()
        elif operation_name == '3d_rendering':
            self.mock_3d_rendering()
        elif operation_name == 'camera_sync':
            self.mock_camera_sync()

    def _update_datapoint_from_navigation(self, datapoint_idx: int, settings_3d: Dict, camera_state: Dict) -> List[Dict]:
        """Mock navigation callback - similar to actual viewer callback."""
        # Simulate dataset validation
        if datapoint_idx < 0 or datapoint_idx >= len(self.dataset['datapoints']):
            raise PreventUpdate

        # Simulate expensive backend processing
        self._simulate_expensive_work('backend_processing', 0.05)

        # Get datapoint with current transforms
        datapoint = get_mock_datapoint(
            self.dataset,
            datapoint_idx,
            self.ui_state['selected_transforms']
        )

        # Simulate expensive display creation
        self._simulate_expensive_work('display_creation', 0.08)

        # Update state
        self.ui_state['current_datapoint_index'] = datapoint_idx
        self.ui_state['3d_settings'].update(settings_3d)
        self.ui_state['camera_state'].update(camera_state)

        # Return mock display result
        return [{
            'type': 'pcr_display',
            'datapoint_index': datapoint_idx,
            'source_points': datapoint['meta_info']['source_num_points'],
            'target_points': datapoint['meta_info']['target_num_points'],
            'settings': settings_3d.copy()
        }]

    def _update_index_from_buttons(self, prev_clicks: Optional[int], next_clicks: Optional[int], current_value: int) -> List[int]:
        """Mock button navigation callback."""
        if prev_clicks is None and next_clicks is None:
            raise PreventUpdate

        # Simulate lightweight processing
        self._simulate_expensive_work('backend_processing', 0.02)

        # Determine new index (simplified logic)
        if prev_clicks and prev_clicks > 0:
            new_value = max(0, current_value - 1)
        else:
            new_value = min(len(self.dataset['datapoints']) - 1, current_value + 1)

        self.ui_state['current_datapoint_index'] = new_value
        return [new_value]

    def _update_3d_settings(self, point_size: float, point_opacity: float, sym_diff_radius: float,
                           corr_radius: float, lod_type: str, density_percentage: int) -> List[Dict]:
        """Mock 3D settings callback."""
        if point_size is None or point_opacity is None:
            raise PreventUpdate

        # Simulate moderate processing time for 3D settings
        self._simulate_expensive_work('3d_rendering', 0.06)

        settings = {
            'point_size': point_size,
            'point_opacity': point_opacity,
            'sym_diff_radius': sym_diff_radius,
            'corr_radius': corr_radius,
            'lod_type': lod_type,
            'density_percentage': density_percentage
        }

        self.ui_state['3d_settings'].update(settings)
        return [settings]

    def _update_datapoint_from_transforms(self, transform_values: List[List[int]], settings_3d: Dict,
                                         camera_state: Dict, datapoint_idx: int) -> List[Dict]:
        """Mock transform selection callback."""
        # Get selected transform indices
        selected_indices = [
            idx for values in transform_values
            for idx in values if values
        ]

        if not selected_indices:
            selected_indices = [0]  # Default to identity

        # Simulate expensive reprocessing with new transforms
        self._simulate_expensive_work('backend_processing', 0.07)

        # Get datapoint with new transforms
        datapoint = get_mock_datapoint(self.dataset, datapoint_idx, selected_indices)

        # Simulate display recreation
        self._simulate_expensive_work('display_creation', 0.09)

        self.ui_state['selected_transforms'] = selected_indices

        return [{
            'type': 'pcr_display',
            'transforms_applied': selected_indices,
            'source_points': datapoint['meta_info']['source_num_points'],
            'target_points': datapoint['meta_info']['target_num_points']
        }]

    def _sync_camera_state(self, all_relayout_data: List[Dict], all_figures: List[Dict]) -> tuple:
        """Mock camera synchronization callback."""
        # Find triggered camera change (simplified)
        new_camera = None
        for relayout_data in all_relayout_data:
            if relayout_data and 'scene.camera' in relayout_data:
                new_camera = relayout_data['scene.camera']
                break

        if new_camera is None:
            raise PreventUpdate

        # Simulate camera sync processing
        self._simulate_expensive_work('camera_sync', 0.04)

        # Update all figures (mock)
        updated_figures = []
        for i, figure in enumerate(all_figures):
            if i == 0:  # Skip source figure (matches real implementation)
                updated_figures.append('no_update')
            else:
                updated_figures.append({
                    'layout': {'scene': {'camera': new_camera}},
                    'data': figure.get('data', [])
                })

        self.ui_state['camera_state'].update(new_camera)
        return updated_figures, new_camera


def create_mock_app(use_debounce: bool = True, **kwargs) -> MockViewerApp:
    """Factory function to create a mock app instance.

    Args:
        use_debounce: Whether to apply debouncing decorators
        **kwargs: Additional arguments for MockViewerApp

    Returns:
        Configured mock app instance
    """
    return MockViewerApp(use_debounce=use_debounce, **kwargs)