"""Simulate various user interaction patterns for benchmarking."""

import time
from typing import List, Dict, Any, Callable
from dataclasses import dataclass


@dataclass
class InteractionEvent:
    """Represents a single user interaction event."""
    timestamp: float  # Time offset from scenario start (seconds)
    callback_name: str  # Name of callback to invoke
    args: tuple  # Positional arguments
    kwargs: dict  # Keyword arguments
    event_type: str  # Type of interaction ('slider', 'button', 'checkbox', etc.)


class InteractionSimulator:
    """Simulates realistic user interaction patterns."""

    def __init__(self, app):
        """Initialize simulator with a mock app instance."""
        self.app = app
        self.events_executed = 0
        self.events_prevented = 0

    def simulate_navigation_slider_dragging(self) -> List[InteractionEvent]:
        """Simulate user dragging navigation slider from index 0 to 50."""
        events = []

        # Simulate dragging with 20ms intervals between events
        for i in range(51):
            event = InteractionEvent(
                timestamp=i * 0.02,  # 20ms between events = 50 events per second
                callback_name='update_datapoint_from_navigation',
                args=(i, self.app.ui_state['3d_settings'].copy(), self.app.ui_state['camera_state'].copy()),
                kwargs={},
                event_type='slider_drag'
            )
            events.append(event)

        return events

    def simulate_3d_settings_adjustment(self) -> List[InteractionEvent]:
        """Simulate adjusting multiple 3D settings simultaneously."""
        events = []
        base_time = 0.0

        # Point size adjustment: 3.0 -> 8.0 (in 0.5 steps)
        for i, point_size in enumerate([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]):
            event = InteractionEvent(
                timestamp=base_time + i * 0.03,  # 30ms intervals
                callback_name='update_3d_settings',
                args=(
                    point_size,
                    0.8,  # opacity
                    0.1,  # sym_diff_radius
                    0.05, # corr_radius
                    'continuous',  # lod_type
                    80    # density
                ),
                kwargs={},
                event_type='slider_drag'
            )
            events.append(event)

        base_time += 0.5  # Pause before next adjustment

        # Opacity adjustment: 0.8 -> 1.0
        for i, opacity in enumerate([0.8, 0.85, 0.9, 0.95, 1.0]):
            event = InteractionEvent(
                timestamp=base_time + i * 0.04,
                callback_name='update_3d_settings',
                args=(8.0, opacity, 0.1, 0.05, 'continuous', 80),
                kwargs={},
                event_type='slider_drag'
            )
            events.append(event)

        return events

    def simulate_mixed_interactions(self) -> List[InteractionEvent]:
        """Simulate realistic mixed usage: navigation + 3D settings + transforms."""
        events = []

        # Start with navigation to index 10
        nav_events = []
        for i in range(11):
            nav_events.append(InteractionEvent(
                timestamp=i * 0.025,
                callback_name='update_datapoint_from_navigation',
                args=(i, self.app.ui_state['3d_settings'].copy(), self.app.ui_state['camera_state'].copy()),
                kwargs={},
                event_type='slider_drag'
            ))
        events.extend(nav_events)

        # Short pause, then adjust point size
        base_time = 0.3
        for i, point_size in enumerate([3.0, 4.0, 5.0, 6.0]):
            events.append(InteractionEvent(
                timestamp=base_time + i * 0.05,
                callback_name='update_3d_settings',
                args=(point_size, 0.8, 0.1, 0.05, 'continuous', 80),
                kwargs={},
                event_type='slider_drag'
            ))

        # Toggle some transforms
        base_time = 0.6
        transform_sequences = [
            [[0]],      # Identity only
            [[0, 1]],   # Identity + noise 0.01
            [[0, 1, 2]], # Identity + noise 0.01 + noise 0.05
        ]

        for i, transform_values in enumerate(transform_sequences):
            events.append(InteractionEvent(
                timestamp=base_time + i * 0.1,
                callback_name='update_datapoint_from_transforms',
                args=(
                    transform_values,
                    self.app.ui_state['3d_settings'].copy(),
                    self.app.ui_state['camera_state'].copy(),
                    10  # current datapoint index
                ),
                kwargs={},
                event_type='checkbox_toggle'
            ))

        # Navigate to another index
        base_time = 1.0
        for i in range(10, 21):  # Navigate from 10 to 20
            events.append(InteractionEvent(
                timestamp=base_time + (i-10) * 0.03,
                callback_name='update_datapoint_from_navigation',
                args=(i, self.app.ui_state['3d_settings'].copy(), self.app.ui_state['camera_state'].copy()),
                kwargs={},
                event_type='slider_drag'
            ))

        return events

    def simulate_stress_test(self) -> List[InteractionEvent]:
        """Simulate extreme rapid interactions - 100+ events in 1 second."""
        events = []

        # Rapid navigation slider dragging (100 events in 1 second)
        for i in range(100):
            events.append(InteractionEvent(
                timestamp=i * 0.01,  # 10ms intervals = 100 events/second
                callback_name='update_datapoint_from_navigation',
                args=(i % 50, self.app.ui_state['3d_settings'].copy(), self.app.ui_state['camera_state'].copy()),
                kwargs={},
                event_type='slider_drag'
            ))

        # Concurrent 3D settings changes
        for i in range(50):
            events.append(InteractionEvent(
                timestamp=i * 0.02 + 0.005,  # Offset by 5ms from navigation events
                callback_name='update_3d_settings',
                args=(
                    3.0 + (i % 10) * 0.5,  # Point size cycling
                    0.5 + (i % 5) * 0.1,   # Opacity cycling
                    0.1, 0.05, 'continuous', 80
                ),
                kwargs={},
                event_type='slider_drag'
            ))

        # Rapid transform toggles
        for i in range(25):
            transform_idx = i % 5
            events.append(InteractionEvent(
                timestamp=i * 0.04 + 0.01,
                callback_name='update_datapoint_from_transforms',
                args=([[transform_idx]], self.app.ui_state['3d_settings'].copy(), self.app.ui_state['camera_state'].copy(), i % 10),
                kwargs={},
                event_type='checkbox_toggle'
            ))

        return events

    def simulate_button_rapid_clicking(self) -> List[InteractionEvent]:
        """Simulate rapid button clicking for navigation."""
        events = []

        # Rapid next button clicks (10 clicks in 0.5 seconds)
        for i in range(10):
            events.append(InteractionEvent(
                timestamp=i * 0.05,
                callback_name='update_index_from_buttons',
                args=(None, i+1, i),  # prev_clicks=None, next_clicks=i+1, current_value=i
                kwargs={},
                event_type='button_click'
            ))

        # Short pause, then rapid prev button clicks
        base_time = 0.6
        for i in range(10):
            events.append(InteractionEvent(
                timestamp=base_time + i * 0.05,
                callback_name='update_index_from_buttons',
                args=(i+1, None, 10-i),  # prev_clicks=i+1, next_clicks=None, current_value=10-i
                kwargs={},
                event_type='button_click'
            ))

        return events

    def simulate_camera_manipulation(self) -> List[InteractionEvent]:
        """Simulate rapid camera dragging/rotation."""
        events = []

        # Simulate rapid camera rotations by changing eye position
        base_camera = self.app.ui_state['camera_state'].copy()

        for i in range(30):
            # Rotate camera around the scene
            angle = i * 0.2  # Incremental rotation
            eye_x = 1.5 + 0.5 * (i % 10 - 5) / 5  # Vary x position
            eye_y = 1.5 + 0.3 * (i % 7 - 3) / 3   # Vary y position
            eye_z = 1.5 + 0.2 * (i % 5 - 2) / 2   # Vary z position

            new_camera = base_camera.copy()
            new_camera['eye'] = {'x': eye_x, 'y': eye_y, 'z': eye_z}

            # Mock relayout data structure
            relayout_data = [
                {'scene.camera': new_camera} if j == 0 else None
                for j in range(3)  # 3 figures
            ]

            mock_figures = [
                {'data': [], 'layout': {'scene': {'camera': base_camera}}},
                {'data': [], 'layout': {'scene': {'camera': base_camera}}},
                {'data': [], 'layout': {'scene': {'camera': base_camera}}}
            ]

            events.append(InteractionEvent(
                timestamp=i * 0.03,  # 30ms intervals
                callback_name='sync_camera_state',
                args=(relayout_data, mock_figures),
                kwargs={},
                event_type='camera_drag'
            ))

        return events

    def execute_scenario(self, events: List[InteractionEvent]) -> Dict[str, Any]:
        """Execute a list of interaction events and collect metrics."""
        start_time = time.time()
        execution_results = []

        self.events_executed = 0
        self.events_prevented = 0

        for event in events:
            # Wait until it's time for this event
            target_time = start_time + event.timestamp
            current_time = time.time()
            if target_time > current_time:
                time.sleep(target_time - current_time)

            # Execute the callback
            callback = getattr(self.app, event.callback_name)

            try:
                exec_start = time.time()
                result = callback(*event.args, **event.kwargs)
                exec_end = time.time()

                execution_results.append({
                    'event_type': event.event_type,
                    'callback_name': event.callback_name,
                    'executed': True,
                    'execution_time': exec_end - exec_start,
                    'timestamp': exec_start - start_time,
                    'result_size': len(str(result)) if result else 0
                })
                self.events_executed += 1

            except Exception as e:
                # Handle PreventUpdate and other exceptions
                execution_results.append({
                    'event_type': event.event_type,
                    'callback_name': event.callback_name,
                    'executed': False,
                    'exception': str(type(e).__name__),
                    'timestamp': time.time() - start_time
                })
                self.events_prevented += 1

        # Wait a bit for any delayed executions
        time.sleep(1.5)  # Wait for debounced callbacks to complete

        total_time = time.time() - start_time

        return {
            'events': execution_results,
            'total_events': len(events),
            'executed_events': self.events_executed,
            'prevented_events': self.events_prevented,
            'total_scenario_time': total_time,
            'events_per_second': len(events) / (events[-1].timestamp if events else 1)
        }


def get_scenario_events(scenario_name: str, app) -> List[InteractionEvent]:
    """Get interaction events for a named scenario.

    Args:
        scenario_name: Name of scenario to generate
        app: Mock app instance

    Returns:
        List of interaction events for the scenario
    """
    simulator = InteractionSimulator(app)

    if scenario_name == 'navigation':
        return simulator.simulate_navigation_slider_dragging()
    elif scenario_name == '3d_settings':
        return simulator.simulate_3d_settings_adjustment()
    elif scenario_name == 'mixed':
        return simulator.simulate_mixed_interactions()
    elif scenario_name == 'stress':
        return simulator.simulate_stress_test()
    elif scenario_name == 'buttons':
        return simulator.simulate_button_rapid_clicking()
    elif scenario_name == 'camera':
        return simulator.simulate_camera_manipulation()
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
