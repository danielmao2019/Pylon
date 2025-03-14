"""Viewer state management.

This module contains the ViewerState class which manages the state of the dataset viewer,
including the current dataset, index, and transform settings.
"""
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import json
from pathlib import Path


class ViewerEvent(Enum):
    """Events that can occur in the viewer."""
    DATASET_CHANGED = "dataset_changed"
    INDEX_CHANGED = "index_changed"
    TRANSFORMS_CHANGED = "transforms_changed"
    SETTINGS_CHANGED = "settings_changed"
    STATE_CHANGED = "state_changed"


@dataclass
class DatasetInfo:
    """Information about the currently loaded dataset."""
    name: str = ""
    length: int = 0
    class_labels: Dict[int, str] = None
    is_3d: bool = False
    transforms: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.class_labels is None:
            self.class_labels = {}
        if self.transforms is None:
            self.transforms = []

    def validate(self) -> bool:
        """Validate the dataset information.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(self.name, str):
            return False
        if not isinstance(self.length, int) or self.length < 0:
            return False
        if not isinstance(self.class_labels, dict):
            return False
        if not isinstance(self.is_3d, bool):
            return False
        if not isinstance(self.transforms, list):
            return False
        return True


@dataclass
class ViewerStateSnapshot:
    """Snapshot of the viewer state for undo/redo functionality."""
    dataset_info: DatasetInfo
    current_index: int
    transforms: Dict[str, bool]
    point_size: float
    point_opacity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            'dataset_info': asdict(self.dataset_info),
            'current_index': self.current_index,
            'transforms': self.transforms,
            'point_size': self.point_size,
            'point_opacity': self.point_opacity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ViewerStateSnapshot':
        """Create snapshot from dictionary."""
        return cls(
            dataset_info=DatasetInfo(**data['dataset_info']),
            current_index=data['current_index'],
            transforms=data['transforms'],
            point_size=data['point_size'],
            point_opacity=data['point_opacity']
        )


class ViewerState:
    """Manages the viewer's state and configuration.
    
    This class maintains the state of the dataset viewer, including:
    - Current dataset information
    - Current datapoint index
    - Transform settings
    - 3D visualization settings
    """
    
    def __init__(self, max_history: int = 50):
        """Initialize the viewer state.
        
        Args:
            max_history: Maximum number of states to keep in history
        """
        # Dataset information
        self.current_dataset: Optional[str] = None
        self.dataset_info: DatasetInfo = DatasetInfo()
        
        # Navigation state
        self.current_index: int = 0
        self.min_index: int = 0
        self.max_index: int = 0
        
        # Transform settings
        self.transforms: Dict[str, bool] = {}
        
        # 3D visualization settings
        self.point_size: float = 1.0
        self.point_opacity: float = 1.0
        
        # Event handlers
        self._event_handlers: Dict[ViewerEvent, List[Callable]] = {
            event: [] for event in ViewerEvent
        }

        # Undo/redo history
        self._max_history: int = max_history
        self._history: List[ViewerStateSnapshot] = []
        self._current_history_index: int = -1
    
    def subscribe(self, event: ViewerEvent, handler: Callable) -> None:
        """Subscribe to a viewer event.
        
        Args:
            event: The event to subscribe to
            handler: Callback function to handle the event
        """
        self._event_handlers[event].append(handler)
    
    def unsubscribe(self, event: ViewerEvent, handler: Callable) -> None:
        """Unsubscribe from a viewer event.
        
        Args:
            event: The event to unsubscribe from
            handler: Callback function to remove
        """
        if handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)
    
    def _emit_event(self, event: ViewerEvent, data: Any = None) -> None:
        """Emit an event to all subscribers.
        
        Args:
            event: The event to emit
            data: Optional data to pass to handlers
        """
        for handler in self._event_handlers[event]:
            handler(data)

    def _save_to_history(self) -> None:
        """Save current state to history."""
        # Remove any future states if we're not at the end
        if self._current_history_index < len(self._history) - 1:
            self._history = self._history[:self._current_history_index + 1]

        # Create new snapshot
        snapshot = ViewerStateSnapshot(
            dataset_info=self.dataset_info,
            current_index=self.current_index,
            transforms=self.transforms.copy(),
            point_size=self.point_size,
            point_opacity=self.point_opacity
        )

        # Add to history
        self._history.append(snapshot)
        self._current_history_index += 1

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
            self._current_history_index = len(self._history) - 1

    def undo(self) -> bool:
        """Undo the last state change.
        
        Returns:
            bool: True if undo was successful, False otherwise
        """
        if self._current_history_index <= 0:
            return False

        self._current_history_index -= 1
        self._restore_snapshot(self._history[self._current_history_index])
        return True

    def redo(self) -> bool:
        """Redo the last undone state change.
        
        Returns:
            bool: True if redo was successful, False otherwise
        """
        if self._current_history_index >= len(self._history) - 1:
            return False

        self._current_history_index += 1
        self._restore_snapshot(self._history[self._current_history_index])
        return True

    def _restore_snapshot(self, snapshot: ViewerStateSnapshot) -> None:
        """Restore state from a snapshot.
        
        Args:
            snapshot: The snapshot to restore
        """
        self.dataset_info = snapshot.dataset_info
        self.current_index = snapshot.current_index
        self.transforms = snapshot.transforms.copy()
        self.point_size = snapshot.point_size
        self.point_opacity = snapshot.point_opacity
        self._emit_event(ViewerEvent.STATE_CHANGED)

    def save_state(self, filepath: str) -> None:
        """Save current state to file.
        
        Args:
            filepath: Path to save state to
        """
        snapshot = ViewerStateSnapshot(
            dataset_info=self.dataset_info,
            current_index=self.current_index,
            transforms=self.transforms,
            point_size=self.point_size,
            point_opacity=self.point_opacity
        )
        
        with open(filepath, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)

    def load_state(self, filepath: str) -> None:
        """Load state from file.
        
        Args:
            filepath: Path to load state from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        snapshot = ViewerStateSnapshot.from_dict(data)
        self._restore_snapshot(snapshot)
        self._save_to_history()
    
    def update_dataset_info(self, name: str, length: int, class_labels: Dict[int, str], 
                          is_3d: bool, transforms: List[Dict[str, Any]] = None) -> None:
        """Update the current dataset information.
        
        Args:
            name: Name of the dataset
            length: Number of datapoints in the dataset
            class_labels: Dictionary mapping class indices to labels
            is_3d: Whether the dataset contains 3D data
            transforms: List of transform info dictionaries
        """
        # Create new dataset info
        new_info = DatasetInfo(
            name=name,
            length=length,
            class_labels=class_labels,
            is_3d=is_3d,
            transforms=transforms or []
        )

        # Validate
        if not new_info.validate():
            raise ValueError("Invalid dataset information")

        # Update state
        self.current_dataset = name
        self.dataset_info = new_info
        self.min_index = 0
        self.max_index = length - 1
        self.current_index = 0
        
        # Initialize transform states
        self.transforms = {str(i): False for i in range(len(new_info.transforms))}
        
        # Save to history and emit event
        self._save_to_history()
        self._emit_event(ViewerEvent.DATASET_CHANGED, {
            'name': name,
            'length': length,
            'is_3d': is_3d
        })
    
    def update_index(self, index: int) -> None:
        """Update the current datapoint index.
        
        Args:
            index: New index value
        """
        if self.dataset_info.length == 0:
            return
            
        old_index = self.current_index
        self.current_index = max(self.min_index, min(self.max_index, index))
        
        if old_index != self.current_index:
            self._save_to_history()
            self._emit_event(ViewerEvent.INDEX_CHANGED, {
                'old_index': old_index,
                'new_index': self.current_index
            })
    
    def update_transforms(self, transforms: List[Dict[str, Any]]) -> None:
        """Update the transform settings.
        
        Args:
            transforms: List of transform info dictionaries
        """
        # Initialize transform states
        self.transforms = {
            str(transform['index']): False for transform in transforms
        }
        self._save_to_history()
        self._emit_event(ViewerEvent.TRANSFORMS_CHANGED, self.transforms)
    
    def update_3d_settings(self, point_size: float, point_opacity: float) -> None:
        """Update 3D visualization settings.
        
        Args:
            point_size: Size of points in 3D visualization
            point_opacity: Opacity of points in 3D visualization
        """
        # Validate settings
        if point_size <= 0:
            raise ValueError("Point size must be positive")
        if not 0 <= point_opacity <= 1:
            raise ValueError("Point opacity must be between 0 and 1")

        self.point_size = point_size
        self.point_opacity = point_opacity
        self._save_to_history()
        self._emit_event(ViewerEvent.SETTINGS_CHANGED, {
            'point_size': point_size,
            'point_opacity': point_opacity
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state as a dictionary.
        
        Returns:
            Dictionary containing the current state
        """
        return {
            'current_dataset': self.current_dataset,
            'dataset_info': asdict(self.dataset_info),
            'current_index': self.current_index,
            'min_index': self.min_index,
            'max_index': self.max_index,
            'transforms': self.transforms,
            'point_size': self.point_size,
            'point_opacity': self.point_opacity
        }
    
    def reset(self) -> None:
        """Reset the state to its initial values."""
        self.current_dataset = None
        self.dataset_info = DatasetInfo()
        self.current_index = 0
        self.min_index = 0
        self.max_index = 0
        self.transforms = {}
        self.point_size = 1.0
        self.point_opacity = 1.0
        self._save_to_history()
        self._emit_event(ViewerEvent.STATE_CHANGED) 