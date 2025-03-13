"""Viewer state management.

This module contains the ViewerState class which manages the state of the dataset viewer,
including the current dataset, index, and transform settings.
"""
from typing import Dict, Any, Optional, List, Callable
from enum import Enum


class ViewerEvent(Enum):
    """Events that can occur in the viewer."""
    DATASET_CHANGED = "dataset_changed"
    INDEX_CHANGED = "index_changed"
    TRANSFORMS_CHANGED = "transforms_changed"
    SETTINGS_CHANGED = "settings_changed"


class DatasetInfo:
    """Information about the currently loaded dataset."""
    def __init__(self):
        """Initialize dataset information."""
        self.name: str = ""
        self.length: int = 0
        self.class_labels: Dict[int, str] = {}
        self.is_3d: bool = False
        self.available_transforms: List[str] = []


class ViewerState:
    """Manages the viewer's state and configuration.
    
    This class maintains the state of the dataset viewer, including:
    - Current dataset information
    - Current datapoint index
    - Transform settings
    - 3D visualization settings
    """
    
    def __init__(self):
        """Initialize the viewer state."""
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
        self.is_3d: bool = False
        self.point_size: float = 1.0
        self.point_opacity: float = 1.0
        
        # Event handlers
        self._event_handlers: Dict[ViewerEvent, List[Callable]] = {
            event: [] for event in ViewerEvent
        }
    
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
    
    def update_dataset_info(self, name: str, length: int, class_labels: Dict[int, str], 
                          is_3d: bool, available_transforms: List[str] = None) -> None:
        """Update the current dataset information.
        
        Args:
            name: Name of the dataset
            length: Number of datapoints in the dataset
            class_labels: Dictionary mapping class indices to labels
            is_3d: Whether the dataset contains 3D data
            available_transforms: List of available transforms for the dataset
        """
        self.current_dataset = name
        self.dataset_info = DatasetInfo()
        self.dataset_info.name = name
        self.dataset_info.length = length
        self.dataset_info.class_labels = class_labels
        self.dataset_info.is_3d = is_3d
        self.dataset_info.available_transforms = available_transforms or []
        
        self.is_3d = is_3d
        self.min_index = 0
        self.max_index = length - 1
        self.current_index = 0
        
        # Reset transforms when dataset changes
        self.transforms = {transform: False for transform in self.dataset_info.available_transforms}
        
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
            self._emit_event(ViewerEvent.INDEX_CHANGED, {
                'old_index': old_index,
                'new_index': self.current_index
            })
    
    def update_transforms(self, transforms: Dict[str, bool]) -> None:
        """Update the transform settings.
        
        Args:
            transforms: Dictionary mapping transform names to enabled state
        """
        self.transforms = transforms
        self._emit_event(ViewerEvent.TRANSFORMS_CHANGED, transforms)
    
    def update_3d_settings(self, point_size: float, point_opacity: float) -> None:
        """Update 3D visualization settings.
        
        Args:
            point_size: Size of points in 3D visualization
            point_opacity: Opacity of points in 3D visualization
        """
        self.point_size = point_size
        self.point_opacity = point_opacity
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
            'dataset_info': {
                'name': self.dataset_info.name,
                'length': self.dataset_info.length,
                'class_labels': self.dataset_info.class_labels,
                'is_3d': self.dataset_info.is_3d,
                'available_transforms': self.dataset_info.available_transforms
            },
            'current_index': self.current_index,
            'min_index': self.min_index,
            'max_index': self.max_index,
            'transforms': self.transforms,
            'is_3d': self.is_3d,
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
        self.is_3d = False
        self.point_size = 1.0
        self.point_opacity = 1.0 