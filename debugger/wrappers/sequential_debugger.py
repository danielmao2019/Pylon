from typing import Dict, Any, List
import os
import threading
import queue
import joblib
import sys
import torch
from debugger.base_debugger import BaseDebugger
from debugger.forward_debugger import ForwardDebugger
from debugger.utils import get_layer_by_name
from utils.builders import build_from_config


class SequentialDebugger(BaseDebugger):
    """Wrapper that runs multiple debuggers sequentially and manages page-based saving."""
    
    def __init__(self, debuggers_config: List[Dict[str, Any]], 
                 model: torch.nn.Module,
                 page_size_mb: int = 100):
        """Initialize sequential debugger.
        
        Args:
            debuggers_config: List of debugger configurations
            model: The model to debug (for forward hook registration)
            page_size_mb: Size limit for each page file in MB
        """
        self.page_size = page_size_mb * 1024 * 1024  # Convert to bytes
        self.enabled = False
        
        # Thread and queue management (following base_criterion.py pattern)
        self._buffer_lock = threading.Lock()
        self._buffer_queue = queue.Queue()
        self._buffer_thread = threading.Thread(target=self._buffer_worker, daemon=True)
        self._buffer_thread.start()
        
        # Page management
        self.current_page_idx = 0
        self.current_page_size = 0
        self.current_page_data = {}  # Dict mapping datapoint_idx to debug_outputs
        self.output_dir = None
        
        # Build debuggers from config with name validation
        self.debuggers = {}  # Dict mapping names to debuggers
        self.forward_debuggers = {}  # layer_name -> list of debuggers
        
        debugger_names = set()
        for cfg in debuggers_config:
            name = cfg['name']
            assert name not in debugger_names, f"Duplicate debugger name: {name}"
            debugger_names.add(name)
            
            debugger = build_from_config(cfg['debugger_config'])
            self.debuggers[name] = debugger
            
            # Track forward debuggers by layer for hook registration
            if isinstance(debugger, ForwardDebugger):
                layer_name = debugger.layer_name
                if layer_name not in self.forward_debuggers:
                    self.forward_debuggers[layer_name] = []
                self.forward_debuggers[layer_name].append(debugger)
        
        # Register forward hooks on model
        self._register_forward_hooks(model)
    
    def _register_forward_hooks(self, model: torch.nn.Module) -> None:
        """Register forward hooks on the model for all forward debuggers."""
        for layer_name, debuggers in self.forward_debuggers.items():
            layer = get_layer_by_name(model, layer_name)
            if layer is not None:
                for debugger in debuggers:
                    layer.register_forward_hook(debugger.forward_hook_fn)
                print(f"Registered {len(debuggers)} forward debugger(s) on layer '{layer_name}'")
            else:
                print(f"Warning: Could not find layer '{layer_name}' for debugger")
    
    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run all debuggers sequentially on the datapoint.
        
        Args:
            datapoint: Dict with inputs, labels, meta_info, outputs
            
        Returns:
            Dict mapping debugger names to their outputs
        """
        if not self.enabled:
            return {}
            
        debug_outputs = {}
        for name, debugger in self.debuggers.items():
            debug_outputs[name] = debugger(datapoint)
        
        # Handle buffering internally (like metric does)
        if debug_outputs:  # Only add to buffer if there are debug outputs
            self.add_to_buffer(debug_outputs, datapoint)
            
        return debug_outputs
    
    def add_to_buffer(self, debug_outputs: Dict[str, Any], datapoint: Dict[str, Dict[str, Any]]):
        """Add debug outputs to buffer for async processing.
        
        Args:
            debug_outputs: Debug outputs from all debuggers
            datapoint: Complete datapoint to extract idx from
        """
        if not self.enabled:
            return
            
        # Extract idx from datapoint meta_info (following BaseMetric pattern)
        assert 'meta_info' in datapoint and 'idx' in datapoint['meta_info']
        idx_raw = datapoint['meta_info']['idx']
        
        # Handle different idx formats (similar to BaseMetric)
        if isinstance(idx_raw, torch.Tensor):
            # Handle tensor format from DataLoader collation
            assert idx_raw.shape == (1,), f"Expected single element tensor, got {idx_raw}"
            assert idx_raw.dtype == torch.int64
            datapoint_idx = idx_raw.item()
        elif isinstance(idx_raw, list):
            # Handle list format
            assert len(idx_raw) == 1
            assert isinstance(idx_raw[0], int)
            datapoint_idx = idx_raw[0]
        elif isinstance(idx_raw, int):
            # Handle direct int format
            datapoint_idx = idx_raw
        else:
            raise ValueError(f"Unsupported idx format: {type(idx_raw)} with value {idx_raw}")
            
        # Calculate memory size using sys.getsizeof recursively
        data_size = self._get_deep_size(debug_outputs)
        
        # Add to queue for background processing
        self._buffer_queue.put({
            'datapoint_idx': datapoint_idx,
            'debug_outputs': debug_outputs,
            'data_size': data_size
        })
    
    def _buffer_worker(self) -> None:
        """Background thread to handle buffer updates (following base_criterion.py pattern)."""
        while True:
            try:
                item = self._buffer_queue.get()
                datapoint_idx = item['datapoint_idx']
                debug_outputs = item['debug_outputs']
                data_size = item['data_size']
                
                with self._buffer_lock:
                    # Add to current page
                    self.current_page_data[datapoint_idx] = debug_outputs
                    self.current_page_size += data_size
                    
                    # Check if we need to save current page
                    if self.current_page_size >= self.page_size:
                        self._save_current_page()
                
                self._buffer_queue.task_done()
            except Exception as e:
                print(f"Debugger buffer worker error: {e}")
    
    def _get_deep_size(self, obj) -> int:
        """Get the deep memory size of an object using sys.getsizeof recursively."""
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(self._get_deep_size(k) + self._get_deep_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(self._get_deep_size(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += self._get_deep_size(obj.__dict__)
            
        return size
    
    def _save_current_page(self):
        """Save current page to disk and reset for next page."""
        if not self.current_page_data or not self.output_dir:
            return
            
        page_path = os.path.join(self.output_dir, f"page_{self.current_page_idx}.pkl")
        try:
            joblib.dump(self.current_page_data, page_path)
        except Exception as e:
            print(f"Error saving debug page {page_path}: {e}")
        
        # Reset for next page
        self.current_page_data = {}
        self.current_page_size = 0
        self.current_page_idx += 1
    
    def save_all(self, output_dir: str):
        """Save all remaining buffer to disk.
        
        Args:
            output_dir: Directory to save debug outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Wait for all queued items to be processed
        self._buffer_queue.join()
        
        # Save any remaining data in current page
        with self._buffer_lock:
            if self.current_page_data:
                self._save_current_page()
    
    def reset_buffer(self):
        """Reset buffer for new epoch (following base_criterion.py pattern)."""
        # Wait for queue to empty before resetting
        self._buffer_queue.join()
        
        # Assert queue is empty (following base_criterion.py and base_metric.py pattern)
        assert self._buffer_queue.empty(), "Buffer queue is not empty when resetting buffer"
        
        with self._buffer_lock:
            self.current_page_data = {}
            self.current_page_size = 0
            self.current_page_idx = 0