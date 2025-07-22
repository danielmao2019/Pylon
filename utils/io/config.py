from typing import Dict, Any
import importlib.util
import threading


# Thread-safe config loading with caching
_config_cache = {}
_config_cache_lock = threading.Lock()


def load_config(config_path: str) -> Dict[str, Any]:
    """Thread-safe config loading with caching to prevent import deadlocks.
    
    This function prevents Python import deadlocks that occur when multiple threads
    try to import the same modules simultaneously during concurrent config loading.
    
    Args:
        config_path: Path to the .py config file
        
    Returns:
        The config dictionary from the loaded module
    """
    with _config_cache_lock:
        if config_path not in _config_cache:
            spec = importlib.util.spec_from_file_location("config_file", config_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load config from {config_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _config_cache[config_path] = module.config
        return _config_cache[config_path]
