from typing import Dict, Any
import importlib.util


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config using importlib.util pattern from main.py.
    
    Args:
        config_path: Path to the .py config file
        
    Returns:
        The config dictionary from the loaded module
    """
    spec = importlib.util.spec_from_file_location("config_file", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config
