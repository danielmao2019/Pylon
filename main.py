import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib.util


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-filepath')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        name="config_file", location=args.config_filepath
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config
    config['runner'](config=config).train()
