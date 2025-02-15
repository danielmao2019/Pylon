import sys
import os
import importlib.util
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-filepath', required=True)
    parser.add_argument('--test', action='store_true', help="Modify config for testing mode")

    args = parser.parse_args()

    # Load the config module
    spec = importlib.util.spec_from_file_location(name="config_file", location=args.config_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config

    # Modify config if --test is provided
    if args.test:
        for dataset_key in ['train_dataset', 'val_dataset', 'test_dataset']:
            dataloader_key = dataset_key.replace('_dataset', '_dataloader')
            
            if dataset_key in config and config[dataset_key] is not None:
                if dataloader_key in config and config[dataloader_key] is not None:
                    batch_size = config[dataloader_key]['args'].get('batch_size', 1)
                    config[dataset_key]['args']['indices'] = list(range(2 * batch_size))

    # Run training
    config['runner'](config=config).train()
