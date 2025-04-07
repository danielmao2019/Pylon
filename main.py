import sys
import os
import importlib.util
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-filepath', required=True)
    parser.add_argument('--debug', action='store_true', help="Modify config for debugging mode")

    args = parser.parse_args()

    # Load the config module
    spec = importlib.util.spec_from_file_location("config_file", args.config_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config

    # Modify config if --debug is provided
    if args.debug:
        config['work_dir'] = os.path.join("./logs", "debug", os.path.relpath(config['work_dir'], start="./logs"))
        os.system(f"rm -rf {config['work_dir']}")
        config['epochs'] = 3
        config['train_seeds'] = config['train_seeds'][:3]
        for split in ['train', 'val', 'test']:
            dataset_key = split + "_dataset"
            dataloader_key = split + "_dataloader"

            if dataset_key in config and config[dataset_key] is not None:
                if dataloader_key in config and config[dataloader_key] is not None:
                    batch_size = config[dataloader_key]['args'].get('batch_size', 1)
                    config[dataset_key]['args']['indices'] = list(range(3 * batch_size))

    # Run training
    config['runner'](config=config).run()
