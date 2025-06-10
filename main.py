import argparse
import os
import sys
import importlib.util
import cProfile
import pstats
from pstats import SortKey

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-filepath', required=True)
    parser.add_argument('--debug', action='store_true', help="Modify config for debugging mode")
    parser.add_argument('--profile', action='store_true', help="Enable profiling")

    args = parser.parse_args()

    # Load the config module
    spec = importlib.util.spec_from_file_location("config_file", args.config_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config

    if isinstance(config, dict):
        config_list = [config]
    else:
        config_list = config

    for config_idx, config in enumerate(config_list):
        # Modify config if --debug is provided
        if args.debug:
            config['work_dir'] = os.path.join("./logs", "debug", os.path.relpath(config['work_dir'], start="./logs"))
            os.system(f"rm -rf {config['work_dir']}")
            config['epochs'] = 3
            config['train_seeds'] = config['train_seeds'][:3]
            config['val_seeds'] = config['val_seeds'][:3]
            for split in ['train', 'val', 'test']:
                dataset_key = split + "_dataset"
                dataloader_key = split + "_dataloader"

                if dataset_key in config and config[dataset_key] is not None:
                    if dataloader_key in config and config[dataloader_key] is not None:
                        batch_size = config[dataloader_key]['args'].get('batch_size', 1)
                        config[dataset_key]['args']['indices'] = list(range(3 * batch_size))

        # Run training with profiling if enabled
        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Run the training
            config['runner'](config=config).run()
            
            profiler.disable()
            
            # Save profile results with config index only if multiple configs
            profile_suffix = f"_{config_idx}" if len(config_list) > 1 else ""
            profile_path = os.path.join(config['work_dir'], f'run_profile{profile_suffix}.prof')
            stats = pstats.Stats(profiler)
            stats.sort_stats(SortKey.CUMULATIVE)
            
            # Save the raw profile data
            stats.dump_stats(profile_path)
            
            # Save a human-readable summary
            summary_path = os.path.join(config['work_dir'], f'run_profile{profile_suffix}_summary.txt')
            with open(summary_path, 'w') as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.sort_stats(SortKey.CUMULATIVE)
                stats.print_stats(50)  # Print top 50 time-consuming functions
                stats.print_callers(50)  # Print who called these functions
                stats.print_callees(50)  # Print what these functions called
        else:
            # Run training without profiling
            config['runner'](config=config).run()
