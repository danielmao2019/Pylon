from typing import List
import dash
import os
project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(project_root)
import sys
sys.path.append(project_root)
os.chdir(project_root)

from runners.eval_viewer.layouts.main_layout import create_layout
from runners.eval_viewer.callbacks.update_plots import register_callbacks
from runners.eval_viewer.callbacks.datapoint_viewer import register_datapoint_viewer_callbacks
from runners.eval_viewer.backend.initialization import initialize_log_dirs
from runners.eval_viewer.backend.datapoint_viewer import DatapointViewer
from data.viewer.managers.dataset_manager import DatasetManager


def create_app(log_dirs: List[str], force_reload: bool = False) -> dash.Dash:
    """Create the Dash application.

    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force reload of cached data

    Returns:
        app: Dash application instance
    """
    # Initialize log directories
    max_epochs, metric_names, dataset_class, dataset_type, log_dir_infos = initialize_log_dirs(log_dirs, force_reload)

    # Initialize dataset manager
    dataset_manager = DatasetManager()
    dataset_manager.load_dataset(dataset_class)

    # Create datapoint viewer
    datapoint_viewer = DatapointViewer(dataset_manager, dataset_class, dataset_type)

    # Create app
    app = dash.Dash(__name__)

    # Create layout
    app.layout = create_layout(max_epochs, metric_names, len(log_dirs))

    # Register callbacks
    register_callbacks(app, metric_names, log_dir_infos)
    register_datapoint_viewer_callbacks(app, datapoint_viewer)
    return app


def run_app(log_dirs: List[str], force_reload: bool = False, debug: bool = True, port: int = 8050):
    """Run the Dash application.

    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force reload of cached data
        debug: Whether to run in debug mode
        port: Port to run the server on
    """
    app = create_app(log_dirs, force_reload)
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    parser.add_argument("--force-reload", action="store_true", help="Force recreation of cache")
    args = parser.parse_args()

    log_dirs = [
        # "./logs/benchmarks/change_detection/air_change/BiFA_run_0",
        # "./logs/benchmarks/change_detection/air_change/CDMaskFormer_run_0",
        # "./logs/benchmarks/change_detection/air_change/CDXFormer_run_0",
        # "./logs/benchmarks/change_detection/air_change/CSA_CDGAN_run_0",
        # "./logs/benchmarks/change_detection/air_change/ChangeMamba-Base_run_0",
        # "./logs/benchmarks/change_detection/air_change/ChangeMamba-Small_run_0",
        # "./logs/benchmarks/change_detection/air_change/ChangeMamba-Tiny_run_0",
        # "./logs/benchmarks/change_detection/air_change/ChangeNextV1_run_0",
        # "./logs/benchmarks/change_detection/air_change/ChangeNextV2_run_0",
        # "./logs/benchmarks/change_detection/air_change/ChangeNextV3_run_0",
        # "./logs/benchmarks/change_detection/air_change/Changer-mit-b0_run_0",
        # "./logs/benchmarks/change_detection/air_change/Changer-mit-b1_run_0",
        # "./logs/benchmarks/change_detection/air_change/Changer-r18_run_0",
        # "./logs/benchmarks/change_detection/air_change/Changer-s101_run_0",
        # "./logs/benchmarks/change_detection/air_change/Changer-s50_run_0",
        # "./logs/benchmarks/change_detection/air_change/DSAMNet_run_0",
        # "./logs/benchmarks/change_detection/air_change/DSIFN_run_0",
        # "./logs/benchmarks/change_detection/air_change/DsferNet_run_0",
        # "./logs/benchmarks/change_detection/air_change/FC-EF_run_0",
        # "./logs/benchmarks/change_detection/air_change/FC-Siam-conc_run_0",
        # "./logs/benchmarks/change_detection/air_change/FC-Siam-diff_run_0",
        # "./logs/benchmarks/change_detection/air_change/FTN_run_0",
        # "./logs/benchmarks/change_detection/air_change/HANet_run_0",
        # "./logs/benchmarks/change_detection/air_change/HCGMNet_run_0",
        # "./logs/benchmarks/change_detection/air_change/RFL_CDNet_run_0",
        # "./logs/benchmarks/change_detection/air_change/SNUNet_ECAM_run_0",
        # "./logs/benchmarks/change_detection/air_change/SRCNet_run_0",
        # "./logs/benchmarks/change_detection/air_change/TinyCD_run_0",
        # "./logs/benchmarks/change_detection/cdd/BiFA_run_0",
        # "./logs/benchmarks/change_detection/cdd/CDMaskFormer_run_0",
        # "./logs/benchmarks/change_detection/cdd/CDXFormer_run_0",
        # "./logs/benchmarks/change_detection/cdd/CSA_CDGAN_run_0",
        # "./logs/benchmarks/change_detection/cdd/ChangeMamba-Base_run_0",
        # "./logs/benchmarks/change_detection/cdd/ChangeMamba-Small_run_0",
        # "./logs/benchmarks/change_detection/cdd/ChangeMamba-Tiny_run_0",
        # "./logs/benchmarks/change_detection/cdd/ChangeNextV1_run_0",
        # "./logs/benchmarks/change_detection/cdd/ChangeNextV2_run_0",
        # "./logs/benchmarks/change_detection/cdd/ChangeNextV3_run_0",
        # "./logs/benchmarks/change_detection/cdd/Changer-mit-b0_run_0",
        # "./logs/benchmarks/change_detection/cdd/Changer-mit-b1_run_0",
        # "./logs/benchmarks/change_detection/cdd/Changer-r18_run_0",
        # "./logs/benchmarks/change_detection/cdd/Changer-s101_run_0",
        # "./logs/benchmarks/change_detection/cdd/Changer-s50_run_0",
        # "./logs/benchmarks/change_detection/cdd/DSAMNet_run_0",
        # "./logs/benchmarks/change_detection/cdd/DSIFN_run_0",
        # "./logs/benchmarks/change_detection/cdd/DsferNet_run_0",
        # "./logs/benchmarks/change_detection/cdd/FC-EF_run_0",
        # "./logs/benchmarks/change_detection/cdd/FC-Siam-conc_run_0",
        # "./logs/benchmarks/change_detection/cdd/FC-Siam-diff_run_0",
        # "./logs/benchmarks/change_detection/cdd/FTN_run_0",
        # "./logs/benchmarks/change_detection/cdd/HANet_run_0",
        # "./logs/benchmarks/change_detection/cdd/HCGMNet_run_0",
        # "./logs/benchmarks/change_detection/cdd/RFL_CDNet_run_0",
        # "./logs/benchmarks/change_detection/cdd/SNUNet_ECAM_run_0",
        # "./logs/benchmarks/change_detection/cdd/SRCNet_run_0",
        # "./logs/benchmarks/change_detection/cdd/TinyCD_run_0",
        "./logs/benchmarks/change_detection/levir_cd/BiFA_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/CDMaskFormer_run_0",
        "./logs/benchmarks/change_detection/levir_cd/CDXFormer_run_0",
        "./logs/benchmarks/change_detection/levir_cd/CSA_CDGAN_run_0",
        "./logs/benchmarks/change_detection/levir_cd/ChangeMamba-Base_run_0",
        "./logs/benchmarks/change_detection/levir_cd/ChangeMamba-Small_run_0",
        "./logs/benchmarks/change_detection/levir_cd/ChangeMamba-Tiny_run_0",
        "./logs/benchmarks/change_detection/levir_cd/ChangeNextV1_run_0",
        "./logs/benchmarks/change_detection/levir_cd/ChangeNextV2_run_0",
        "./logs/benchmarks/change_detection/levir_cd/ChangeNextV3_run_0",
        "./logs/benchmarks/change_detection/levir_cd/Changer-mit-b0_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/Changer-mit-b1_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/Changer-r18_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/Changer-s101_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/Changer-s50_run_0",
        "./logs/benchmarks/change_detection/levir_cd/DSAMNet_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/DSIFN_run_0",
        "./logs/benchmarks/change_detection/levir_cd/DsferNet_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/FC-EF_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/FC-Siam-conc_run_0",
        "./logs/benchmarks/change_detection/levir_cd/FC-Siam-diff_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/FTN_run_0",
        "./logs/benchmarks/change_detection/levir_cd/HANet_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/HCGMNet_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/RFL_CDNet_run_0",
        "./logs/benchmarks/change_detection/levir_cd/SNUNet_ECAM_run_0",
        "./logs/benchmarks/change_detection/levir_cd/SRCNet_run_0",
        # "./logs/benchmarks/change_detection/levir_cd/TinyCD_run_0",
        # "./logs/benchmarks/change_detection/oscd/BiFA_run_0",
        # "./logs/benchmarks/change_detection/oscd/CDMaskFormer_run_0",
        # "./logs/benchmarks/change_detection/oscd/CDXFormer_run_0",
        # "./logs/benchmarks/change_detection/oscd/CSA_CDGAN_run_0",
        # "./logs/benchmarks/change_detection/oscd/ChangeMamba-Base_run_0",
        # "./logs/benchmarks/change_detection/oscd/ChangeMamba-Small_run_0",
        # "./logs/benchmarks/change_detection/oscd/ChangeMamba-Tiny_run_0",
        # "./logs/benchmarks/change_detection/oscd/ChangeNextV1_run_0",
        # "./logs/benchmarks/change_detection/oscd/ChangeNextV2_run_0",
        # "./logs/benchmarks/change_detection/oscd/ChangeNextV3_run_0",
        # "./logs/benchmarks/change_detection/oscd/Changer-mit-b0_run_0",
        # "./logs/benchmarks/change_detection/oscd/Changer-mit-b1_run_0",
        # "./logs/benchmarks/change_detection/oscd/Changer-r18_run_0",
        # "./logs/benchmarks/change_detection/oscd/Changer-s101_run_0",
        # "./logs/benchmarks/change_detection/oscd/Changer-s50_run_0",
        # "./logs/benchmarks/change_detection/oscd/DSAMNet_run_0",
        # "./logs/benchmarks/change_detection/oscd/DSIFN_run_0",
        # "./logs/benchmarks/change_detection/oscd/DsferNet_run_0",
        # "./logs/benchmarks/change_detection/oscd/FC-EF_run_0",
        # "./logs/benchmarks/change_detection/oscd/FC-Siam-conc_run_0",
        # "./logs/benchmarks/change_detection/oscd/FC-Siam-diff_run_0",
        # "./logs/benchmarks/change_detection/oscd/FTN_run_0",
        # "./logs/benchmarks/change_detection/oscd/HANet_run_0",
        # "./logs/benchmarks/change_detection/oscd/HCGMNet_run_0",
        # "./logs/benchmarks/change_detection/oscd/RFL_CDNet_run_0",
        # "./logs/benchmarks/change_detection/oscd/SNUNet_ECAM_run_0",
        # "./logs/benchmarks/change_detection/oscd/SRCNet_run_0",
        # "./logs/benchmarks/change_detection/oscd/TinyCD_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/BiFA_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/CDMaskFormer_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/CDXFormer_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/CSA_CDGAN_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/ChangeMamba-Base_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/ChangeMamba-Small_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/ChangeMamba-Tiny_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/ChangeNextV1_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/ChangeNextV2_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/ChangeNextV3_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/Changer-mit-b0_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/Changer-mit-b1_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/Changer-r18_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/Changer-s101_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/Changer-s50_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/DSAMNet_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/DSIFN_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/DsferNet_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/FC-EF_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/FC-Siam-conc_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/FC-Siam-diff_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/FTN_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/HANet_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/HCGMNet_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/RFL_CDNet_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/SNUNet_ECAM_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/SRCNet_run_0",
        # "./logs/benchmarks/change_detection/sysu_cd/TinyCD_run_0",
        # "./logs/benchmarks/change_detection/change_star_v1/xview2_run_0",
        # "./logs/benchmarks/change_detection/ppsl/whu_bd_run_0",
    ]

    run_app(log_dirs=log_dirs, debug=args.debug, port=args.port, force_reload=args.force_reload)
