import importlib
from typing import List, Optional
import torch

from data.viewer.ivision.ivision_4d_scene_viewer import iVISION_4D_Scene_Viewer
from project.active_learning.debug_ply_dataset import DebugPLY3DSceneDataset


class iVISION_Debug_PLY_Viewer(iVISION_4D_Scene_Viewer):
    """Viewer variant that loads debugging .ply checkpoints via DebugPLY3DSceneDataset.

    Avoids parent dataset initialization to support raw .ply inputs.
    """

    def __init__(
        self,
        scene_paths: List[str],
        max_resolution: Optional[int] = None,
        max_workers: Optional[int] = None,
        device: torch.device = torch.device('cuda'),
        record_cameras_filepath: str = "./recorded_cameras.json",
        overwrite_cache: bool = False,
        show_cameras: bool = True,
    ) -> None:
        # Initialize shared state (without constructing parent's dataset)
        import os
        import dash
        from dash import html

        self.scene_paths = scene_paths
        self.max_resolution = max_resolution
        self.max_workers = max_workers
        self.device = device
        self.record_cameras_filepath = record_cameras_filepath
        self.show_cameras = show_cameras

        # Instantiate debugging dataset
        self.dataset = DebugPLY3DSceneDataset(
            scene_paths=self.scene_paths,
            data_root=None,
            split='train',
            device=self.device,
            overwrite_cache=overwrite_cache,
        )

        # Match base viewer metadata expectations
        self.scene_path_groups = [list(self.scene_paths)]
        self.group_count = 1
        self.scene_count = len(self.dataset)
        self.group_titles = [None]
        self.datasets = [self.dataset]
        self.recorded_cameras = []

        self.current_scene_idx = 0
        self._init_camera()

        # Set EGL environment as parent does
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        os.environ['EGL_DEVICE_ID'] = '0'
        os.environ['EGL_PLATFORM'] = 'surfaceless'

        # Initialize Dash app and UI
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._setup_layout()
        self._setup_callbacks()
