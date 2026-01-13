import datetime as _dt
import os
import re
from typing import Any, List, Optional, Tuple

import dash
import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.nerfstudio import callbacks as nerfstudio_callbacks
from models.three_d.nerfstudio import states as nerfstudio_states
from models.three_d.nerfstudio.layout import build_display
from models.three_d.nerfstudio.render import render_display
from project.datasets.ivision.ivision_dataset_utils import (
    is_valid_scene_directory_name,
)
from utils.three_d.splatfacto.load_splatfacto import load_splatfacto_model


class NerfstudioSceneModel(BaseSceneModel):

    def _load_model(self) -> Any:
        return load_splatfacto_model(self.resolved_path, device=self.device)

    @staticmethod
    def parse_scene_path(path: str) -> str:
        """Resolve a Nerfstudio output path to a concrete training job directory.

        Supported inputs (all dirs):
        - Output dir that contains a single scene dir (scene name)
        - Scene dir (ends with scene name)
        - Method dir (ends with 'splatfacto')
        - Job dir (ends with timestamp 'YYYY-MM-DD_HHMMSS')

        Behavior:
        - Always descend into 'splatfacto' and pick the latest completed job dir
          (by timestamp) that contains required files.
        - If starting from an output dir, first extract the single scene dir name.

        Returns the absolute path to the resolved job directory.
        """
        assert os.path.isdir(path), "Expected an existing directory"
        path = os.path.normpath(path)

        # Helper validators (defined in reverse order per request)
        def is_valid_job_dir(job_dir: str) -> bool:
            base = os.path.basename(job_dir)
            try:
                _dt.datetime.strptime(base, "%Y-%m-%d_%H%M%S")
            except Exception:
                return False
            required = [
                os.path.join(job_dir, "nerfstudio_models", "step-000029999.ckpt"),
                os.path.join(job_dir, "config.yml"),
                os.path.join(job_dir, "dataparser_transforms.json"),
            ]
            return all(os.path.isfile(f) for f in required)

        def is_valid_method_dir(method_dir: str) -> bool:
            if os.path.basename(method_dir) != "splatfacto":
                return False
            return any(
                (lambda p: os.path.isdir(p) and is_valid_job_dir(p))(
                    os.path.join(method_dir, entry)
                )
                for entry in os.listdir(method_dir)
            )

        def is_valid_scene_dir(scene_dir: str) -> bool:
            base = os.path.basename(scene_dir)
            if not is_valid_scene_directory_name(base):
                return False
            return os.path.isdir(os.path.join(scene_dir, "splatfacto"))

        def is_valid_output_dir(output_dir: str) -> bool:
            subdirs = [
                d
                for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d))
            ]
            if len(subdirs) != 1:
                return False
            scene_subdir = subdirs[0]
            if not is_valid_scene_directory_name(scene_subdir):
                return False
            # Ensure the subdir name is a prefix of the output dir basename
            base = os.path.basename(os.path.normpath(output_dir))
            return base.startswith(scene_subdir)

        current = path

        # Case A: output dir -> descend to scene dir
        if is_valid_output_dir(current):
            only_scene = [
                d
                for d in os.listdir(current)
                if os.path.isdir(os.path.join(current, d))
            ][0]
            current = os.path.join(current, only_scene)

        # Case B: scene dir -> splatfacto
        if is_valid_scene_dir(current):
            current = os.path.join(current, "splatfacto")

        # Case C: method dir -> choose latest valid job
        if is_valid_method_dir(current):
            job_dirs = [
                d
                for d in os.listdir(current)
                if os.path.isdir(os.path.join(current, d))
                and is_valid_job_dir(os.path.join(current, d))
            ]
            # Sort descending by timestamp name
            job_dirs.sort(reverse=True)
            found = False
            for d in job_dirs:
                candidate = os.path.join(current, d)
                if is_valid_job_dir(candidate):
                    current = candidate
                    found = True
                    break
            assert found, f"Expected at least one valid job dir in '{current}'"

        # Case D: after resolution, we must be at a valid job dir
        assert is_valid_job_dir(
            current
        ), f"Expected a valid job directory, got '{current}'"

        # Only return resolved job dir
        return current

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        scene_dir = os.path.dirname(os.path.dirname(resolved_path))
        scene_name = os.path.basename(scene_dir)
        return scene_name

    @staticmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        config_path = os.path.join(resolved_path, "config.yml")
        assert os.path.isfile(config_path), f"config.yml not found: {config_path}"

        with open(config_path, 'r', encoding='utf-8') as handle:
            config_text = handle.read()

        data_pattern = (
            r'data:\s+&\w+\s+!!python/object/apply:pathlib\.PosixPath\s*\n'
            r'((?:\s*-\s+.+\n)+)'
        )
        match = re.search(data_pattern, config_text)
        assert match, f"Could not locate data path in config: {config_path}"

        components_text = match.group(1)
        components: List[str] = []
        for line in components_text.strip().split('\n'):
            comp_match = re.match(r'\s*-\s+(.+)', line.strip())
            if comp_match:
                components.append(comp_match.group(1))

        assert components, f"No path components found in config: {config_path}"

        data_path = os.path.join(*components)
        data_path = os.path.normpath(data_path)

        return data_path

    @staticmethod
    def register_callbacks(
        dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
    ) -> None:
        nerfstudio_callbacks.register_callbacks(
            dataset=dataset,
            app=app,
            viewer=viewer,
        )

    @staticmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        nerfstudio_states.setup_states(app=app)

    def display_render(
        self,
        camera: Camera,
        resolution: Tuple[int, int],
        camera_name: Optional[str] = None,
        display_cameras: Optional[List[Camera]] = None,
        title: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Any:
        """Render and display a Splatfacto (NerfStudio) scene."""
        assert isinstance(camera, Camera), f"{type(camera)=}"
        target_camera_name = camera_name if camera_name is not None else camera.name
        render_outputs = render_display(
            scene_model=self,
            camera=camera,
            resolution=resolution,
            camera_name=target_camera_name,
            display_cameras=display_cameras,
            title=title,
            device=device,
        )
        return build_display(render_outputs)
