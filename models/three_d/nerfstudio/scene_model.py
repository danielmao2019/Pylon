import datetime as _dt
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import dash
import torch
from nerfstudio.pipelines.base_pipeline import Pipeline

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.nerfstudio import callbacks as nerfstudio_callbacks
from models.three_d.nerfstudio import states as nerfstudio_states
from models.three_d.nerfstudio.config_utils import read_data_dir_from_config_path
from models.three_d.nerfstudio.layout import build_display
from models.three_d.nerfstudio.render import render_display


class NerfstudioSceneModel(BaseSceneModel):

    def _load_model(self) -> Any:
        # Intentional lazy import to avoid package import cycle:
        # nerfstudio.__init__ -> scene_model -> splatfacto -> nerfstudio.config_utils.
        from models.three_d.nerfstudio.splatfacto.load_splatfacto import (
            load_splatfacto_model,
        )

        return load_splatfacto_model(self.resolved_path, device=self.device)

    def extract_positions(self) -> torch.Tensor:
        pipeline = self.model
        assert isinstance(pipeline, Pipeline), f"{type(pipeline)=}"
        return pipeline.model.means

    @staticmethod
    def parse_scene_path(path: str) -> str:
        """Resolve a Nerfstudio output path to a concrete training job directory.

        Supported inputs (all dirs):
        - Output dir that contains a single scene dir (any name)
        - Scene dir (ends with scene name)
        - Method dir (ends with 'splatfacto')
        - Job dir (must use default Nerfstudio timestamp format `%Y-%m-%d_%H%M%S`)

        Behavior:
        - Always descend into 'splatfacto' and pick the latest completed job dir
          (by parsed timestamp) that contains required files.
        - If starting from an output dir, first extract the single scene dir name.

        Returns the absolute path to the resolved job directory.
        """
        assert os.path.isdir(path), "Expected an existing directory"
        path = os.path.normpath(path)

        # Helper validators (defined in reverse order per request)
        def parse_default_job_timestamp(job_dir_name: str) -> Optional[_dt.datetime]:
            assert isinstance(job_dir_name, str), f"{type(job_dir_name)=}"
            try:
                return _dt.datetime.strptime(job_dir_name, "%Y-%m-%d_%H%M%S")
            except ValueError:
                return None

        def is_valid_job_dir(job_dir: str) -> bool:
            if not os.path.isdir(job_dir):
                return False
            ckpt_dir = os.path.join(job_dir, "nerfstudio_models")
            if not os.path.isdir(ckpt_dir):
                return False
            checkpoint_files = [
                entry
                for entry in os.listdir(ckpt_dir)
                if entry.startswith("step-") and entry.endswith(".ckpt")
            ]
            if len(checkpoint_files) == 0:
                return False
            required = [
                os.path.join(job_dir, "config.yml"),
                os.path.join(job_dir, "dataparser_transforms.json"),
            ]
            return all(os.path.isfile(f) for f in required)

        def is_valid_method_dir(method_dir: str) -> bool:
            if not os.path.isdir(method_dir):
                return False
            if os.path.basename(method_dir) != "splatfacto":
                return False
            child_names = os.listdir(method_dir)
            if len(child_names) == 0:
                return False
            child_paths = [
                os.path.join(method_dir, child_name) for child_name in child_names
            ]
            if not all(os.path.isdir(child_path) for child_path in child_paths):
                return False
            valid_job_dirs = [
                child_path for child_path in child_paths if is_valid_job_dir(child_path)
            ]
            return len(valid_job_dirs) > 0

        def is_valid_scene_dir(scene_dir: str) -> bool:
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
            valid_job_dirs = [
                os.path.join(current, d)
                for d in os.listdir(current)
                if os.path.isdir(os.path.join(current, d))
                and is_valid_job_dir(os.path.join(current, d))
            ]
            assert (
                len(valid_job_dirs) > 0
            ), f"Expected at least one valid job dir in '{current}'"
            if len(valid_job_dirs) == 1:
                current = valid_job_dirs[0]
            else:
                all_timestamp_named = all(
                    parse_default_job_timestamp(os.path.basename(job_dir)) is not None
                    for job_dir in valid_job_dirs
                )
                assert all_timestamp_named, (
                    f"do not know which job to pick in '{current}': "
                    "multiple valid job dirs and not all are timestamp-named"
                )
                valid_job_dirs.sort(
                    key=lambda d: parse_default_job_timestamp(os.path.basename(d)),
                    reverse=True,
                )
                current = valid_job_dirs[0]

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
        data_dir_path = read_data_dir_from_config_path(config_path=Path(config_path))
        return os.path.normpath(str(data_dir_path))

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
