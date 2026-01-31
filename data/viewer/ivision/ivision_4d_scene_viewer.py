import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.nerfstudio.nerfstudio import NerfStudio
from data.viewer.ivision.callbacks import register_viewer_callbacks
from data.viewer.ivision.layout import (
    CAMERA_NONE_VALUE,
    MODEL_STORE_CONTAINER_ID,
    build_layout,
)
from models.three_d.base import BaseSceneModel
from project.datasets.ivision.ivision_3d_scene_dataset import (
    iVISION_3D_Scene_Dataset,
)
from utils.three_d.rotation.pitch_yaw import matrix_to_pitch_yaw, pitch_yaw_to_matrix


class iVISION_4D_Scene_Viewer:
    """Dash-based viewer for comparing multiple methods within a dataset/scene."""

    def __init__(
        self,
        registry: Dict[str, Dict[str, Dict[str, str]]],
        max_resolution: Optional[int] = None,
        max_workers: Optional[int] = None,
        device: str = "cuda",
        record_cameras_filepath: str = "./recorded_cameras.json",
        overwrite_cache: bool = False,
    ) -> None:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["EGL_DEVICE_ID"] = "0"
        os.environ["EGL_PLATFORM"] = "surfaceless"
        self.registry = registry
        self.max_resolution = max_resolution
        self.max_workers = max_workers
        self.device = torch.device(device)
        self.record_cameras_filepath = record_cameras_filepath
        self.overwrite_cache = overwrite_cache
        self._init_app()

    # --- App initialization ---

    def _init_app(self) -> None:
        self._init_camera_state()
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._init_all_datasets(registry=self.registry)
        dataset_options = [
            {"label": name, "value": name} for name in self.dataset_order
        ]
        build_layout(app=self.app, dataset_options=dataset_options)
        self._setup_states()
        self._build_model_init_layouts()
        self._register_callbacks()

    def _init_camera_state(self) -> None:
        self._current_camera_selection: Optional[str] = None
        self.position = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.rotation = torch.zeros(2, device=self.device, dtype=torch.float32)
        self._dataset_initial_center: Dict[str, torch.Tensor] = {}
        self.recorded_cameras: List[List[float]] = []

    def _init_all_datasets(
        self, registry: Dict[str, Dict[str, Dict[str, str]]]
    ) -> None:
        assert registry, "Registry must be populated before initialization"
        self.dataset_order: List[str] = list(registry.keys())
        self.scene_order: Dict[str, List[str]] = {
            dataset: list(scene_map.keys()) for dataset, scene_map in registry.items()
        }
        self.method_order: Dict[str, Dict[str, List[str]]] = {
            dataset: {
                scene: list(method_map.keys())
                for scene, method_map in scene_map.items()
            }
            for dataset, scene_map in registry.items()
        }
        self.dataset_cache: Dict[str, Dict[str, iVISION_3D_Scene_Dataset]] = {}
        self.method_index: Dict[str, Dict[str, Dict[str, int]]] = {}
        for dataset_name, scene_map in registry.items():
            self.dataset_cache[dataset_name] = {}
            self.method_index[dataset_name] = {}

            def _load_scene(
                scene_name: str, method_map: Dict[str, str]
            ) -> Tuple[str, iVISION_3D_Scene_Dataset, Dict[str, int]]:
                method_names = list(method_map.keys())
                scene_paths = [method_map[method] for method in method_names]
                dataset_instance = iVISION_3D_Scene_Dataset(
                    scene_paths=scene_paths,
                    data_root=None,
                    device=self.device,
                    overwrite_cache=self.overwrite_cache,
                )
                index_map = {method: idx for idx, method in enumerate(method_names)}
                return scene_name, dataset_instance, index_map

            max_workers = min(
                len(scene_map),
                max(
                    1,
                    (
                        self.max_workers
                        if self.max_workers is not None
                        else int(os.cpu_count() or 1)
                    ),
                ),
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for scene_name, method_map in scene_map.items():
                    futures.append(executor.submit(_load_scene, scene_name, method_map))
                for future in futures:
                    scene_name, dataset_instance, index_map = future.result()
                    self.dataset_cache[dataset_name][scene_name] = dataset_instance
                    self.method_index[dataset_name][scene_name] = index_map
        self.current_dataset = None
        self.current_scene = None
        self._datapoint_cache: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    def _setup_states(self) -> None:
        layout = self.app.layout
        assert hasattr(layout, "children"), "app layout must expose children"
        layout_children = layout.children
        assert isinstance(layout_children, list), "app layout children must be a list"
        assert (
            len(layout_children) >= 2
        ), "app layout must include model store container as second child"
        container = layout_children[1]
        assert getattr(container, "id", None) == MODEL_STORE_CONTAINER_ID
        container.children = []
        for dataset_name, scene_map in self.dataset_cache.items():
            for scene_name, dataset_instance in scene_map.items():
                method_names = self.method_order[dataset_name][scene_name]
                dataset_instance.setup_states(
                    app=self.app,
                    dataset_name=dataset_name,
                    scene_name=scene_name,
                    method_names=method_names,
                )

    def _register_callbacks(self) -> None:
        for dataset_name, scene_map in self.dataset_cache.items():
            for scene_name, dataset_instance in scene_map.items():
                dataset_instance.register_callbacks(
                    app=self.app,
                    viewer=self,
                    dataset_name=dataset_name,
                    scene_name=scene_name,
                )
        register_viewer_callbacks(app=self.app, viewer=self)

    def _build_model_init_layouts(self) -> None:
        self._static_model_layouts = {}
        for dataset_name, scene_map in self.dataset_cache.items():
            self._static_model_layouts[dataset_name] = {}
            for scene_name, dataset_instance in scene_map.items():
                self._static_model_layouts[dataset_name][scene_name] = {}
                method_names = self.method_order[dataset_name][scene_name]

                def _build_static(method_name: str) -> Tuple[str, Any]:
                    method_idx = self.method_index[dataset_name][scene_name][
                        method_name
                    ]
                    datapoint = dataset_instance[method_idx]
                    scene_model = datapoint["inputs"]["model"]
                    container = scene_model.build_static_container(
                        dataset_name=dataset_name,
                        scene_name=scene_name,
                        method_name=method_name,
                        debugger_enabled=False,
                    )
                    return method_name, container

                max_workers = min(
                    len(method_names),
                    max(
                        1,
                        (
                            self.max_workers
                            if self.max_workers is not None
                            else int(os.cpu_count() or 1)
                        ),
                    ),
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for method_name in method_names:
                        futures.append(executor.submit(_build_static, method_name))
                    for future in futures:
                        method_name, container = future.result()
                        self._static_model_layouts[dataset_name][scene_name][
                            method_name
                        ] = container

    def run(self, host: str = "0.0.0.0", port: int = 8050) -> None:
        self.app.run(debug=False, host=host, port=port)

    # --- Getters and setters ---

    def set_dataset(self, dataset_name: Optional[str]) -> None:
        if dataset_name == self.current_dataset:
            return
        self._clear_scene_method_cache()
        self.current_dataset = dataset_name
        self.current_scene = None
        self._current_camera_selection = None

    def set_scene(self, scene_name: str) -> None:
        assert self.current_dataset is not None, "Dataset must be selected before scene"
        assert (
            scene_name in self.scene_order[self.current_dataset]
        ), f"Unknown scene {scene_name} for dataset {self.current_dataset}"
        if scene_name != self.current_scene:
            self._clear_scene_method_cache()
        self.current_scene = scene_name
        self._current_camera_selection = None
        if self.current_dataset not in self._dataset_initial_center:
            reference_datapoint = self.get_datapoint(
                dataset_name=self.current_dataset,
                scene_name=scene_name,
                method_name=self.get_reference_method(),
            )
            meta_info = reference_datapoint["meta_info"]
            position_mean = torch.tensor(
                meta_info["position_mean"], dtype=torch.float32, device=self.device
            )
            self._dataset_initial_center[self.current_dataset] = position_mean
        self.set_camera()

    def set_scene_by_offset(self, offset: int) -> None:
        assert self.current_dataset is not None, "Dataset must be selected"
        scene_names = self.scene_order[self.current_dataset]
        if self.current_scene is None:
            target_idx = 0
        else:
            current_idx = scene_names.index(self.current_scene)
            target_idx = max(0, min(current_idx + offset, len(scene_names) - 1))
        self.set_scene(scene_names[target_idx])

    def set_camera(self, camera_selection: Optional[str] = None) -> None:
        reference_annotation = self.get_reference_annotation()
        assert (
            "nerfstudio_data" in reference_annotation
        ), f"Annotation missing nerfstudio_data key, keys={list(reference_annotation.keys())}"
        nerfstudio_data = reference_annotation["nerfstudio_data"]
        assert isinstance(
            nerfstudio_data, NerfStudio
        ), f"nerfstudio_data must be NerfStudio, got {type(nerfstudio_data)}"
        if camera_selection is None:
            if nerfstudio_data.train_filenames is not None:
                assert (
                    nerfstudio_data.train_filenames
                ), "train_filenames must be non-empty"
                default_split = "train"
                file_path = nerfstudio_data.train_filenames[0]
            else:
                assert (
                    nerfstudio_data.filenames
                ), "transforms filenames must be non-empty"
                default_split = "all"
                file_path = nerfstudio_data.filenames[0]
            camera_selection = self._build_camera_selection(
                split_name=default_split,
                file_path=file_path,
            )

        self._current_camera_selection = camera_selection
        assert (
            ":" in camera_selection
        ), f"Camera selection '{camera_selection}' is missing ':' separator"
        split_key, file_path = camera_selection.split(":", 1)

        camera_name = Path(file_path).stem
        assert camera_name, f"Empty camera name parsed from '{file_path}'"
        camera = nerfstudio_data.cameras[camera_name]
        c2w_standard = camera.to(device=self.device, convention="standard").extrinsics
        self.position = c2w_standard[:3, 3].clone()
        rotation_matrix = c2w_standard[:3, :3]
        pitch_yaw = matrix_to_pitch_yaw(rotation_matrix)
        self.rotation = torch.rad2deg(pitch_yaw).to(
            dtype=torch.float32, device=self.device
        )

    def set_camera_by_selection(self, selection_key: Optional[str]) -> None:
        if selection_key is None or selection_key == CAMERA_NONE_VALUE:
            self.set_novel_view_state()
            return
        self.set_camera(camera_selection=selection_key)

    def set_pose_from_keyboard(
        self,
        key: str,
        translation_step: float,
        rotation_step: float,
    ) -> None:
        assert key, "keyboard key must be non-empty"
        assert (
            self.current_dataset is not None and self.current_scene is not None
        ), "Dataset and scene must be selected for keyboard actions"
        rotation_matrix = pitch_yaw_to_matrix(torch.deg2rad(self.rotation))
        forward_axis = rotation_matrix[:, 1]
        right_axis = rotation_matrix[:, 0]

        forward = torch.nn.functional.normalize(forward_axis, dim=0)
        horizontal_forward = torch.nn.functional.normalize(
            torch.tensor(
                [forward_axis[0], forward_axis[1], 0.0],
                dtype=torch.float32,
                device=self.device,
            ),
            dim=0,
        )
        right = torch.nn.functional.normalize(right_axis, dim=0)

        if key == 'w':
            self.position += horizontal_forward * translation_step
            self.set_novel_view_state()
        elif key == 's':
            self.position -= horizontal_forward * translation_step
            self.set_novel_view_state()
        elif key == 'a':
            self.position -= right * translation_step
            self.set_novel_view_state()
        elif key == 'd':
            self.position += right * translation_step
            self.set_novel_view_state()
        elif key == 'f':
            self.position += forward * translation_step
            self.set_novel_view_state()
        elif key == 'r':
            self.position -= forward * translation_step
            self.set_novel_view_state()
        elif key == ' ':
            self.position[2] += translation_step
            self.set_novel_view_state()
        elif key == 'Shift':
            self.position[2] -= translation_step
            self.set_novel_view_state()
        elif key == 'ArrowUp':
            self.rotation[0] += rotation_step
            self.set_novel_view_state()
        elif key == 'ArrowDown':
            self.rotation[0] -= rotation_step
            self.set_novel_view_state()
        elif key == 'ArrowLeft':
            self.rotation[1] += rotation_step
            self.set_novel_view_state()
        elif key == 'ArrowRight':
            self.rotation[1] -= rotation_step
            self.set_novel_view_state()

        self.rotation[0] = torch.clamp(self.rotation[0], -80, +80)

    def set_novel_view_state(self) -> Optional[str]:
        self._current_camera_selection = None
        reference_annotation = self.get_reference_annotation()
        assert (
            "nerfstudio_data" in reference_annotation
        ), f"Annotation missing nerfstudio_data key, keys={list(reference_annotation.keys())}"
        nerfstudio_data = reference_annotation["nerfstudio_data"]
        assert isinstance(
            nerfstudio_data, NerfStudio
        ), f"nerfstudio_data must be NerfStudio, got {type(nerfstudio_data)}"
        split_lookup: Dict[str, str] = {}
        if nerfstudio_data.train_filenames is not None:
            for file_path in nerfstudio_data.train_filenames:
                split_lookup[file_path] = "train"
            for file_path in nerfstudio_data.val_filenames:
                split_lookup[file_path] = "val"
            for file_path in nerfstudio_data.test_filenames:
                if file_path not in split_lookup:
                    split_lookup[file_path] = "test"
        else:
            for file_path in nerfstudio_data.filenames:
                split_lookup[file_path] = "all"
        current_pose = self.get_camera().extrinsics
        for file_path in nerfstudio_data.filenames:
            assert (
                file_path in split_lookup
            ), f"File '{file_path}' missing split assignment"
            camera_name = Path(file_path).stem
            assert camera_name, f"Empty camera name parsed from '{file_path}'"
            candidate_pose = (
                nerfstudio_data.cameras[camera_name]
                .to(device=self.device, convention="standard")
                .extrinsics
            )
            if torch.allclose(candidate_pose, current_pose, atol=1e-4, rtol=1e-4):
                self._current_camera_selection = self._build_camera_selection(
                    split_name=split_lookup[file_path],
                    file_path=file_path,
                )
                return self._current_camera_selection
        return None

    def get_render_current_scene(
        self,
        model_state: Dict[str, Dict[str, Dict[str, Any]]],
        show_cameras: bool,
    ) -> Dict[str, Any]:
        assert isinstance(model_state, dict), f"{type(model_state)=}"
        assert isinstance(show_cameras, bool), f"{type(show_cameras)=}"
        if self.current_dataset is None or self.current_scene is None:
            return {}

        reference_annotation = self.get_reference_annotation()
        assert (
            "nerfstudio_data" in reference_annotation
        ), f"Annotation missing nerfstudio_data key, keys={list(reference_annotation.keys())}"
        nerfstudio_data = reference_annotation["nerfstudio_data"]
        assert isinstance(
            nerfstudio_data, NerfStudio
        ), f"nerfstudio_data must be NerfStudio, got {type(nerfstudio_data)}"
        reference_resolution = nerfstudio_data.resolution
        render_resolution = self.get_render_resolution(reference_resolution)
        display_cameras = (
            self.get_scene_camera_overlays(annotation=reference_annotation)
            if show_cameras
            else None
        )

        camera_name = None
        if self._current_camera_selection is not None:
            assert (
                ":" in self._current_camera_selection
            ), f"Camera selection '{self._current_camera_selection}' is missing ':' separator"
            camera_name = self._current_camera_selection.split(":", 1)[1]

        method_names = self.method_order[self.current_dataset][self.current_scene]
        dataset_name = self.current_dataset
        scene_name = self.current_scene
        assert dataset_name is not None and scene_name is not None
        render_camera = self.get_camera()

        def _render_single(method_name: str) -> Tuple[str, Any]:
            datapoint = self.get_datapoint(
                dataset_name=dataset_name,
                scene_name=scene_name,
                method_name=method_name,
            )
            scene_model = datapoint["inputs"]["model"]
            method_states = (
                model_state.get(dataset_name, {})
                .get(scene_name, {})
                .get(method_name, {})
            )
            state_for_method = dict(method_states or {})
            state_for_method["method"] = method_name
            state_for_method["dataset_name"] = dataset_name
            state_for_method["scene_name"] = scene_name
            body = scene_model.display_render(
                camera=render_camera,
                resolution=render_resolution,
                camera_name=camera_name,
                display_cameras=display_cameras,
                title=method_name,
                device=self.device,
                **state_for_method,
            )
            return method_name, body

        max_workers = min(
            len(method_names),
            max(
                1,
                (
                    self.max_workers
                    if self.max_workers is not None
                    else int(os.cpu_count() or 1)
                ),
            ),
        )

        bodies: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_render_single, method_name)
                for method_name in method_names
            ]
            for future in futures:
                method_name, body = future.result()
                bodies[method_name] = body

        return bodies

    def get_datapoint(
        self, dataset_name: Optional[str], scene_name: Optional[str], method_name: str
    ) -> Dict[str, Dict[str, Any]]:
        assert dataset_name is not None and scene_name is not None
        if (
            dataset_name in self._datapoint_cache
            and scene_name in self._datapoint_cache[dataset_name]
            and method_name in self._datapoint_cache[dataset_name][scene_name]
        ):
            return self._datapoint_cache[dataset_name][scene_name][method_name]

        assert (
            dataset_name in self.dataset_cache
            and scene_name in self.dataset_cache[dataset_name]
        ), f"Dataset {dataset_name} scene {scene_name} not initialized"
        dataset_instance = self.dataset_cache[dataset_name][scene_name]
        method_idx = self.method_index[dataset_name][scene_name][method_name]
        datapoint = dataset_instance[method_idx]
        assert "inputs" in datapoint and "model" in datapoint["inputs"]
        model = datapoint["inputs"]["model"]
        assert isinstance(model, BaseSceneModel)

        self._datapoint_cache.setdefault(dataset_name, {}).setdefault(scene_name, {})[
            method_name
        ] = datapoint
        return datapoint

    def get_reference_method(self) -> str:
        assert self.current_dataset is not None and self.current_scene is not None
        return self.method_order[self.current_dataset][self.current_scene][0]

    def get_reference_annotation(self) -> Dict[str, Any]:
        reference_dp = self.get_datapoint(
            dataset_name=self.current_dataset,
            scene_name=self.current_scene,
            method_name=self.get_reference_method(),
        )
        return reference_dp["meta_info"]

    def get_camera(self) -> Camera:
        annotation = self.get_reference_annotation()
        assert (
            "nerfstudio_data" in annotation
        ), f"Annotation missing nerfstudio_data key, keys={list(annotation.keys())}"
        nerfstudio_data = annotation["nerfstudio_data"]
        assert isinstance(
            nerfstudio_data, NerfStudio
        ), f"nerfstudio_data must be NerfStudio, got {type(nerfstudio_data)}"
        intrinsics = nerfstudio_data.intrinsics.to(self.device)
        rotation_matrix = pitch_yaw_to_matrix(torch.deg2rad(self.rotation))
        c2w = torch.eye(4, dtype=rotation_matrix.dtype, device=self.device)
        c2w[:3, :3] = rotation_matrix
        c2w[:3, 3] = self.position
        camera_name: Optional[str] = None
        camera_id: Optional[int] = None
        if self._current_camera_selection is not None:
            assert ":" in self._current_camera_selection
            _, file_path = self._current_camera_selection.split(":", 1)
            camera_name = Path(file_path).stem
            camera = nerfstudio_data.cameras[camera_name]
            camera_id = camera.id
        return Camera(
            intrinsics=intrinsics,
            extrinsics=c2w,
            convention="standard",
            name=camera_name,
            id=camera_id,
            device=self.device,
        )

    def get_translation_max_step(self) -> float:
        reference_dp = self.get_datapoint(
            dataset_name=self.current_dataset,
            scene_name=self.current_scene,
            method_name=self.get_reference_method(),
        )
        meta_info = reference_dp["meta_info"]
        position_min = meta_info["position_min"]
        position_max = meta_info["position_max"]
        diffs = [b - a for a, b in zip(position_min, position_max)]
        squared = [diff * diff for diff in diffs]
        diagonal = math.sqrt(sum(squared))
        assert diagonal > 0.0, "Scene diagonal must be positive"
        return diagonal

    def get_rotation_max_step(self) -> float:
        return 90.0

    def get_translation_step(self, slider_value: Optional[float]) -> float:
        normalized = 0.1 if slider_value is None else float(slider_value)
        max_step = self.get_translation_max_step()
        return self._compute_step_size(normalized=normalized, max_step=max_step)

    def get_rotation_step(self, slider_value: Optional[float]) -> float:
        normalized = 0.1 if slider_value is None else float(slider_value)
        max_step = self.get_rotation_max_step()
        return self._compute_step_size(normalized=normalized, max_step=max_step)

    def get_camera_info(self) -> Dict[str, Any]:
        camera = self.get_camera()
        intrinsics = camera.intrinsics.detach().cpu()
        reference_annotation = self.get_reference_annotation()
        assert (
            "nerfstudio_data" in reference_annotation
        ), f"Annotation missing nerfstudio_data key, keys={list(reference_annotation.keys())}"
        nerfstudio_data = reference_annotation["nerfstudio_data"]
        assert isinstance(
            nerfstudio_data, NerfStudio
        ), f"nerfstudio_data must be NerfStudio, got {type(nerfstudio_data)}"
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        assert (
            fx > 0.0 and fy > 0.0
        ), "Camera intrinsics must have positive focal lengths"

        frustum_width = max(0.0, 2.0 * cx)
        frustum_height = max(0.0, 2.0 * cy)
        frustum_height_px, frustum_width_px = nerfstudio_data.resolution

        fov_x = math.degrees(2.0 * math.atan2(frustum_width, 2.0 * fx))
        fov_y = math.degrees(2.0 * math.atan2(frustum_height, 2.0 * fy))

        position = self.position.detach().cpu().numpy()
        pitch = float(self.rotation[0].item())
        yaw = float(self.rotation[1].item())

        return {
            "fov_x": fov_x,
            "fov_y": fov_y,
            "frustum_resolution": (int(frustum_width_px), int(frustum_height_px)),
            "position": (float(position[0]), float(position[1]), float(position[2])),
            "pitch": pitch,
            "yaw": yaw,
        }

    def get_scene_camera_overlays(
        self, annotation: Dict[str, Any]
    ) -> Optional[List[Camera]]:
        assert (
            "nerfstudio_data" in annotation
        ), f"Annotation missing nerfstudio_data key, keys={list(annotation.keys())}"
        nerfstudio_data = annotation["nerfstudio_data"]
        assert isinstance(
            nerfstudio_data, NerfStudio
        ), f"nerfstudio_data must be NerfStudio, got {type(nerfstudio_data)}"
        display_cameras: List[Camera] = []
        for camera in nerfstudio_data.cameras:
            display_cameras.append(camera.to(device=self.device, convention="standard"))

        if not display_cameras:
            return None

        return display_cameras

    def get_camera_selector_options(self) -> Optional[Dict[str, Any]]:
        if self.current_dataset is None or self.current_scene is None:
            return None
        annotation = self.get_reference_annotation()
        assert (
            "nerfstudio_data" in annotation
        ), f"Annotation missing nerfstudio_data key, keys={list(annotation.keys())}"
        nerfstudio_data = annotation["nerfstudio_data"]
        assert isinstance(
            nerfstudio_data, NerfStudio
        ), f"nerfstudio_data must be NerfStudio, got {type(nerfstudio_data)}"
        splits = self._build_camera_splits(
            nerfstudio_data=nerfstudio_data,
            scene_name=annotation["scene_name"],
        )

        entries: Dict[str, List[Dict[str, Any]]] = {}
        for split_name, file_list in splits.items():
            entries[split_name] = [
                {
                    "label": file_path,
                    "value": self._build_camera_selection(split_name, file_path),
                }
                for file_path in file_list
            ]

        return {
            "dataset": self.current_dataset,
            "scene": self.current_scene,
            "splits": entries,
            "selection": self._current_camera_selection,
        }

    def get_scene_index(self) -> Optional[int]:
        if self.current_dataset is None or self.current_scene is None:
            return None
        return self.scene_order[self.current_dataset].index(self.current_scene)

    def get_scene_count(self) -> int:
        if self.current_dataset is None:
            return 0
        return len(self.scene_order[self.current_dataset])

    def get_render_resolution(
        self, original_resolution: Tuple[int, int]
    ) -> Tuple[int, int]:
        orig_h, orig_w = original_resolution
        assert orig_h > 0 and orig_w > 0, "Original resolution must be positive"

        if self.max_resolution is None:
            return orig_h, orig_w

        orig_pixels = orig_h * orig_w
        if orig_pixels <= self.max_resolution:
            return orig_h, orig_w

        scale = math.sqrt(self.max_resolution / float(orig_pixels))
        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))
        return new_h, new_w

    # --- Helper methods ---

    @staticmethod
    def _build_camera_splits(
        nerfstudio_data: NerfStudio, scene_name: str
    ) -> Dict[str, List[str]]:
        if nerfstudio_data.train_filenames is not None:
            assert nerfstudio_data.val_filenames is not None
            assert nerfstudio_data.test_filenames is not None
            splits = {
                "train": nerfstudio_data.train_filenames,
                "val": nerfstudio_data.val_filenames,
                "test": nerfstudio_data.test_filenames,
            }
        else:
            splits = {"all": nerfstudio_data.filenames}
        if "train" in splits:
            assert splits[
                "train"
            ], f"No training frame identified for scene '{scene_name}'"
        else:
            assert splits["all"], f"No cameras listed for scene '{scene_name}'"
        return splits

    def _clear_scene_method_cache(self) -> None:
        self._datapoint_cache = {}

    def _compute_step_size(self, normalized: float, max_step: float) -> float:
        assert 0.0 <= normalized <= 1.0, "slider value must be in [0, 1]"
        assert max_step > 0.0, "max step must be positive"
        scale = math.log(max_step + 1.0)
        assert scale > 0.0, "step scale must be positive"
        return math.exp(normalized * scale) - 1.0

    @staticmethod
    def _build_camera_selection(split_name: str, file_path: str) -> str:
        return f"{split_name}:{file_path}"
