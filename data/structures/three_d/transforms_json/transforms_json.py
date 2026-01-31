import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.transforms_json.load import (
    load_applied_transform,
    load_camera_model,
    load_cameras,
    load_filenames,
    load_intrinsic_params,
    load_intrinsics,
    load_ply_file_path,
    load_resolution,
    load_split_filenames,
)
from data.structures.three_d.transforms_json.save import (
    save_applied_transform,
    save_camera_model,
    save_cameras,
    save_intrinsic_params,
    save_ply_file_path,
    save_resolution,
    save_split_filenames,
)
from data.structures.three_d.transforms_json.validate import (
    validate_applied_transform,
    validate_applied_transform_data,
    validate_camera_model,
    validate_camera_model_data,
    validate_cameras,
    validate_data,
    validate_device,
    validate_filenames,
    validate_frames_data,
    validate_intrinsic_params,
    validate_intrinsics,
    validate_intrinsics_data,
    validate_ply_file_path,
    validate_ply_file_path_data,
    validate_resolution,
    validate_resolution_data,
    validate_split_filenames,
    validate_split_filenames_data,
)


class TransformsJSON:
    _CACHE: Dict[Tuple[Path, str, float], "TransformsJSON"] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> "TransformsJSON":
        if "filepath" in kwargs or (args and isinstance(args[0], (str, Path))):
            filepath = kwargs["filepath"] if "filepath" in kwargs else args[0]
            device = (
                kwargs["device"]
                if "device" in kwargs
                else (args[1] if len(args) > 1 else torch.device("cuda"))
            )
            path = Path(filepath).resolve()
            assert path.is_file(), f"transforms.json not found: {path}"
            assert isinstance(device, (str, torch.device)), f"{type(device)=}"
            target_device = torch.device(device)
            cache_key = (path, str(target_device), path.stat().st_mtime)
            if cache_key in cls._CACHE:
                return cls._CACHE[cache_key]
            instance = super().__new__(cls)
            cls._CACHE[cache_key] = instance
            return instance
        return super().__new__(cls)

    def __init__(
        self,
        data: Dict[str, Any],
        device: torch.device,
        intrinsic_params: Dict[str, float | int],
        resolution: Tuple[int, int],
        camera_model: str,
        intrinsics: torch.Tensor,
        applied_transform: np.ndarray,
        ply_file_path: str,
        cameras: Cameras,
        filenames: List[str],
        train_filenames: List[str] | None,
        val_filenames: List[str] | None,
        test_filenames: List[str] | None,
    ) -> None:
        validate_data(data)
        validate_device(device)
        validate_intrinsic_params(intrinsic_params)
        validate_intrinsics_data(intrinsic_params)
        validate_resolution(resolution=resolution, intrinsic_params=intrinsic_params)
        validate_camera_model(camera_model)
        validate_intrinsics(intrinsics)
        validate_applied_transform(applied_transform)
        validate_ply_file_path(ply_file_path)
        validate_cameras(cameras)
        validate_filenames(filenames)
        validate_split_filenames(
            train=train_filenames,
            val=val_filenames,
            test=test_filenames,
            filenames=filenames,
        )

        self.data = data
        self.device = device
        self.intrinsic_params = intrinsic_params
        self.resolution = resolution
        self.camera_model = camera_model
        self.intrinsics = intrinsics
        self.applied_transform = applied_transform
        self.ply_file_path = ply_file_path
        self.cameras = cameras
        self.filenames = filenames
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames
        self.test_filenames = test_filenames

    def __copy__(self) -> "TransformsJSON":
        assert hasattr(
            self, "_filepath"
        ), "TransformsJSON._filepath is required for copy"
        assert (
            self._filepath is not None
        ), "TransformsJSON._filepath is required for copy"
        return type(self).load(filepath=self._filepath, device=self.device)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TransformsJSON":
        assert hasattr(
            self, "_filepath"
        ), "TransformsJSON._filepath is required for copy"
        assert (
            self._filepath is not None
        ), "TransformsJSON._filepath is required for copy"
        return type(self).load(filepath=self._filepath, device=self.device)

    @classmethod
    def load(
        cls, filepath: str | Path, device: str | torch.device = torch.device("cuda")
    ) -> "TransformsJSON":
        # Input validations
        assert isinstance(filepath, (str, Path)), f"{type(filepath)=}"
        assert isinstance(device, (str, torch.device)), f"{type(device)=}"

        # Input normalizations
        path = Path(filepath).resolve()
        target_device = torch.device(device)

        assert path.is_file(), f"transforms.json not found: {path}"
        instance = cls.__new__(cls, filepath=path, device=target_device)
        if hasattr(instance, "_initialized") and instance._initialized:
            return instance
        with path.open("r", encoding="utf-8") as handle:
            data: Dict[str, Any] = json.load(handle)

        validate_data(data)
        validate_intrinsic_params(data)
        validate_resolution_data(data)
        validate_camera_model_data(data)
        validate_intrinsics_data(data)
        validate_applied_transform_data(data)
        validate_ply_file_path_data(data)
        validate_frames_data(data)
        validate_split_filenames_data(data)

        intrinsic_params = load_intrinsic_params(data)
        resolution = load_resolution(data)
        camera_model = load_camera_model(data)
        intrinsics = load_intrinsics(data=data, device=target_device)
        applied_transform = load_applied_transform(data)
        ply_file_path = load_ply_file_path(data)
        train_filenames, val_filenames, test_filenames = load_split_filenames(data)
        cameras = load_cameras(data=data, device=target_device)
        filenames = load_filenames(data)

        instance.__init__(
            data=data,
            device=target_device,
            intrinsic_params=intrinsic_params,
            resolution=resolution,
            camera_model=camera_model,
            intrinsics=intrinsics,
            applied_transform=applied_transform,
            ply_file_path=ply_file_path,
            cameras=cameras,
            filenames=filenames,
            train_filenames=train_filenames,
            val_filenames=val_filenames,
            test_filenames=test_filenames,
        )
        instance._initialized = True
        instance._filepath = path
        return instance

    @staticmethod
    def save(data: "TransformsJSON" | Dict[str, Any], filepath: str | Path) -> None:
        # Input validations
        assert isinstance(data, (TransformsJSON, dict)), f"{type(data)=}"
        assert isinstance(filepath, (str, Path)), f"{type(filepath)=}"

        # Input normalizations
        path = Path(filepath)

        if isinstance(data, TransformsJSON):
            modalities = data.data.get("modalities", [])
            intrinsic_params = data.intrinsic_params
            resolution = data.resolution
            camera_model = data.camera_model
            applied_transform = data.applied_transform
            ply_file_path = data.ply_file_path
            cameras = data.cameras
            train_filenames = data.train_filenames
            val_filenames = data.val_filenames
            test_filenames = data.test_filenames
        else:
            assert "modalities" in data, f"{data.keys()=}"
            modalities = data["modalities"]
            assert isinstance(modalities, list), f"{type(modalities)=}"
            assert all(isinstance(item, str) for item in modalities), f"{modalities=}"
            assert "intrinsic_params" in data, f"{data.keys()=}"
            intrinsic_params = data["intrinsic_params"]
            assert isinstance(intrinsic_params, dict), f"{type(intrinsic_params)=}"
            assert "resolution" in data, f"{data.keys()=}"
            resolution = data["resolution"]
            assert isinstance(resolution, tuple), f"{type(resolution)=}"
            assert "camera_model" in data, f"{data.keys()=}"
            camera_model = data["camera_model"]
            assert isinstance(camera_model, str), f"{type(camera_model)=}"
            assert "applied_transform" in data, f"{data.keys()=}"
            applied_transform = data["applied_transform"]
            assert isinstance(
                applied_transform, np.ndarray
            ), f"{type(applied_transform)=}"
            assert "ply_file_path" in data, f"{data.keys()=}"
            ply_file_path = data["ply_file_path"]
            assert isinstance(ply_file_path, str), f"{type(ply_file_path)=}"
            assert "cameras" in data, f"{data.keys()=}"
            cameras = data["cameras"]
            assert isinstance(cameras, list), f"{type(cameras)=}"
            train_filenames = data.get("train_filenames")
            val_filenames = data.get("val_filenames")
            test_filenames = data.get("test_filenames")

        payload: Dict[str, Any] = {}
        payload.update(save_intrinsic_params(intrinsic_params))
        payload.update(save_resolution(resolution))
        payload.update(save_camera_model(camera_model))
        payload.update(save_applied_transform(applied_transform))
        payload.update(save_ply_file_path(ply_file_path))
        payload.update(save_cameras(cameras, modalities=modalities))
        payload.update(
            save_split_filenames(
                train=train_filenames,
                val=val_filenames,
                test=test_filenames,
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
