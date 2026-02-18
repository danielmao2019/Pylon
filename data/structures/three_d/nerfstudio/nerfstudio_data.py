from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.nerfstudio.load import (
    load_nerfstudio_data,
)
from data.structures.three_d.nerfstudio.save import save_nerfstudio_data
from data.structures.three_d.nerfstudio.transform import transform_nerfstudio
from data.structures.three_d.nerfstudio.validate import (
    validate_applied_transform,
    validate_camera_model,
    validate_cameras,
    validate_data,
    validate_device,
    validate_intrinsic_params,
    validate_intrinsics,
    validate_intrinsics_data,
    validate_modalities,
    validate_ply_file_path,
    validate_resolution,
    validate_split_filenames,
)


class NerfStudio_Data:
    _CACHE: Dict[Tuple[Path, str, float], "NerfStudio_Data"] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> "NerfStudio_Data":
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
        modalities: List[str],
        train_filenames: List[str] | None = None,
        val_filenames: List[str] | None = None,
        test_filenames: List[str] | None = None,
    ) -> None:
        # Input validations
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
        validate_modalities(modalities)
        validate_split_filenames(
            train_filenames=train_filenames,
            val_filenames=val_filenames,
            test_filenames=test_filenames,
            filenames=list(cameras.names),
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
        self.modalities = modalities
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames
        self.test_filenames = test_filenames

    def __copy__(self) -> "NerfStudio_Data":
        assert hasattr(
            self, "_filepath"
        ), "NerfStudio_Data._filepath is required for copy"
        assert (
            self._filepath is not None
        ), "NerfStudio_Data._filepath is required for copy"
        return type(self).load(filepath=self._filepath, device=self.device)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "NerfStudio_Data":
        assert hasattr(
            self, "_filepath"
        ), "NerfStudio_Data._filepath is required for copy"
        assert (
            self._filepath is not None
        ), "NerfStudio_Data._filepath is required for copy"
        return type(self).load(filepath=self._filepath, device=self.device)

    @classmethod
    def load(
        cls, filepath: str | Path, device: str | torch.device = torch.device("cuda")
    ) -> "NerfStudio_Data":
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
        (
            data,
            intrinsic_params,
            resolution,
            camera_model,
            intrinsics,
            applied_transform,
            ply_file_path,
            cameras,
            modalities,
            train_filenames,
            val_filenames,
            test_filenames,
        ) = load_nerfstudio_data(filepath=path, device=target_device)

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
            modalities=modalities,
            train_filenames=train_filenames,
            val_filenames=val_filenames,
            test_filenames=test_filenames,
        )
        instance._initialized = True
        instance._filepath = path
        return instance

    def save(self, output_path: str | Path) -> None:
        # Input validations
        assert isinstance(output_path, (str, Path)), f"{type(output_path)=}"

        save_nerfstudio_data(data=self, filepath=output_path)

    def to(
        self,
        device: str | torch.device | None = None,
        convention: str | None = None,
    ) -> "NerfStudio_Data":
        # Input validations
        assert device is None or isinstance(
            device, (str, torch.device)
        ), f"{type(device)=}"
        assert convention is None or isinstance(convention, str), f"{type(convention)=}"

        # Input normalizations
        target_device = self.device if device is None else torch.device(device)

        self.cameras = self.cameras.to(
            device=target_device,
            convention=convention,
        )
        self.device = target_device
        return self

    def transform(
        self,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        # Input validations
        assert isinstance(scale, (int, float)), f"{type(scale)=}"
        assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
        assert rotation.shape == (3, 3), f"{rotation.shape=}"
        assert rotation.dtype == np.float32, f"{rotation.dtype=}"
        assert isinstance(translation, np.ndarray), f"{type(translation)=}"
        assert translation.shape == (3,), f"{translation.shape=}"
        assert translation.dtype == np.float32, f"{translation.dtype=}"

        transformed_cameras = transform_nerfstudio(
            cameras=self.cameras,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        self.cameras = transformed_cameras

    @property
    def filenames(self) -> List[str]:
        return list(self.cameras.names)
