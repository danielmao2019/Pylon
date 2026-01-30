import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.validation import validate_camera_intrinsics


class TransformsJSON:
    _CACHE: Dict[tuple[Path, str, float], "TransformsJSON"] = {}

    def __new__(
        cls, filepath: str | Path, device: str | torch.device = torch.device("cuda")
    ) -> "TransformsJSON":
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

    def __init__(
        self, filepath: str | Path, device: str | torch.device = torch.device("cuda")
    ) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        path = Path(filepath).resolve()
        assert path.is_file(), f"transforms.json not found: {path}"
        assert isinstance(device, (str, torch.device)), f"{type(device)=}"
        self.device = torch.device(device)
        with path.open("r", encoding="utf-8") as handle:
            data: Dict[str, Any] = json.load(handle)

        assert isinstance(
            data, dict
        ), f"transforms.json payload must be dict, got {type(data)}"
        self.data: Dict[str, Any] = data

        TransformsJSON._validate_intrinsic_params(data)
        TransformsJSON._validate_resolution(data)
        TransformsJSON._validate_camera_model(data)
        TransformsJSON._validate_intrinsics(data)
        TransformsJSON._validate_applied_transform(data)
        TransformsJSON._validate_ply_file_path(data)
        TransformsJSON._validate_frames(data)
        TransformsJSON._validate_split_filenames(data)

        self.intrinsic_params: Dict[str, float | int] = self._load_intrinsic_params(
            data
        )
        self.resolution: Tuple[int, int] = self._load_resolution(data)
        self.camera_model: str = self._load_camera_model(data)
        self.intrinsics: torch.Tensor = self._load_intrinsics(
            data=data, device=self.device
        )
        self.applied_transform: np.ndarray = self._load_applied_transform(data)
        self.ply_file_path: str = self._load_ply_file_path(data)
        (
            self.train_filenames,
            self.val_filenames,
            self.test_filenames,
        ) = self._load_split_filenames(data)
        self.cameras = self._load_cameras(data=data, device=self.device)
        self.filenames: List[str] = self._load_filenames(data)
        self._initialized = True
        self._filepath = path

    def __copy__(self) -> "TransformsJSON":
        return type(self)(self._filepath, device=self.device)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TransformsJSON":
        return type(self)(self._filepath, device=self.device)

    @staticmethod
    def save(data: "TransformsJSON" | Dict[str, Any], filepath: str | Path) -> None:
        # Input validations
        assert isinstance(data, (TransformsJSON, dict)), f"{type(data)=}"
        path = Path(filepath)
        assert isinstance(path, Path), f"{type(path)=}"
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
        payload.update(TransformsJSON._save_intrinsic_params(intrinsic_params))
        payload.update(TransformsJSON._save_resolution(resolution))
        payload.update(TransformsJSON._save_camera_model(camera_model))
        payload.update(TransformsJSON._save_applied_transform(applied_transform))
        payload.update(TransformsJSON._save_ply_file_path(ply_file_path))
        payload.update(TransformsJSON._save_cameras(cameras, modalities=modalities))
        payload.update(
            TransformsJSON._save_split_filenames(
                train=train_filenames,
                val=val_filenames,
                test=test_filenames,
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    @staticmethod
    def _validate_intrinsic_params(data: Dict[str, Any]) -> None:
        assert "fl_x" in data, "transforms.json missing fl_x"
        assert "fl_y" in data, "transforms.json missing fl_y"
        assert "cx" in data, "transforms.json missing cx"
        assert "cy" in data, "transforms.json missing cy"
        assert isinstance(data["fl_x"], float), f"{type(data['fl_x'])=}"
        assert isinstance(data["fl_y"], float), f"{type(data['fl_y'])=}"
        assert isinstance(data["cx"], float), f"{type(data['cx'])=}"
        assert isinstance(data["cy"], float), f"{type(data['cy'])=}"
        assert "k1" in data, "transforms.json missing k1"
        assert "k2" in data, "transforms.json missing k2"
        assert "p1" in data, "transforms.json missing p1"
        assert "p2" in data, "transforms.json missing p2"
        assert isinstance(data["k1"], float), f"{type(data['k1'])=}"
        assert isinstance(data["k2"], float), f"{type(data['k2'])=}"
        assert isinstance(data["p1"], float), f"{type(data['p1'])=}"
        assert isinstance(data["p2"], float), f"{type(data['p2'])=}"
        assert float(data["k1"]) == 0.0, f"k1 must be 0, got {data['k1']}"
        assert float(data["k2"]) == 0.0, f"k2 must be 0, got {data['k2']}"
        assert float(data["p1"]) == 0.0, f"p1 must be 0, got {data['p1']}"
        assert float(data["p2"]) == 0.0, f"p2 must be 0, got {data['p2']}"

    @staticmethod
    def _validate_resolution(data: Dict[str, Any]) -> None:
        assert "w" in data and "h" in data, "transforms.json must include w and h"
        assert isinstance(data["w"], int), f"{type(data['w'])=}"
        assert isinstance(data["h"], int), f"{type(data['h'])=}"
        assert (
            data["w"] > 0 and data["h"] > 0
        ), f"w/h must be positive, got {data['w']}, {data['h']}"

        assert "cx" in data and "cy" in data, "transforms.json must include cx and cy"
        assert isinstance(data["cx"], float), f"{type(data['cx'])=}"
        assert isinstance(data["cy"], float), f"{type(data['cy'])=}"
        assert (
            data["cx"] > 0.0 and data["cy"] > 0.0
        ), f"cx/cy must be positive, got {data['cx']}, {data['cy']}"

        assert data["w"] == int(round(2 * float(data["cx"]))), "w must equal 2*cx"
        assert data["h"] == int(round(2 * float(data["cy"]))), "h must equal 2*cy"

    @staticmethod
    def _validate_camera_model(data: Dict[str, Any]) -> None:
        assert "camera_model" in data, "transforms.json missing camera_model"
        assert (
            data["camera_model"] == "OPENCV"
        ), f"Unsupported camera_model: {data['camera_model']}"

    @staticmethod
    def _validate_intrinsics(data: Dict[str, Any]) -> None:
        assert float(data["fl_x"]) > 0.0, "fl_x must be positive"
        assert float(data["fl_y"]) > 0.0, "fl_y must be positive"
        assert float(data["cx"]) >= 0.0, "cx must be non-negative"
        assert float(data["cy"]) >= 0.0, "cy must be non-negative"

    @staticmethod
    def _validate_applied_transform(data: Dict[str, Any]) -> None:
        assert "applied_transform" in data, "transforms.json missing applied_transform"
        assert np.asarray(data["applied_transform"], dtype=np.float32).shape == (3, 4)

    @staticmethod
    def _validate_ply_file_path(data: Dict[str, Any]) -> None:
        assert "ply_file_path" in data, "transforms.json missing ply_file_path"
        assert isinstance(data["ply_file_path"], str), f"{type(data['ply_file_path'])=}"

    @staticmethod
    def _validate_frames(data: Dict[str, Any]) -> None:
        assert "frames" in data, "transforms.json missing frames"
        assert isinstance(data["frames"], list), "frames must be a list"
        assert data["frames"], "frames must be non-empty"
        assert all("file_path" in frame for frame in data["frames"])
        assert all(
            isinstance(frame["file_path"], str)
            and frame["file_path"].startswith("images/")
            and frame["file_path"].endswith(".png")
            for frame in data["frames"]
        )
        assert all(
            ("mask_path" not in frame)
            or (
                isinstance(frame["mask_path"], str)
                and frame["mask_path"].startswith("masks/")
                and frame["mask_path"].endswith(".png")
            )
            for frame in data["frames"]
        )
        assert all("transform_matrix" in frame for frame in data["frames"])
        assert all(
            ("colmap_image_id" not in frame)
            or isinstance(frame["colmap_image_id"], int)
            for frame in data["frames"]
        )

    @staticmethod
    def _validate_split_filenames(data: Dict[str, Any]) -> None:
        assert (
            "train_filenames" in data
            and "val_filenames" in data
            and "test_filenames" in data
        ) or (
            "train_filenames" not in data
            and "val_filenames" not in data
            and "test_filenames" not in data
        ), "train/val/test filenames must all be provided together or all omitted"
        if "train_filenames" in data:
            assert isinstance(
                data["train_filenames"], list
            ), f"{type(data['train_filenames'])=}"
            assert isinstance(
                data["val_filenames"], list
            ), f"{type(data['val_filenames'])=}"
            assert isinstance(
                data["test_filenames"], list
            ), f"{type(data['test_filenames'])=}"
            assert data["train_filenames"], "train_filenames must be non-empty"
            assert data["val_filenames"], "val_filenames must be non-empty"
            assert data["test_filenames"], "test_filenames must be non-empty"
            assert {frame["file_path"] for frame in data["frames"]} == set(
                data["train_filenames"]
            ) | set(data["val_filenames"]) | set(
                data["test_filenames"]
            ), "train/val/test filenames must match frames file_path entries"

    @staticmethod
    def _load_intrinsic_params(data: Dict[str, Any]) -> Dict[str, float | int]:
        keys = ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"]
        return {key: data[key] for key in keys}

    @staticmethod
    def _load_resolution(data: Dict[str, Any]) -> Tuple[int, int]:
        return (data["h"], data["w"])

    @staticmethod
    def _load_camera_model(data: Dict[str, Any]) -> str:
        return data["camera_model"]

    @staticmethod
    def _load_intrinsics(
        data: Dict[str, Any], device: str | torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        intrinsics = torch.tensor(
            [
                [
                    float(data["fl_x"]),
                    0.0,
                    float(data["cx"]),
                ],
                [
                    0.0,
                    float(data["fl_y"]),
                    float(data["cy"]),
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=torch.device(device),
        )
        validate_camera_intrinsics(intrinsics)
        return intrinsics

    @staticmethod
    def _load_applied_transform(data: Dict[str, Any]) -> np.ndarray:
        return np.asarray(data["applied_transform"], dtype=np.float32)

    @staticmethod
    def _load_ply_file_path(data: Dict[str, Any]) -> str:
        return data["ply_file_path"]

    @staticmethod
    def _load_cameras(
        data: Dict[str, Any], device: str | torch.device = torch.device("cpu")
    ) -> Cameras:
        frames: List[Any] = data["frames"]
        intrinsics = TransformsJSON._load_intrinsics(data, device=device)
        batched_intrinsics = intrinsics.repeat(len(frames), 1, 1)
        batched_extrinsics = torch.stack(
            [
                torch.tensor(
                    frame["transform_matrix"],
                    dtype=torch.float32,
                    device=device,
                )
                for frame in frames
            ],
            dim=0,
        )
        conventions = ["opengl"] * len(frames)
        names: List[str | None] = [Path(frame["file_path"]).stem for frame in frames]
        ids: List[int | None] = [frame.get("colmap_image_id") for frame in frames]
        return Cameras(
            intrinsics=batched_intrinsics,
            extrinsics=batched_extrinsics,
            conventions=conventions,
            names=names,
            ids=ids,
            device=device,
        )

    @staticmethod
    def _load_filenames(data: Dict[str, Any]) -> List[str]:
        frames: List[Any] = data["frames"]
        return [frame["file_path"] for frame in frames]

    @staticmethod
    def _load_split_filenames(
        data: Dict[str, Any],
    ) -> Tuple[List[str] | None, List[str] | None, List[str] | None]:
        if "train_filenames" not in data:
            return None, None, None
        return data["train_filenames"], data["val_filenames"], data["test_filenames"]

    @staticmethod
    def _save_intrinsic_params(params: Dict[str, float | int]) -> Dict[str, Any]:
        keys = ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"]
        return {key: params[key] for key in keys}

    @staticmethod
    def _save_resolution(resolution: Tuple[int, int]) -> Dict[str, Any]:
        height, width = resolution
        return {"h": height, "w": width}

    @staticmethod
    def _save_camera_model(camera_model: str) -> Dict[str, Any]:
        return {"camera_model": camera_model}

    @staticmethod
    def _save_applied_transform(transform: np.ndarray) -> Dict[str, Any]:
        return {"applied_transform": transform.tolist()}

    @staticmethod
    def _save_ply_file_path(ply_file_path: str) -> Dict[str, Any]:
        return {"ply_file_path": ply_file_path}

    @staticmethod
    def _save_cameras(
        cameras: Cameras | List[Camera], modalities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        frames: List[Dict[str, Any]] = []
        include_masks = modalities is not None and "masks" in modalities
        for camera in cameras:
            assert (
                camera.name is not None
            ), "Camera name required to save transforms.json"
            frame_entry: Dict[str, Any] = {
                "file_path": f"images/{camera.name}.png",
                "transform_matrix": camera.extrinsics.detach().cpu().tolist(),
            }
            if camera.id is not None:
                frame_entry["colmap_image_id"] = camera.id
            if include_masks:
                frame_entry["mask_path"] = f"masks/{camera.name}.png"
            frames.append(frame_entry)
        frames.sort(key=lambda entry: entry["file_path"])
        return {"frames": frames}

    @staticmethod
    def _save_split_filenames(
        train: List[str] | None,
        val: List[str] | None,
        test: List[str] | None,
    ) -> Dict[str, Any]:
        if train is None:
            return {}
        return {
            "train_filenames": train,
            "val_filenames": val,
            "test_filenames": test,
        }
