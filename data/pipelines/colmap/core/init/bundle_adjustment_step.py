"""Step that runs COLMAP bundle_adjuster on the initialized model."""

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.load import (
    _load_colmap_cameras_bin,
    _load_colmap_images_bin,
)


class ColmapBundleAdjustmentStep(BaseStep):
    """Refine the initialized sparse model using bundle adjustment."""

    STEP_NAME = "colmap_bundle_adjustment"

    def __init__(self, scene_root: str | Path) -> None:
        # Input validations
        assert isinstance(scene_root, (str, Path)), f"{type(scene_root)=}"

        # Input normalizations
        scene_root = Path(scene_root)

        self.scene_root = scene_root
        self.model_dir: Path | None = None
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        seed_count = self._seed_count()
        self.model_dir = self.scene_root / f"seed{seed_count}_triangulated"
        self.input_files = [
            f"seed{seed_count}_triangulated/cameras.bin",
            f"seed{seed_count}_triangulated/images.bin",
            f"seed{seed_count}_triangulated/points3D.bin",
        ]

    def _init_output_files(self) -> None:
        seed_count = self._seed_count()
        self.model_dir = self.scene_root / f"seed{seed_count}_triangulated"
        self.output_files = [
            f"seed{seed_count}_triangulated/cameras.bin",
            f"seed{seed_count}_triangulated/images.bin",
            f"seed{seed_count}_triangulated/points3D.bin",
        ]

    def build(self, force: bool = False) -> None:
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_model()
            return True
        except Exception as e:
            return False

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            return {}

        cmd_parts = self._build_colmap_command()
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        ret_code = result.returncode
        assert ret_code == 0, (
            f"COLMAP bundle_adjuster failed with code {ret_code} "
            f"for model {self.model_dir}"
        )

        self._validate_model()
        return {}

    def _build_colmap_command(self) -> List[str]:
        assert self.model_dir is not None, "model_dir unset"
        return [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            str(self.model_dir),
            "--output_path",
            str(self.model_dir),
            "--BundleAdjustment.refine_principal_point",
            "0",
            "--BundleAdjustment.refine_focal_length",
            "0",
            "--BundleAdjustment.refine_extra_params",
            "0",
            "--log_to_stderr",
            "1",
        ]

    def _validate_model(self) -> None:
        assert self.model_dir is not None, "model_dir unset"
        cameras = _load_colmap_cameras_bin(
            path_to_model_file=str(self.model_dir / "cameras.bin")
        )
        images = _load_colmap_images_bin(
            path_to_model_file=str(self.model_dir / "images.bin")
        )
        assert cameras, f"No cameras parsed from {self.model_dir / 'cameras.bin'}"
        assert (
            len(cameras) == 1
        ), f"Expected exactly one camera, got {len(cameras)} in {self.model_dir}"
        assert (
            images
        ), f"No registered images parsed from {self.model_dir / 'images.bin'}"

    def _seed_count(self) -> int:
        init_images_path = self.scene_root / "colmap_init" / "images.bin"
        images = _load_colmap_images_bin(path_to_model_file=str(init_images_path))
        seed_count = len(images)
        assert seed_count > 0, f"No images parsed from {init_images_path}"
        return seed_count
