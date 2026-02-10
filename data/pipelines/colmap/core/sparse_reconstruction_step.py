"""Step that runs COLMAP sparse reconstruction."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.load import (
    _load_colmap_cameras_bin,
    _load_colmap_images_bin,
    _load_colmap_points_bin,
)


class ColmapSparseReconstructionStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP mapper stage."""

    STEP_NAME = "colmap_sparse_reconstruction"

    def __init__(
        self,
        scene_root: str | Path,
        use_init_model: bool = False,
        strict: bool = True,
    ) -> None:
        # Input validations
        assert isinstance(scene_root, (str, Path)), f"{type(scene_root)=}"
        assert isinstance(use_init_model, bool), f"{type(use_init_model)=}"
        assert isinstance(strict, bool), f"{type(strict)=}"

        scene_root = Path(scene_root)
        self.scene_root = scene_root
        self.input_images_dir = scene_root / "input"
        self.distorted_dir = scene_root / "distorted"
        self.sparse_output_dir = scene_root / "distorted" / "sparse"
        self.init_model_dir: Path | None = None
        self.use_init_model = use_init_model
        self.strict = strict
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        image_entries = sorted(self.input_images_dir.glob("*.png"))
        self.input_files = [f"input/{entry.name}" for entry in image_entries]
        self.input_files.append("distorted/database.db")
        if self.use_init_model:
            seed_count = self._seed_count()
            self.init_model_dir = self.scene_root / f"seed{seed_count}_triangulated"
            self.input_files.extend(
                [
                    f"seed{seed_count}_triangulated/cameras.bin",
                    f"seed{seed_count}_triangulated/images.bin",
                    f"seed{seed_count}_triangulated/points3D.bin",
                ]
            )

    def _init_output_files(self) -> None:
        self.output_files = [
            "distorted/sparse/cameras.bin",
            "distorted/sparse/images.bin",
            "distorted/sparse/points3D.bin",
        ]

    def build(self, force: bool = False) -> None:
        if self._built:
            return
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_sparse_files()
            return True
        except Exception:
            return False

    def _validate_sparse_files(self) -> None:
        sparse_model_dir = self.sparse_output_dir
        logging.info(
            "VALIDATE_MAPPER_OUTPUT layout=flat model_dir=%s files_present=%s",
            sparse_model_dir,
            1,
        )

        cameras_path = sparse_model_dir / "cameras.bin"
        images_path = sparse_model_dir / "images.bin"
        points_path = sparse_model_dir / "points3D.bin"
        cameras = _load_colmap_cameras_bin(path_to_model_file=str(cameras_path))
        images = _load_colmap_images_bin(path_to_model_file=str(images_path))
        points3d = _load_colmap_points_bin(path_to_model_file=str(points_path))

        assert cameras, f"No cameras parsed from {cameras_path}"
        assert (
            len(cameras) == 1
        ), f"Expected exactly one camera in {cameras_path}, found {len(cameras)}"
        camera = next(iter(cameras.values()))
        assert camera.model == "OPENCV", f"{camera.model=}"
        assert images, f"No registered images parsed from {images_path}"
        entries = sorted(self.input_images_dir.iterdir())
        assert entries, f"Empty input dir or no files: {self.input_images_dir}"
        assert all(entry.is_file() for entry in entries), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        expected_names = {entry.name for entry in entries}
        registered_names = {img.name for img in images.values()}
        assert registered_names, (
            f"No registered images found in {images_path} "
            f"(expected={len(expected_names)})"
        )
        logging.info(
            "Mapper registered images: %s / %s",
            len(registered_names),
            len(expected_names),
        )
        if self.strict:
            assert registered_names == expected_names, (
                "Registered images must match all inputs. "
                f"expected={len(expected_names)} actual={len(registered_names)}"
            )
        else:
            assert registered_names.issubset(expected_names), (
                "Registered image names contain entries not in inputs. "
                f"expected={len(expected_names)} actual={len(registered_names)}"
            )
            ratio = len(registered_names) / float(len(expected_names))
            assert ratio > 0.1, f"Registered image ratio too low: {ratio:.4f}"
        assert points3d, f"No points parsed from {points_path}"
        # image_ids = {img.id for img in images.values()}
        # for point in points3d.values():
        #     assert len(point.image_ids) == len(
        #         point.point2D_idxs
        #     ), "points3D.bin contains mismatched image_ids and point2D_idxs lengths"
        #     assert set(point.image_ids).issubset(
        #         image_ids
        #     ), "points3D.bin references image ids not present in images.bin"

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            return {}

        logging.info("   🏗️ Sparse reconstruction")
        self._clean_sparse_output_dir()
        self._log_pre_mapper_output_state()
        cmd_parts = self._build_colmap_command()
        result = subprocess.run(cmd_parts)
        assert result.returncode == 0, (
            f"COLMAP mapper failed with code {result.returncode}. "
            f"Command: {' '.join(cmd_parts)}"
        )

        self._validate_sparse_files()
        return {}

    def _build_colmap_command(self) -> List[str]:
        distorted_db_path = self.distorted_dir / "database.db"
        cmd_parts = [
            "colmap",
            "mapper",
            "--database_path",
            str(distorted_db_path),
            "--image_path",
            str(self.input_images_dir),
            "--output_path",
            str(self.sparse_output_dir),
            "--Mapper.multiple_models",
            "0",
            "--Mapper.ba_global_function_tolerance",
            "0.000001",
            "--Mapper.ba_refine_focal_length",
            "1",
            "--Mapper.ba_refine_extra_params",
            "1",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.tri_min_angle",
            "1",
            "--log_to_stderr",
            "1",
        ]
        if self.use_init_model:
            assert self.init_model_dir is not None, "init_model_dir unset"
            cmd_parts.extend(["--input_path", str(self.init_model_dir)])
        return cmd_parts

    def _clean_sparse_output_dir(self) -> None:
        output_path = self.sparse_output_dir
        if output_path.exists():
            entries = sorted(output_path.iterdir(), key=lambda entry: entry.name)
            logging.info(
                "CLEAN_MAPPER_OUTPUT path=%s action=deleting_existing entries=%s",
                output_path,
                [entry.name for entry in entries],
            )
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(
            "CLEAN_MAPPER_OUTPUT path=%s action=created_empty",
            output_path,
        )

    def _log_pre_mapper_output_state(self) -> None:
        output_path = self.sparse_output_dir
        exists = output_path.exists()
        children = []
        if exists:
            children = [entry.name for entry in sorted(output_path.iterdir())]
        logging.info(
            "PRE_MAPPER_OUTPUT_STATE path=%s exists=%s children=%s",
            output_path,
            int(exists),
            children,
        )

    def _seed_count(self) -> int:
        init_images_path = self.scene_root / "colmap_init" / "images.bin"
        images = _load_colmap_images_bin(path_to_model_file=str(init_images_path))
        seed_count = len(images)
        assert seed_count > 0, f"No images parsed from {init_images_path}"
        return seed_count
