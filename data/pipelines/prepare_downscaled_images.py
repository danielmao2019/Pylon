"""Prepare downscaled PNG image copies for a scene."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from data.pipelines.base_step import BaseStep


if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = Image.LANCZOS


DOWNSCALE_FACTORS = (2, 4, 8)


class PrepareDownscaledImages(BaseStep):
    STEP_NAME = "prepare_downscaled_images"

    def __init__(self, scene_root: Path) -> None:
        self.scene_root = Path(scene_root)
        self.images_dir = self.scene_root / "images"
        super().__init__(input_root=self.images_dir, output_root=self.scene_root)

    def _init_input_files(self) -> None:
        children = list(self.images_dir.iterdir())
        self.input_files = [child.name for child in children]
        self.image_names = sorted(self.input_files)

    def _init_output_files(self) -> None:
        output_files: List[str] = []
        for factor in DOWNSCALE_FACTORS:
            output_files.extend(f"images_{factor}/{name}" for name in self.image_names)
        self.output_files = output_files

    def check_outputs(self) -> bool:
        if not super().check_outputs():
            return False
        try:
            self._assert_outputs_clean()
        except Exception:
            return False
        return True

    def _assert_outputs_clean(self) -> None:
        for factor in DOWNSCALE_FACTORS:
            downscale_dir = self.output_root / f"images_{factor}"
            assert (
                downscale_dir.is_dir()
            ), f"Expected downscale dir {downscale_dir} to exist"
            children = list(downscale_dir.iterdir())
            assert all(
                child.is_file() for child in children
            ), f"Expected only files in {downscale_dir}, got {children}"
            assert all(
                child.suffix == ".png" for child in children
            ), f"Expected only .png files in {downscale_dir}, got {children}"

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            logging.info("🪞 Downscaled images already exist - SKIPPED")
            return {}
        self._downscale_images(force=force)
        self._clean_other_files()
        return {}

    def _downscale_images(self, force: bool) -> None:
        base_images = [self.images_dir / name for name in self.image_names]
        logging.info("🪞 Creating downscaled image copies (%s)", DOWNSCALE_FACTORS)
        for factor in DOWNSCALE_FACTORS:
            target_dir = self.output_root / f"images_{factor}"
            target_dir.mkdir(exist_ok=True)
        jobs: List[Future[None]] = []

        def _resize_and_save(
            source: Path, target_dir: Path, factor: int, force_downscale: bool
        ) -> None:
            assert source.name.endswith(
                ".png"
            ), f"Expected .png source for {source.name}"
            target = target_dir / source.name
            if target.exists() and not force_downscale:
                return
            with Image.open(source) as base_img:
                assert (
                    base_img.mode == "RGB"
                ), f"Expected RGB image for {source.name}, got {base_img.mode}"
                w, h = base_img.size
                new_size = (max(1, w // factor), max(1, h // factor))
                resized = base_img.resize(new_size, RESAMPLE_LANCZOS)
                resized.save(target, format="PNG")

        with ThreadPoolExecutor() as executor:
            for factor in DOWNSCALE_FACTORS:
                target_dir = self.output_root / f"images_{factor}"
                for src in base_images:
                    jobs.append(
                        executor.submit(
                            _resize_and_save,
                            src,
                            target_dir,
                            factor,
                            force,
                        )
                    )
            for job in jobs:
                job.result()
        logging.info("   ✓ Generated downscaled copies for %d images", len(base_images))

    def _clean_other_files(self) -> None:
        expected_downscaled = {
            self.output_root / f"images_{factor}/{name}"
            for factor in DOWNSCALE_FACTORS
            for name in self.image_names
        }
        for factor in DOWNSCALE_FACTORS:
            downscale_dir = self.output_root / f"images_{factor}"
            for child in downscale_dir.iterdir():
                if child not in expected_downscaled:
                    child.unlink()
