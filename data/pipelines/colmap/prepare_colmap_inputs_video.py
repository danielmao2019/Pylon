"""Prepare COLMAP inputs from video(s) by extracting frames.

This generic step takes one training video and optional test videos, and writes
extracted PNG frames into `<output_root>/input`.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from data.pipelines.base_step import BaseStep


def _common_parent(paths: Sequence[Path]) -> Path:
    assert paths, "No paths provided"
    parts_lists = [p.resolve().parts for p in paths]
    common_parts: List[str] = []
    for idx in range(min(len(pl) for pl in parts_lists)):
        candidate = parts_lists[0][idx]
        if all(pl[idx] == candidate for pl in parts_lists):
            common_parts.append(candidate)
        else:
            break
    return Path(os.path.sep.join(common_parts)) if common_parts else paths[0].parent


class PrepareColmapInputsVideo(BaseStep):
    STEP_NAME = "prepare_colmap_inputs_video"

    def __init__(
        self,
        video_filepath: Path,
        test_video_filepaths: Iterable[Path] | None,
        output_root: Path,
        sample_every_n: int = 10,
    ) -> None:
        # Input validations
        assert isinstance(video_filepath, Path), f"{type(video_filepath)=}"
        assert test_video_filepaths is None or isinstance(
            test_video_filepaths, (list, tuple)
        ), f"{type(test_video_filepaths)=}"
        assert isinstance(output_root, Path), f"{type(output_root)=}"
        assert isinstance(sample_every_n, int), f"{type(sample_every_n)=}"
        assert sample_every_n > 0, f"{sample_every_n=}"

        all_videos: List[Path] = [video_filepath]
        if test_video_filepaths:
            all_videos.extend(list(test_video_filepaths))
        self.video_filepath = video_filepath
        self.test_video_filepaths = (
            list(test_video_filepaths) if test_video_filepaths else []
        )
        self.sample_every_n = sample_every_n

        input_root = _common_parent(all_videos)
        super().__init__(input_root=input_root, output_root=output_root)

    def build(self, force: bool = False) -> None:
        if self._built:
            return
        super().build(force=force)
        self.run({}, force=False)

    def check_inputs(self) -> None:
        self.input_files = [str(self.video_filepath)] + [
            str(v) for v in self.test_video_filepaths
        ]
        return super().check_inputs()

    def check_outputs(self) -> bool:
        output_dir = self.output_root / "input"
        if not output_dir.is_dir():
            return False
        try:
            paths = [p for p in output_dir.iterdir() if p.is_file()]
            assert all(
                p.suffix == ".png" for p in paths
            ), "Non-PNG files present in input dir"
            return len(paths) > 0
        except Exception as e:
            logging.debug("Video COLMAP input validation failed: %s", e)
            return False

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            logging.info("📥 Video frames already extracted - SKIPPED")
            return {}
        self._prepare_inputs()
        return {}

    def _prepare_inputs(self) -> None:
        counts = 0
        counts += self._extract_frames(
            output_dir=self.output_root / "input",
            video=self.video_filepath,
            prefix=f"{self.video_filepath.stem}_frame_",
            sample_every_n=self.sample_every_n,
        )
        for idx, v in enumerate(self.test_video_filepaths):
            counts += self._extract_frames(
                output_dir=self.output_root / "input",
                video=v,
                prefix=f"{v.stem}_frame_",
                sample_every_n=self.sample_every_n,
            )
        assert counts > 0, "No frames were extracted from provided videos"

    @staticmethod
    def _extract_frames(
        output_dir: Path, video: Path, prefix: str, sample_every_n: int
    ) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(output_dir / f"{prefix}%06d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=not(mod(n\\,{sample_every_n}))",
            "-vsync",
            "vfr",
            "-start_number",
            "0",
            pattern,
        ]
        logging.info("   ▶ %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            result.returncode == 0
        ), f"ffmpeg failed ({result.returncode}): {result.stderr}"
        paths = [
            p for p in output_dir.iterdir() if p.is_file() and p.name.startswith(prefix)
        ]
        assert all(
            p.suffix == ".png" for p in paths
        ), "Non-PNG files present in input dir"
        return len(paths)
