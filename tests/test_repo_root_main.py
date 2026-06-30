from pathlib import Path

from project.mvp.pipeline.fit.fit_pipeline import FitPipeline

import main


def test_repo_root_pipeline_build_accepts_model_config_dict(
    tmp_path: Path,
) -> None:
    """Build the repo-root fit pipeline from its public config-dict contract.

    Args:
        tmp_path: Pytest-provided temporary directory root.

    Returns:
        None.
    """
    scene_root = tmp_path / "scene"
    output_root = scene_root / "outputs"

    pipeline = main._build_pipeline(
        data_root=scene_root,
        model_dir=output_root,
    )
    assert isinstance(pipeline, FitPipeline), (
        "Expected repo-root _build_pipeline to return a FitPipeline. "
        f"{type(pipeline)=}"
    )

    pipeline.build(force=False)
