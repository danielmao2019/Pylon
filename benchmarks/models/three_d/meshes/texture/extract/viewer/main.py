"""CLI entrypoint for the texture-extraction benchmark viewer."""

from argparse import ArgumentParser
from pathlib import Path

from benchmarks.models.three_d.meshes.texture.extract.viewer.backend.benchmark_backend import (
    get_results_root,
    prepare_benchmark_results,
)
from benchmarks.models.three_d.meshes.texture.extract.viewer.build_app import build_app


def main() -> None:
    """Prepare results and launch the benchmark Dash app.

    Args:
        None.

    Returns:
        None.
    """

    parser = ArgumentParser(description="Texture extraction benchmark viewer")
    parser.add_argument(
        "--port",
        type=int,
        default=8092,
        help="Port for the Dash server.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=get_results_root(),
        help="Benchmark results root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all benchmark results before launch.",
    )
    args = parser.parse_args()

    prepare_benchmark_results(
        results_root=args.results_root,
        force=args.force,
    )
    app = build_app(results_root=args.results_root)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
