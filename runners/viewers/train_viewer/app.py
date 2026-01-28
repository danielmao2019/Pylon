import argparse

from dash import Dash

from runners.viewers.train_viewer.callbacks.register import register_callbacks
from runners.viewers.train_viewer.layout.components import build_layout


def create_app() -> Dash:
    app = Dash(__name__)
    build_layout(app=app)
    register_callbacks(app=app)
    return app


def run_app(port: int) -> None:
    # Input validations
    assert isinstance(port, int), f"port must be int, got {type(port)}"
    assert 1024 <= port <= 65535, f"port must be between 1024 and 65535, got {port}"

    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the training losses viewer")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_app(port=args.port)


if __name__ == "__main__":
    main()
