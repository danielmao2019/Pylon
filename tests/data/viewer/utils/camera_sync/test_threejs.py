"""Tests for shared Three.js camera-sync utilities."""

import json
import subprocess
from pathlib import Path

from dash import Dash, html

from data.viewer.utils.camera_sync import register_threejs_camera_sync
from data.viewer.utils.camera_sync.threejs import build_threejs_camera_sync_script


def test_register_threejs_camera_sync_serves_script_and_appends_resource() -> None:
    """Register one served script resource for Three.js camera sync.

    Args:
        None.

    Returns:
        None.
    """

    app = Dash(__name__)
    script_url_path = register_threejs_camera_sync(app=app)

    assert isinstance(script_url_path, str), f"{type(script_url_path)=}"
    assert (
        script_url_path == "/__shared_threejs_camera_sync/host.js"
    ), f"{script_url_path=}"
    app.layout = html.Div()
    response = app.server.test_client().get(script_url_path)
    assert response.status_code == 200, f"{response.status_code=}"
    response_text = response.get_data(as_text=True)
    assert "threejs-camera-sync-change" in response_text, f"{response_text[:240]=}"
    assert "threejs-camera-sync-apply" in response_text, f"{response_text[:240]=}"
    assert (
        script_url_path in app.config.external_scripts
    ), f"{app.config.external_scripts=}"
    assert script_url_path in app.index(), f"{app.index()=}"


def test_build_threejs_camera_sync_script_embeds_viewer_payload() -> None:
    """Embed the requested viewer id and sync group in the generated script.

    Args:
        None.

    Returns:
        None.
    """

    script_source = build_threejs_camera_sync_script(
        viewer_id=" viewer-a ",
        camera_sync_group=" sync-group ",
    )

    assert isinstance(script_source, str), f"{type(script_source)=}"
    assert "viewerId: 'viewer-a'" in script_source, f"{script_source[:240]=}"
    assert "cameraSyncGroup: 'sync-group'" in script_source, f"{script_source[:240]=}"
    assert "window.__threejsCameraSync" in script_source, f"{script_source[:240]=}"


def test_build_threejs_camera_sync_script_returns_empty_string_without_group() -> None:
    """Skip iframe sync script generation when sync is disabled.

    Args:
        None.

    Returns:
        None.
    """

    script_source = build_threejs_camera_sync_script(
        viewer_id="viewer-a",
        camera_sync_group=None,
    )

    assert script_source == "", f"{script_source=}"


def _build_threejs_iframe_harness_script(
    script_source: str,
) -> str:
    """Build one Node.js harness for the iframe-local Three.js sync script.

    Args:
        script_source: Generated iframe-local sync script.

    Returns:
        JavaScript harness source code.
    """

    assert isinstance(script_source, str), f"{type(script_source)=}"
    assert script_source != "", f"{script_source=}"

    return f"""
const assetSource = {json.dumps(script_source)};

function buildVector3(x, y, z) {{
  return {{
    x,
    y,
    z,
    set(nextX, nextY, nextZ) {{
      this.x = nextX;
      this.y = nextY;
      this.z = nextZ;
    }},
  }};
}}

global.camera = {{
  position: buildVector3(12.0, 23.0, 34.0),
  up: buildVector3(0.0, 1.0, 0.0),
  fov: 45.0,
  updateProjectionMatrix() {{}},
  lookAt() {{}},
}};

global.controls = {{
  target: buildVector3(1.0, 2.0, 3.0),
  update() {{}},
  addEventListener() {{}},
}};

global.window = {{
  __threejsCameraSyncViewBounds: {{
    center: {{ x: 100.0, y: 200.0, z: 300.0 }},
    camera_coordinate_scale: 10.0,
  }},
  parent: {{
    postMessage() {{}},
  }},
  addEventListener() {{}},
  requestAnimationFrame(callback) {{
    callback();
    return 1;
  }},
}};

eval(assetSource);

process.stdout.write(
  JSON.stringify(window.__threejsCameraSync.getCameraState()),
);
"""


def test_build_threejs_camera_sync_script_reports_raw_world_camera_state(
    tmp_path: Path,
) -> None:
    """Require the iframe-local sync API to report raw Three.js world coordinates.

    Args:
        tmp_path: Temporary directory for the Node harness file.

    Returns:
        None.
    """

    assert isinstance(tmp_path, Path), f"{type(tmp_path)=}"

    script_source = build_threejs_camera_sync_script(
        viewer_id="viewer-a",
        camera_sync_group="sync-group",
    )
    harness_path = tmp_path / "threejs_iframe_harness.js"
    harness_path.write_text(
        _build_threejs_iframe_harness_script(script_source=script_source)
    )

    completed_process = subprocess.run(
        args=["node", str(harness_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, (
        "Expected the Three.js iframe sync Node harness to succeed. "
        f"{completed_process.returncode=} {completed_process.stderr=}"
    )
    payload = json.loads(completed_process.stdout)
    expected_payload = {
        "eye": {"x": 12.0, "y": 23.0, "z": 34.0},
        "center": {"x": 1.0, "y": 2.0, "z": 3.0},
        "up": {"x": 0.0, "y": 1.0, "z": 0.0},
        "fovyRadians": 45.0 * 3.141592653589793 / 180.0,
    }
    assert payload == expected_payload, f"{payload=}"
