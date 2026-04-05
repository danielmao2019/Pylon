"""Tests for shared mixed Plotly/Three.js camera-sync utilities."""

import json
import subprocess
from pathlib import Path

from dash import Dash, html

from data.viewer.utils.camera_sync import (
    build_plotly_threejs_camera_sync_script,
    register_plotly_threejs_camera_sync,
)


def test_build_plotly_threejs_camera_sync_script_embeds_all_targets() -> None:
    """Embed the requested Plotly ids, iframe ids, and sync group in the relay script.

    Args:
        None.

    Returns:
        None.
    """

    script_source = build_plotly_threejs_camera_sync_script(
        graph_ids=["graph-a", "graph-b"],
        iframe_ids=["iframe-a", "iframe-b"],
        camera_sync_group="mesh-sync",
    )

    assert isinstance(script_source, str), f"{type(script_source)=}"
    assert 'const graphIds = ["graph-a", "graph-b"];' in script_source
    assert 'const iframeIds = ["iframe-a", "iframe-b"];' in script_source
    assert 'const cameraSyncGroup = "mesh-sync";' in script_source
    assert "threejs-camera-sync-change" in script_source, f"{script_source[:320]=}"
    assert "threejs-camera-sync-apply" in script_source, f"{script_source[:320]=}"


def test_register_plotly_threejs_camera_sync_serves_script_and_appends_resource() -> (
    None
):
    """Register one served script resource for mixed Plotly/Three.js camera sync.

    Args:
        None.

    Returns:
        None.
    """

    app = Dash(__name__)
    script_url_path = register_plotly_threejs_camera_sync(
        app=app,
        graph_ids=["graph-a", "graph-b"],
        iframe_ids=["iframe-a", "iframe-b"],
        camera_sync_group="mesh-sync",
    )

    assert isinstance(script_url_path, str), f"{type(script_url_path)=}"
    assert script_url_path.startswith(
        "/__shared_plotly_threejs_camera_sync/"
    ), f"{script_url_path=}"
    app.layout = html.Div()
    response = app.server.test_client().get(script_url_path)
    assert response.status_code == 200, f"{response.status_code=}"
    response_text = response.get_data(as_text=True)
    assert 'const graphIds = ["graph-a", "graph-b"];' in response_text
    assert 'const iframeIds = ["iframe-a", "iframe-b"];' in response_text
    assert (
        script_url_path in app.config.external_scripts
    ), f"{app.config.external_scripts=}"
    assert script_url_path in app.index(), f"{app.index()=}"


def _build_plotly_threejs_harness_script(script_source: str) -> str:
    """Build a Node.js harness for one mixed Plotly/Three.js sync script.

    Args:
        script_source: Generated mixed camera-sync browser script.

    Returns:
        JavaScript harness source code.
    """

    assert isinstance(script_source, str), f"{type(script_source)=}"
    assert script_source != "", f"{script_source=}"

    return f"""
const assetSource = {json.dumps(script_source)};
let nextTimerId = 1;
const scheduledTimers = [];
let attachSync = null;

function cloneCamera(cameraState) {{
  return JSON.parse(JSON.stringify(cameraState));
}}

function scheduleTimer(callback, delayMs) {{
  const timerId = nextTimerId;
  nextTimerId += 1;
  scheduledTimers.push({{ callback, delayMs, timerId }});
  return timerId;
}}

function flushScheduledTimers() {{
  while (scheduledTimers.length > 0) {{
    const nextTimer = scheduledTimers.shift();
    nextTimer.callback();
  }}
}}

function createPlotlyElement(graphId) {{
  return {{
    id: graphId,
    layout: {{
      meta: {{
        meshViewBounds: {{
          center: {{ x: 10.0, y: 20.0, z: 30.0 }},
          camera_coordinate_scale: 5.0,
        }},
      }},
      scene: {{
        camera: {{
          center: {{ x: 0, y: 0, z: 0 }},
          eye: {{ x: 1, y: 1, z: 1 }},
          up: {{ x: 0, y: 0, z: 1 }},
        }},
      }},
    }},
    __handlers: {{}},
    on(eventName, handler) {{
      if (!this.__handlers[eventName]) {{
        this.__handlers[eventName] = [];
      }}
      this.__handlers[eventName].push(handler);
    }},
    emit(eventName, eventData) {{
      const handlers = this.__handlers[eventName] || [];
      handlers.forEach((handler) => handler(eventData));
    }},
  }};
}}

function createGraphContainer(graphId) {{
  const plotlyElement = createPlotlyElement(graphId);
  return {{
    id: graphId,
    __plotlyElement: plotlyElement,
    querySelector(selector) {{
      if (selector === ".js-plotly-plot") {{
        return plotlyElement;
      }}
      return null;
    }},
  }};
}}

function createIframe(iframeId) {{
  return {{
    id: iframeId,
    contentWindow: {{
      __cameraState: null,
      postMessage(message) {{
        if (message.type === "threejs-camera-sync-apply") {{
          this.__cameraState = cloneCamera(message.cameraState);
        }}
      }},
    }},
  }};
}}

const graphContainers = {{
  "graph-a": createGraphContainer("graph-a"),
  "graph-b": createGraphContainer("graph-b"),
}};
const iframeElements = {{
  "iframe-a": createIframe("iframe-a"),
}};
const relayoutCounts = {{
  "graph-a": 0,
  "graph-b": 0,
}};

global.document = {{
  getElementById(elementId) {{
    return graphContainers[elementId] || iframeElements[elementId] || null;
  }},
}};

global.window = {{
  Plotly: null,
  addEventListener() {{}},
  setInterval(callback) {{
    attachSync = callback;
    return 1;
  }},
  clearInterval() {{}},
  setTimeout(callback, delayMs) {{
    return scheduleTimer(callback, delayMs);
  }},
}};

global.Plotly = {{
  relayout(plotlyElement, update) {{
    relayoutCounts[plotlyElement.id] += 1;
    plotlyElement.layout.scene.camera = cloneCamera(update["scene.camera"]);
    return Promise.resolve();
  }},
}};
window.Plotly = global.Plotly;

async function main() {{
  eval(assetSource);
  if (typeof attachSync !== "function") {{
    throw new Error("Expected mixed sync script to register an attach interval.");
  }}

  attachSync();

  const sourcePlotlyElement = graphContainers["graph-a"].__plotlyElement;
  sourcePlotlyElement.layout.scene.camera = {{
    center: {{ x: 0.2, y: 0.3, z: 0.4 }},
    eye: {{ x: 2.0, y: 3.0, z: 4.0 }},
    up: {{ x: 0.0, y: 1.0, z: 0.0 }},
  }};
  sourcePlotlyElement.emit("plotly_relayout", {{}});

  await Promise.resolve();
  flushScheduledTimers();
  await Promise.resolve();

  const relayoutCountsAfterFirstBroadcast = {{
    "graph-a": relayoutCounts["graph-a"],
    "graph-b": relayoutCounts["graph-b"],
  }};
  sourcePlotlyElement.emit("plotly_relayout", {{}});

  await Promise.resolve();
  flushScheduledTimers();
  await Promise.resolve();

  process.stdout.write(JSON.stringify({{
    relayoutCounts,
    relayoutCountsAfterFirstBroadcast,
    sourceCamera: sourcePlotlyElement.layout.scene.camera,
    peerCamera: graphContainers["graph-b"].__plotlyElement.layout.scene.camera,
    iframeCamera: iframeElements["iframe-a"].contentWindow.__cameraState,
  }}));
}}

main().catch((error) => {{
  process.stderr.write(String(error && error.stack ? error.stack : error));
  process.exit(1);
}});
"""


def _build_plotly_threejs_world_camera_harness_script(
    script_source: str,
) -> str:
    """Build a Node.js harness that checks Plotly-specific world-camera conversion.

    Args:
        script_source: Generated mixed camera-sync browser script.

    Returns:
        JavaScript harness source code.
    """

    assert isinstance(script_source, str), f"{type(script_source)=}"
    assert script_source != "", f"{script_source=}"

    return f"""
const assetSource = {json.dumps(script_source)};
let nextTimerId = 1;
const scheduledTimers = [];
let attachSync = null;
let messageHandler = null;

function cloneCamera(cameraState) {{
  return JSON.parse(JSON.stringify(cameraState));
}}

function scheduleTimer(callback, delayMs) {{
  const timerId = nextTimerId;
  nextTimerId += 1;
  scheduledTimers.push({{ callback, delayMs, timerId }});
  return timerId;
}}

function flushScheduledTimers() {{
  while (scheduledTimers.length > 0) {{
    const nextTimer = scheduledTimers.shift();
    nextTimer.callback();
  }}
}}

function createPlotlyElement(graphId, meshViewBounds) {{
  return {{
    id: graphId,
    layout: {{
      meta: {{
        meshViewBounds,
      }},
      scene: {{
        camera: {{
          center: {{ x: 0.2, y: 0.3, z: 0.4 }},
          eye: {{ x: 2.0, y: 3.0, z: 4.0 }},
          up: {{ x: 0.0, y: 1.0, z: 0.0 }},
        }},
      }},
    }},
    __handlers: {{}},
    on(eventName, handler) {{
      if (!this.__handlers[eventName]) {{
        this.__handlers[eventName] = [];
      }}
      this.__handlers[eventName].push(handler);
    }},
    emit(eventName, eventData) {{
      const handlers = this.__handlers[eventName] || [];
      handlers.forEach((handler) => handler(eventData));
    }},
  }};
}}

function createGraphContainer(graphId, meshViewBounds) {{
  const plotlyElement = createPlotlyElement(graphId, meshViewBounds);
  return {{
    id: graphId,
    __plotlyElement: plotlyElement,
    querySelector(selector) {{
      if (selector === ".js-plotly-plot") {{
        return plotlyElement;
      }}
      return null;
    }},
  }};
}}

function createIframe(iframeId) {{
  return {{
    id: iframeId,
    contentWindow: {{
      __messages: [],
      postMessage(message) {{
        if (message.type === "threejs-camera-sync-apply") {{
          this.__messages.push(cloneCamera(message.cameraState));
        }}
      }},
    }},
  }};
}}

const meshViewBounds = {{
  center: {{ x: 10.0, y: 20.0, z: 30.0 }},
  camera_coordinate_scale: 5.0,
}};
const graphContainers = {{
  "graph-a": createGraphContainer("graph-a", meshViewBounds),
  "graph-b": createGraphContainer("graph-b", meshViewBounds),
}};
const iframeElements = {{
  "iframe-a": createIframe("iframe-a"),
}};

global.document = {{
  getElementById(elementId) {{
    return graphContainers[elementId] || iframeElements[elementId] || null;
  }},
}};

global.window = {{
  Plotly: null,
  addEventListener(eventName, handler) {{
    if (eventName === "message") {{
      messageHandler = handler;
    }}
  }},
  setInterval(callback) {{
    attachSync = callback;
    return 1;
  }},
  clearInterval() {{}},
  setTimeout(callback, delayMs) {{
    return scheduleTimer(callback, delayMs);
  }},
}};

global.Plotly = {{
  relayout(plotlyElement, update) {{
    plotlyElement.layout.scene.camera = cloneCamera(update["scene.camera"]);
    return Promise.resolve();
  }},
}};
window.Plotly = global.Plotly;

async function main() {{
  eval(assetSource);
  if (typeof attachSync !== "function") {{
    throw new Error("Expected mixed sync script to register an attach interval.");
  }}
  if (typeof messageHandler !== "function") {{
    throw new Error("Expected mixed sync script to register a message handler.");
  }}

  attachSync();

  const sourcePlotlyElement = graphContainers["graph-a"].__plotlyElement;
  sourcePlotlyElement.emit("plotly_relayout", {{}});

  await Promise.resolve();
  flushScheduledTimers();
  await Promise.resolve();

  const iframeCameraFromPlotly = iframeElements["iframe-a"].contentWindow.__messages.at(-1);

  messageHandler({{
    data: {{
      type: "threejs-camera-sync-change",
      cameraSyncGroup: "mesh-sync",
      sourceViewerId: "iframe-a",
      cameraState: {{
        eye: {{ x: 20.0, y: 35.0, z: 50.0 }},
        center: {{ x: 11.0, y: 21.5, z: 32.0 }},
        up: {{ x: 0.0, y: 1.0, z: 0.0 }},
      }},
    }},
  }});

  await Promise.resolve();
  flushScheduledTimers();
  await Promise.resolve();

  process.stdout.write(JSON.stringify({{
    iframeCameraFromPlotly,
    peerPlotlyCamera: graphContainers["graph-b"].__plotlyElement.layout.scene.camera,
  }}));
}}

main().catch((error) => {{
  process.stderr.write(String(error && error.stack ? error.stack : error));
  process.exit(1);
}});
"""


def test_build_plotly_threejs_camera_sync_script_handles_dash_graph_wrapper(
    tmp_path: Path,
) -> None:
    """Execute the mixed relay against a Dash-style graph wrapper DOM.

    Args:
        tmp_path: Temporary directory for the Node harness file.

    Returns:
        None.
    """

    assert isinstance(tmp_path, Path), f"{type(tmp_path)=}"

    script_source = build_plotly_threejs_camera_sync_script(
        graph_ids=["graph-a", "graph-b"],
        iframe_ids=["iframe-a"],
        camera_sync_group="mesh-sync",
    )
    harness_path = tmp_path / "plotly_threejs_harness.js"
    harness_path.write_text(
        _build_plotly_threejs_harness_script(script_source=script_source)
    )

    completed_process = subprocess.run(
        args=["node", str(harness_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, (
        "Expected the mixed sync Node harness to succeed. "
        f"{completed_process.returncode=} {completed_process.stderr=}"
    )
    payload = json.loads(completed_process.stdout)
    assert payload["relayoutCountsAfterFirstBroadcast"]["graph-a"] == 0, f"{payload=}"
    assert payload["relayoutCountsAfterFirstBroadcast"]["graph-b"] >= 1, f"{payload=}"
    assert (
        payload["relayoutCounts"]["graph-b"]
        == payload["relayoutCountsAfterFirstBroadcast"]["graph-b"]
    ), f"{payload=}"
    assert payload["peerCamera"] == payload["sourceCamera"], f"{payload=}"
    assert payload["iframeCamera"] == {
        "eye": {"x": 20.0, "y": 35.0, "z": 50.0},
        "center": {"x": 11.0, "y": 21.5, "z": 32.0},
        "up": {"x": 0.0, "y": 1.0, "z": 0.0},
    }, f"{payload=}"


def test_build_plotly_threejs_camera_sync_script_converts_plotly_camera_at_the_plotly_boundary(
    tmp_path: Path,
) -> None:
    """Require Plotly cameras to convert to and from world coordinates at the Plotly path.

    Args:
        tmp_path: Temporary directory for the Node harness file.

    Returns:
        None.
    """

    assert isinstance(tmp_path, Path), f"{type(tmp_path)=}"

    script_source = build_plotly_threejs_camera_sync_script(
        graph_ids=["graph-a", "graph-b"],
        iframe_ids=["iframe-a"],
        camera_sync_group="mesh-sync",
    )
    harness_path = tmp_path / "plotly_threejs_world_camera_harness.js"
    harness_path.write_text(
        _build_plotly_threejs_world_camera_harness_script(script_source=script_source)
    )

    completed_process = subprocess.run(
        args=["node", str(harness_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, (
        "Expected the world-camera Plotly/Three.js Node harness to succeed. "
        f"{completed_process.returncode=} {completed_process.stderr=}"
    )
    payload = json.loads(completed_process.stdout)
    assert payload["iframeCameraFromPlotly"] == {
        "eye": {"x": 20.0, "y": 35.0, "z": 50.0},
        "center": {"x": 11.0, "y": 21.5, "z": 32.0},
        "up": {"x": 0.0, "y": 1.0, "z": 0.0},
    }, f"{payload=}"
    assert payload["peerPlotlyCamera"] == {
        "eye": {"x": 2.0, "y": 3.0, "z": 4.0},
        "center": {"x": 0.2, "y": 0.3, "z": 0.4},
        "up": {"x": 0.0, "y": 1.0, "z": 0.0},
    }, f"{payload=}"
