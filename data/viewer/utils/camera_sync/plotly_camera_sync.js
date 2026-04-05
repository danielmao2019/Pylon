(function () {
  const GRAPH_IDS = __GRAPH_IDS_JSON__;
  const GRAPH_ID_TYPE = __GRAPH_ID_TYPE_JSON__;
  const CAMERA_STORE_ID = __CAMERA_STORE_ID_JSON__;

  function cloneCamera(camera) {
    return JSON.parse(JSON.stringify(camera));
  }

  function isVector3(value) {
    return (
      value !== null &&
      typeof value === "object" &&
      Number.isFinite(value.x) &&
      Number.isFinite(value.y) &&
      Number.isFinite(value.z)
    );
  }

  function isCameraState(value) {
    return (
      value !== null &&
      typeof value === "object" &&
      isVector3(value.center) &&
      isVector3(value.up) &&
      isVector3(value.eye)
    );
  }

  function serializeCamera(camera) {
    return JSON.stringify(cloneCamera(camera));
  }

  function resolvePlotlyGraphElement(graphId) {
    const rootElement = document.getElementById(graphId);
    if (!rootElement) {
      return null;
    }
    if (rootElement.classList && rootElement.classList.contains("js-plotly-plot")) {
      return rootElement;
    }
    return rootElement.querySelector(".js-plotly-plot");
  }

  function compareGraphEntries(leftEntry, rightEntry) {
    const leftPatternId = tryParsePatternGraphId(leftEntry.graphId);
    const rightPatternId = tryParsePatternGraphId(rightEntry.graphId);
    const leftIndex =
      leftPatternId && Number.isFinite(leftPatternId.index) ? leftPatternId.index : null;
    const rightIndex =
      rightPatternId && Number.isFinite(rightPatternId.index) ? rightPatternId.index : null;
    if (leftIndex !== null && rightIndex !== null) {
      return leftIndex - rightIndex;
    }
    return leftEntry.graphId.localeCompare(rightEntry.graphId);
  }

  function resolvePatternGraphEntries() {
    return Array.from(document.querySelectorAll("[id]"))
      .map((rootElement) => {
        const patternId = tryParsePatternGraphId(rootElement.id);
        if (!patternId || patternId.type !== GRAPH_ID_TYPE) {
          return null;
        }
        const plotlyElement =
          rootElement.classList && rootElement.classList.contains("js-plotly-plot")
            ? rootElement
            : rootElement.querySelector(".js-plotly-plot");
        if (!plotlyElement || !plotlyElement._fullLayout) {
          return null;
        }
        return {
          graphId: rootElement.id,
          plotlyElement,
        };
      })
      .filter((entry) => entry !== null)
      .sort(compareGraphEntries);
  }

  function tryParsePatternGraphId(graphId) {
    if (typeof graphId !== "string") {
      return null;
    }
    try {
      const parsedGraphId = JSON.parse(graphId);
      if (parsedGraphId === null || typeof parsedGraphId !== "object") {
        return null;
      }
      return parsedGraphId;
    } catch (error) {
      return null;
    }
  }

  function resolveGraphEntries() {
    if (GRAPH_IDS !== null) {
      return GRAPH_IDS
        .map((graphId) => ({
          graphId,
          plotlyElement: resolvePlotlyGraphElement(graphId),
        }))
        .filter((entry) => entry.plotlyElement && entry.plotlyElement._fullLayout);
    }
    return resolvePatternGraphEntries();
  }

  function readRenderedCamera(graphElement) {
    const scene = graphElement && graphElement._fullLayout && graphElement._fullLayout.scene;
    const glScene = scene && scene._scene;
    if (!glScene || typeof glScene.getCamera !== "function") {
      return null;
    }
    return cloneCamera(glScene.getCamera());
  }

  function readCurrentCamera(graphElement) {
    if (!graphElement || !graphElement._fullLayout || !graphElement._fullLayout.scene) {
      return null;
    }
    return cloneCamera(graphElement._fullLayout.scene.camera);
  }

  function readEventCamera(eventData) {
    if (!eventData || !eventData["scene.camera"]) {
      return null;
    }
    return cloneCamera(eventData["scene.camera"]);
  }

  function writeSharedCamera(camera) {
    if (CAMERA_STORE_ID === null) {
      return;
    }
    if (!isCameraState(camera)) {
      return;
    }
    if (!window.dash_clientside || typeof window.dash_clientside.set_props !== "function") {
      return;
    }
    window.dash_clientside.set_props(CAMERA_STORE_ID, {
      data: cloneCamera(camera),
    });
  }

  function shouldSyncCamera(eventData) {
    if (!eventData) {
      return false;
    }
    return Object.keys(eventData).some((key) => key.indexOf("scene.camera") === 0);
  }

  function readPendingCameraSignatures(graphElement) {
    const rawPendingCameraSignatures =
      graphElement.dataset.cameraSyncPendingCameraSignatures;
    if (!rawPendingCameraSignatures) {
      return [];
    }
    try {
      const parsedPendingCameraSignatures = JSON.parse(rawPendingCameraSignatures);
      if (!Array.isArray(parsedPendingCameraSignatures)) {
        return [];
      }
      return parsedPendingCameraSignatures.filter((signature) => typeof signature === "string");
    } catch (error) {
      return [];
    }
  }

  function writePendingCameraSignatures(graphElement, pendingCameraSignatures) {
    graphElement.dataset.cameraSyncPendingCameraSignatures = JSON.stringify(
      pendingCameraSignatures,
    );
  }

  function queuePendingCameraSignature(graphElement, camera) {
    const pendingCameraSignatures = readPendingCameraSignatures(graphElement);
    pendingCameraSignatures.push(serializeCamera(camera));
    writePendingCameraSignatures(graphElement, pendingCameraSignatures);
  }

  function shouldSuppressGraphEvent(graphElement, camera) {
    const pendingCameraSignatures = readPendingCameraSignatures(graphElement);
    const cameraSignature = serializeCamera(camera);
    const pendingCameraIndex = pendingCameraSignatures.indexOf(cameraSignature);
    if (pendingCameraIndex < 0) {
      return false;
    }
    pendingCameraSignatures.splice(pendingCameraIndex, 1);
    writePendingCameraSignatures(graphElement, pendingCameraSignatures);
    return true;
  }

  function applyCameraToGraphs(camera, sourceGraphId, includeSourceGraph) {
    resolveGraphEntries().forEach((graphEntry) => {
      if (!includeSourceGraph && graphEntry.graphId === sourceGraphId) {
        return;
      }
      const graphElement = graphEntry.plotlyElement;
      if (!graphElement || !graphElement._fullLayout) {
        return;
      }
      queuePendingCameraSignature(graphElement, camera);
      Plotly.relayout(graphElement, {
        "scene.camera": cloneCamera(camera),
      });
    });
  }

  function readWheelSyncDeadlineMs(graphElement) {
    const rawWheelSyncDeadlineMs = graphElement.dataset.cameraSyncWheelDeadlineMs;
    if (!rawWheelSyncDeadlineMs) {
      return 0;
    }
    const parsedWheelSyncDeadlineMs = Number(rawWheelSyncDeadlineMs);
    if (!Number.isFinite(parsedWheelSyncDeadlineMs)) {
      return 0;
    }
    return parsedWheelSyncDeadlineMs;
  }

  function writeWheelSyncDeadlineMs(graphElement, wheelSyncDeadlineMs) {
    graphElement.dataset.cameraSyncWheelDeadlineMs = String(wheelSyncDeadlineMs);
  }

  function readWheelLastBroadcastSignature(graphElement) {
    return graphElement.dataset.cameraSyncWheelLastBroadcastSignature || "";
  }

  function writeWheelLastBroadcastSignature(graphElement, cameraSignature) {
    graphElement.dataset.cameraSyncWheelLastBroadcastSignature = cameraSignature;
  }

  function clearWheelSyncLoop(graphElement) {
    graphElement.dataset.cameraSyncWheelLoopActive = "false";
    delete graphElement.dataset.cameraSyncWheelDeadlineMs;
  }

  function pumpWheelSync(graphElement, graphId) {
    const renderedCamera = readRenderedCamera(graphElement);
    if (renderedCamera) {
      const renderedCameraSignature = serializeCamera(renderedCamera);
      if (renderedCameraSignature !== readWheelLastBroadcastSignature(graphElement)) {
        writeWheelLastBroadcastSignature(graphElement, renderedCameraSignature);
        applyCameraToGraphs(renderedCamera, graphId, false);
      }
    }

    if (Date.now() < readWheelSyncDeadlineMs(graphElement)) {
      window.requestAnimationFrame(() => {
        pumpWheelSync(graphElement, graphId);
      });
      return;
    }

    const finalRenderedCamera = readRenderedCamera(graphElement);
    if (finalRenderedCamera) {
      const finalRenderedCameraSignature = serializeCamera(finalRenderedCamera);
      writeWheelLastBroadcastSignature(graphElement, finalRenderedCameraSignature);
      writeSharedCamera(finalRenderedCamera);
      applyCameraToGraphs(finalRenderedCamera, graphId, true);
    }

    clearWheelSyncLoop(graphElement);
  }

  function scheduleWheelSync(graphElement, graphId) {
    writeWheelSyncDeadlineMs(graphElement, Date.now() + 250);
    if (graphElement.dataset.cameraSyncWheelLoopActive === "true") {
      return;
    }
    graphElement.dataset.cameraSyncWheelLoopActive = "true";
    window.requestAnimationFrame(() => {
      pumpWheelSync(graphElement, graphId);
    });
  }

  function attachWheelSync(graphElement, graphId) {
    if (!graphElement || graphElement.dataset.cameraSyncWheelAttached === "true") {
      return;
    }
    const canvas = graphElement.querySelector(".gl-container canvas");
    if (!canvas) {
      return;
    }
    graphElement.dataset.cameraSyncWheelAttached = "true";
    canvas.addEventListener("wheel", () => {
      scheduleWheelSync(graphElement, graphId);
    });
  }

  function attachSync() {
    const graphEntries = resolveGraphEntries();

    if (GRAPH_IDS !== null && graphEntries.length !== GRAPH_IDS.length) {
      return;
    }
    if (GRAPH_IDS === null && graphEntries.length < 2) {
      return;
    }

    graphEntries.forEach((graphEntry) => {
      const graphElement = graphEntry.plotlyElement;
      attachWheelSync(graphElement, graphEntry.graphId);
      if (graphElement.dataset.cameraSyncAttached === "true") {
        return;
      }

      graphElement.dataset.cameraSyncAttached = "true";
      graphElement.on("plotly_relayout", (eventData) => {
        if (!shouldSyncCamera(eventData)) {
          return;
        }
        const sceneCamera = readEventCamera(eventData) || readCurrentCamera(graphElement);
        if (!sceneCamera) {
          return;
        }
        if (shouldSuppressGraphEvent(graphElement, sceneCamera)) {
          return;
        }
        writeWheelLastBroadcastSignature(graphElement, serializeCamera(sceneCamera));
        writeSharedCamera(sceneCamera);
        applyCameraToGraphs(sceneCamera, graphEntry.graphId, true);
      });
    });
  }

  window.setInterval(attachSync, 500);
})();
