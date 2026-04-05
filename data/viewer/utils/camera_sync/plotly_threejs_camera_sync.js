(function() {
  if (window.__plotlyThreejsCameraSyncInstalled) {
    return;
  }
  window.__plotlyThreejsCameraSyncInstalled = true;

  const cameraMatchTolerance = 1.0e-4;
  const graphIds = __GRAPH_IDS_JSON__;
  const iframeIds = __IFRAME_IDS_JSON__;
  const cameraSyncGroup = __CAMERA_SYNC_GROUP_JSON__;
  const syncState = {
    isApplying: false,
    initialCameraSynchronized: false,
    lastWorldCameraState: null,
  };

  // === Camera state primitives ===

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
      isVector3(value.eye) &&
      isVector3(value.center) &&
      isVector3(value.up)
    );
  }

  function cloneCameraState(cameraState) {
    return JSON.parse(JSON.stringify(cameraState));
  }

  function isViewBounds(value) {
    return (
      value !== null &&
      typeof value === "object" &&
      isVector3(value.center) &&
      Number.isFinite(value.camera_coordinate_scale) &&
      value.camera_coordinate_scale > 0.0
    );
  }

  function cameraStatesMatch(leftCameraState, rightCameraState) {
    if (!isCameraState(leftCameraState) || !isCameraState(rightCameraState)) {
      return false;
    }
    const coordinatePairs = [
      [leftCameraState.eye.x, rightCameraState.eye.x],
      [leftCameraState.eye.y, rightCameraState.eye.y],
      [leftCameraState.eye.z, rightCameraState.eye.z],
      [leftCameraState.center.x, rightCameraState.center.x],
      [leftCameraState.center.y, rightCameraState.center.y],
      [leftCameraState.center.z, rightCameraState.center.z],
      [leftCameraState.up.x, rightCameraState.up.x],
      [leftCameraState.up.y, rightCameraState.up.y],
      [leftCameraState.up.z, rightCameraState.up.z],
    ];
    const coordinatesMatch = coordinatePairs.every(([leftValue, rightValue]) => {
      return Math.abs(leftValue - rightValue) <= cameraMatchTolerance;
    });
    if (!coordinatesMatch) {
      return false;
    }
    if (
      Number.isFinite(leftCameraState.fovyRadians) &&
      Number.isFinite(rightCameraState.fovyRadians)
    ) {
      return (
        Math.abs(leftCameraState.fovyRadians - rightCameraState.fovyRadians) <=
        cameraMatchTolerance
      );
    }
    return true;
  }

  function normalizePlotlyFovyRadians(rawFovy) {
    if (!Number.isFinite(rawFovy)) {
      return null;
    }
    if (rawFovy > Math.PI) {
      return rawFovy * Math.PI / 180.0;
    }
    return rawFovy;
  }

  function attachPlotlyFovyRadians(cameraState, plotlyElement) {
    if (!isCameraState(cameraState)) {
      return null;
    }
    const normalizedCameraState = cloneCameraState(cameraState);
    const fovyRadians = normalizePlotlyFovyRadians(
      plotlyElement?._fullLayout?.scene?._scene?.glplot?.fovy,
    );
    if (Number.isFinite(fovyRadians)) {
      normalizedCameraState.fovyRadians = fovyRadians;
    }
    return normalizedCameraState;
  }

  // === Plotly camera conversion ===

  function readPlotlyViewBounds(plotlyElement) {
    const metaViewBounds =
      plotlyElement?.layout?.meta?.meshViewBounds ||
      plotlyElement?._fullLayout?.meta?.meshViewBounds ||
      null;
    if (!isViewBounds(metaViewBounds)) {
      return null;
    }
    return metaViewBounds;
  }

  function buildWorldPointFromPlotlyPoint(plotlyPoint, viewBounds) {
    if (!isVector3(plotlyPoint) || !isViewBounds(viewBounds)) {
      return null;
    }
    return {
      x:
        viewBounds.center.x +
        plotlyPoint.x * viewBounds.camera_coordinate_scale,
      y:
        viewBounds.center.y +
        plotlyPoint.y * viewBounds.camera_coordinate_scale,
      z:
        viewBounds.center.z +
        plotlyPoint.z * viewBounds.camera_coordinate_scale,
    };
  }

  function buildPlotlyPointFromWorldPoint(worldPoint, viewBounds) {
    if (!isVector3(worldPoint) || !isViewBounds(viewBounds)) {
      return null;
    }
    return {
      x:
        (worldPoint.x - viewBounds.center.x) / viewBounds.camera_coordinate_scale,
      y:
        (worldPoint.y - viewBounds.center.y) / viewBounds.camera_coordinate_scale,
      z:
        (worldPoint.z - viewBounds.center.z) / viewBounds.camera_coordinate_scale,
    };
  }

  function buildWorldCameraStateFromPlotlyCamera(plotlyCameraState, plotlyElement) {
    if (!isCameraState(plotlyCameraState)) {
      return null;
    }
    const viewBounds = readPlotlyViewBounds(plotlyElement);
    if (!isViewBounds(viewBounds)) {
      return null;
    }
    const worldEye = buildWorldPointFromPlotlyPoint(
      plotlyCameraState.eye,
      viewBounds,
    );
    const worldCenter = buildWorldPointFromPlotlyPoint(
      plotlyCameraState.center,
      viewBounds,
    );
    if (!isVector3(worldEye) || !isVector3(worldCenter)) {
      return null;
    }
    const worldCameraState = cloneCameraState(plotlyCameraState);
    worldCameraState.eye = worldEye;
    worldCameraState.center = worldCenter;
    return worldCameraState;
  }

  function buildPlotlyCameraFromWorldCamera(worldCameraState, plotlyElement) {
    if (!isCameraState(worldCameraState)) {
      return null;
    }
    const viewBounds = readPlotlyViewBounds(plotlyElement);
    if (!isViewBounds(viewBounds)) {
      return null;
    }
    const plotlyEye = buildPlotlyPointFromWorldPoint(
      worldCameraState.eye,
      viewBounds,
    );
    const plotlyCenter = buildPlotlyPointFromWorldPoint(
      worldCameraState.center,
      viewBounds,
    );
    if (!isVector3(plotlyEye) || !isVector3(plotlyCenter)) {
      return null;
    }
    const plotlyCameraState = cloneCameraState(worldCameraState);
    plotlyCameraState.eye = plotlyEye;
    plotlyCameraState.center = plotlyCenter;
    return plotlyCameraState;
  }

  function buildWorldCameraStateFromThreejsState(threejsCameraState) {
    if (!isCameraState(threejsCameraState)) {
      return null;
    }
    return cloneCameraState(threejsCameraState);
  }

  function buildThreejsCameraStateFromWorldCamera(worldCameraState) {
    if (!isCameraState(worldCameraState)) {
      return null;
    }
    return cloneCameraState(worldCameraState);
  }

  // === Plotly camera readers ===

  function resolvePlotlyElement(graphId) {
    const graphContainer = document.getElementById(graphId);
    if (graphContainer === null) {
      return null;
    }
    if (typeof graphContainer.on === "function") {
      return graphContainer;
    }
    if (typeof graphContainer.querySelector !== "function") {
      return null;
    }
    const plotlyElement = graphContainer.querySelector(".js-plotly-plot");
    if (plotlyElement === null || typeof plotlyElement.on !== "function") {
      return null;
    }
    return plotlyElement;
  }

  function readPlotlyCamera(plotlyElement) {
    if (plotlyElement === null) {
      return null;
    }
    const renderedCamera = plotlyElement._fullLayout?.scene?._scene?.getCamera?.();
    if (isCameraState(renderedCamera)) {
      return attachPlotlyFovyRadians(renderedCamera, plotlyElement);
    }
    const layoutCamera =
      plotlyElement.layout?.scene?.camera ||
      plotlyElement._fullLayout?.scene?.camera ||
      null;
    if (!isCameraState(layoutCamera)) {
      return null;
    }
    return attachPlotlyFovyRadians(layoutCamera, plotlyElement);
  }

  function readPlotlyWorldCamera(plotlyElement) {
    const plotlyCameraState = readPlotlyCamera(plotlyElement);
    if (!isCameraState(plotlyCameraState)) {
      return null;
    }
    return buildWorldCameraStateFromPlotlyCamera(plotlyCameraState, plotlyElement);
  }

  function buildOrderedGraphIds(preferredGraphId) {
    const orderedGraphIds = [];
    if (typeof preferredGraphId === "string" && graphIds.includes(preferredGraphId)) {
      orderedGraphIds.push(preferredGraphId);
    }
    graphIds.forEach((graphId) => {
      if (!orderedGraphIds.includes(graphId)) {
        orderedGraphIds.push(graphId);
      }
    });
    return orderedGraphIds;
  }

  function readAuthoritativePlotlyWorldCamera(preferredGraphId) {
    for (const graphId of buildOrderedGraphIds(preferredGraphId)) {
      const plotlyElement = resolvePlotlyElement(graphId);
      const worldCameraState = readPlotlyWorldCamera(plotlyElement);
      if (isCameraState(worldCameraState)) {
        return worldCameraState;
      }
    }
    return null;
  }

  // === Cross-renderer message helpers ===

  function buildThreejsApplyMessage({threejsCameraState, sourceViewerId}) {
    return {
      type: "threejs-camera-sync-apply",
      cameraSyncGroup,
      sourceViewerId,
      cameraState: threejsCameraState,
    };
  }

  function isThreejsCameraChangeMessage(message) {
    return (
      message !== null &&
      typeof message === "object" &&
      message.type === "threejs-camera-sync-change" &&
      message.cameraSyncGroup === cameraSyncGroup &&
      typeof message.sourceViewerId === "string" &&
      message.sourceViewerId !== ""
    );
  }

  function buildWorldCameraStateFromMessage(message) {
    if (!isThreejsCameraChangeMessage(message)) {
      return null;
    }
    return buildWorldCameraStateFromThreejsState(message.cameraState);
  }

  // === Camera application helpers ===

  function applyWorldCameraStateToPlotlyGraphs(
    worldCameraState,
    sourceViewerId,
  ) {
    const relayoutPromises = [];
    graphIds.forEach((graphId) => {
      if (sourceViewerId !== null && graphId === sourceViewerId) {
        return;
      }
      const plotlyElement = resolvePlotlyElement(graphId);
      if (plotlyElement === null || window.Plotly === undefined) {
        return;
      }
      const plotlyCameraState = buildPlotlyCameraFromWorldCamera(
        worldCameraState,
        plotlyElement,
      );
      if (!isCameraState(plotlyCameraState)) {
        return;
      }
      relayoutPromises.push(
        window.Plotly.relayout(plotlyElement, {
          "scene.camera": plotlyCameraState,
        }),
      );
    });
    return relayoutPromises;
  }

  function applyWorldCameraStateToIframes(worldCameraState, sourceViewerId) {
    iframeIds.forEach((iframeId) => {
      if (sourceViewerId !== null && iframeId === sourceViewerId) {
        return;
      }
      const iframeElement = document.getElementById(iframeId);
      if (iframeElement === null || iframeElement.contentWindow === null) {
        return;
      }
      const threejsCameraState = buildThreejsCameraStateFromWorldCamera(
        worldCameraState,
      );
      if (!isCameraState(threejsCameraState)) {
        return;
      }
      iframeElement.contentWindow.postMessage(
        buildThreejsApplyMessage({
          threejsCameraState,
          sourceViewerId,
        }),
        "*",
      );
    });
  }

  function applyWorldCameraStateToPeers(worldCameraState, sourceViewerId) {
    const relayoutPromises = applyWorldCameraStateToPlotlyGraphs(
      worldCameraState,
      sourceViewerId,
    );
    applyWorldCameraStateToIframes(worldCameraState, sourceViewerId);
    return relayoutPromises;
  }

  // === Synchronization orchestration ===

  function shouldSynchronizeFromPlotlyAuthority(sourceViewerId) {
    return graphIds.includes(sourceViewerId);
  }

  function synchronizeFromPlotlyAuthority(sourceViewerId) {
    if (!shouldSynchronizeFromPlotlyAuthority(sourceViewerId)) {
      return Promise.resolve();
    }
    const authoritativeWorldCamera = readAuthoritativePlotlyWorldCamera(sourceViewerId);
    if (!isCameraState(authoritativeWorldCamera)) {
      return Promise.resolve();
    }
    if (cameraStatesMatch(authoritativeWorldCamera, syncState.lastWorldCameraState)) {
      return Promise.resolve();
    }
    syncState.lastWorldCameraState = cloneCameraState(authoritativeWorldCamera);
    return Promise.allSettled(
      applyWorldCameraStateToPeers(authoritativeWorldCamera, sourceViewerId),
    );
  }

  function broadcastWorldCameraState(worldCameraState, sourceViewerId) {
    if (syncState.isApplying || !isCameraState(worldCameraState)) {
      return;
    }
    if (
      cameraStatesMatch(
        worldCameraState,
        syncState.lastWorldCameraState,
      )
    ) {
      return;
    }
    syncState.isApplying = true;
    syncState.lastWorldCameraState = cloneCameraState(worldCameraState);
    Promise.allSettled(
      applyWorldCameraStateToPeers(worldCameraState, sourceViewerId),
    )
      .then(() => synchronizeFromPlotlyAuthority(sourceViewerId))
      .finally(() => {
        window.setTimeout(() => {
          syncState.isApplying = false;
        }, 0);
      });
  }

  function handlePlotlyCameraChange(plotlyElement, sourceGraphId) {
    if (syncState.isApplying) {
      return;
    }
    const worldCameraState = readPlotlyWorldCamera(plotlyElement);
    if (!isCameraState(worldCameraState)) {
      return;
    }
    if (cameraStatesMatch(worldCameraState, syncState.lastWorldCameraState)) {
      return;
    }
    broadcastWorldCameraState(worldCameraState, sourceGraphId);
  }

  // === Event wiring ===

  function attachGraphListener(graphId) {
    const plotlyElement = resolvePlotlyElement(graphId);
    if (
      plotlyElement === null ||
      plotlyElement.__plotlyThreejsCameraSyncAttached
    ) {
      return;
    }
    plotlyElement.__plotlyThreejsCameraSyncAttached = true;
    plotlyElement.on("plotly_relayout", () => {
      handlePlotlyCameraChange(plotlyElement, graphId);
    });
  }

  function handleThreejsCameraChange(message) {
    const worldCameraState = buildWorldCameraStateFromMessage(message);
    if (!isCameraState(worldCameraState)) {
      return;
    }
    if (cameraStatesMatch(worldCameraState, syncState.lastWorldCameraState)) {
      return;
    }
    broadcastWorldCameraState(worldCameraState, message.sourceViewerId);
  }

  window.addEventListener("message", (event) => {
    handleThreejsCameraChange(event.data);
  });

  function allTargetsMounted() {
    const allGraphsMounted = graphIds.every((graphId) => {
      return resolvePlotlyElement(graphId) !== null;
    });
    const allIframesMounted = iframeIds.every((iframeId) => {
      return document.getElementById(iframeId) !== null;
    });
    return allGraphsMounted && allIframesMounted;
  }

  function synchronizeInitialCamera() {
    if (syncState.initialCameraSynchronized === true) {
      return;
    }
    const initialWorldCamera = readAuthoritativePlotlyWorldCamera(graphIds[0]);
    if (!isCameraState(initialWorldCamera)) {
      return;
    }
    syncState.initialCameraSynchronized = true;
    broadcastWorldCameraState(initialWorldCamera, graphIds[0]);
  }

  function installSynchronization() {
    graphIds.forEach((graphId) => {
      attachGraphListener(graphId);
    });
    if (!allTargetsMounted()) {
      return;
    }
    synchronizeInitialCamera();
    window.clearInterval(installInterval);
  }

  const installInterval = window.setInterval(() => {
    installSynchronization();
  }, 250);
})();
