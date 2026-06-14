(function () {
  if (window.__threejsCameraSyncRelayAttached === true) {
    return;
  }
  window.__threejsCameraSyncRelayAttached = true;

  function isCameraVector(value) {
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
      isCameraVector(value.eye) &&
      isCameraVector(value.center) &&
      isCameraVector(value.up)
    );
  }

  function listSyncedIframes(cameraSyncGroup) {
    return Array.from(
      document.querySelectorAll(
        'iframe[data-camera-sync-group="' + cameraSyncGroup + '"]',
      ),
    );
  }

  window.addEventListener("message", (event) => {
    const message = event.data;
    if (message === null || typeof message !== "object") {
      return;
    }
    if (message.type !== "threejs-camera-sync-change") {
      return;
    }
    if (typeof message.cameraSyncGroup !== "string" || message.cameraSyncGroup === "") {
      return;
    }
    if (typeof message.sourceViewerId !== "string" || message.sourceViewerId === "") {
      return;
    }
    if (!isCameraState(message.cameraState)) {
      return;
    }

    listSyncedIframes(message.cameraSyncGroup).forEach((iframeElement) => {
      if (
        iframeElement.dataset.cameraSyncViewerId === message.sourceViewerId ||
        iframeElement.contentWindow === null
      ) {
        return;
      }
      iframeElement.contentWindow.postMessage(
        {
          type: "threejs-camera-sync-apply",
          cameraSyncGroup: message.cameraSyncGroup,
          sourceViewerId: message.sourceViewerId,
          cameraState: message.cameraState,
        },
        "*",
      );
    });
  });
})();
