const cameraMatchTolerance = 1.0e-4;
const cameraSyncConfig = {
  viewerId: __VIEWER_ID_JSON__,
  cameraSyncGroup: __CAMERA_SYNC_GROUP_JSON__,
};
let isApplyingExternalCamera = false;
let pendingCameraBroadcast = null;
let lastAppliedExternalCamera = null;

// === Camera state primitives ===

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

function cloneCameraState(cameraState) {
  return JSON.parse(JSON.stringify(cameraState));
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

// === Viewer camera readers and writers ===

function readCurrentCameraState() {
  return {
    eye: {
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    },
    center: {
      x: controls.target.x,
      y: controls.target.y,
      z: controls.target.z,
    },
    up: {
      x: camera.up.x,
      y: camera.up.y,
      z: camera.up.z,
    },
    fovyRadians: camera.fov * Math.PI / 180.0,
  };
}

function applyCameraState(cameraState) {
  if (!isCameraState(cameraState)) {
    return;
  }
  isApplyingExternalCamera = true;
  camera.position.set(
    cameraState.eye.x,
    cameraState.eye.y,
    cameraState.eye.z,
  );
  camera.up.set(
    cameraState.up.x,
    cameraState.up.y,
    cameraState.up.z,
  );
  controls.target.set(
    cameraState.center.x,
    cameraState.center.y,
    cameraState.center.z,
  );
  if (Number.isFinite(cameraState.fovyRadians)) {
    camera.fov = cameraState.fovyRadians * 180.0 / Math.PI;
    camera.updateProjectionMatrix();
  }
  camera.lookAt(controls.target);
  controls.update();
  lastAppliedExternalCamera = cloneCameraState(readCurrentCameraState());
  window.requestAnimationFrame(() => {
    isApplyingExternalCamera = false;
  });
}

// === Sync message helpers ===

function buildCameraChangeMessage(cameraState) {
  return {
    type: "threejs-camera-sync-change",
    cameraSyncGroup: cameraSyncConfig.cameraSyncGroup,
    sourceViewerId: cameraSyncConfig.viewerId,
    cameraState: cameraState,
  };
}

function shouldSuppressBroadcast(cameraState) {
  if (!cameraStatesMatch(cameraState, lastAppliedExternalCamera)) {
    return false;
  }
  lastAppliedExternalCamera = null;
  return true;
}

// === Event handlers ===

function scheduleCameraBroadcast() {
  if (isApplyingExternalCamera) {
    return;
  }
  if (pendingCameraBroadcast !== null) {
    return;
  }
  pendingCameraBroadcast = window.requestAnimationFrame(() => {
    pendingCameraBroadcast = null;
    const currentCameraState = readCurrentCameraState();
    if (shouldSuppressBroadcast(currentCameraState)) {
      return;
    }
    window.parent.postMessage(buildCameraChangeMessage(currentCameraState), "*");
  });
}

function isExternalCameraApplyMessage(message) {
  return (
    message !== null &&
    typeof message === "object" &&
    message.type === "threejs-camera-sync-apply" &&
    message.cameraSyncGroup === cameraSyncConfig.cameraSyncGroup &&
    message.sourceViewerId !== cameraSyncConfig.viewerId
  );
}

function handleParentMessage(message) {
  if (!isExternalCameraApplyMessage(message)) {
    return;
  }
  applyCameraState(message.cameraState);
}

window.__threejsCameraSync = {
  viewerId: cameraSyncConfig.viewerId,
  cameraSyncGroup: cameraSyncConfig.cameraSyncGroup,
  getCameraState: readCurrentCameraState,
  applyCameraState: applyCameraState,
};

window.addEventListener("message", (event) => {
  handleParentMessage(event.data);
});

controls.addEventListener("change", scheduleCameraBroadcast);
