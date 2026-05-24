const container = document.getElementById("mesh-root");
const viewerConfig = {
  positionValues: __POSITION_VALUES_JSON__,
  uvValues: __UV_VALUES_JSON__,
  textureDataUrl: __TEXTURE_DATA_URL_JSON__,
  meshViewBounds: __MESH_VIEW_BOUNDS_JSON__,
};

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.NoToneMapping;
renderer.setClearColor(0xf5f7fb, 1.0);
container.appendChild(renderer.domElement);
const cameraPersistenceStorageKey = "mesh-display-camera:" + __VIEWER_ID_JSON__;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf5f7fb);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000.0);
camera.rotation.order = "YXZ";
const controls = createTrackballMeshCameraControls(camera, renderer.domElement);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute(
  "position",
  new THREE.Float32BufferAttribute(viewerConfig.positionValues, 3),
);
geometry.setAttribute(
  "uv",
  new THREE.Float32BufferAttribute(viewerConfig.uvValues, 2),
);
geometry.computeVertexNormals();

const texture = new THREE.TextureLoader().load(viewerConfig.textureDataUrl);
texture.encoding = THREE.sRGBEncoding;
texture.flipY = true; // Default Three.js value.
const material = new THREE.MeshBasicMaterial({
  map: texture,
  side: THREE.DoubleSide,
  toneMapped: false,
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

function isFiniteVector3(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    Number.isFinite(value.x) &&
    Number.isFinite(value.y) &&
    Number.isFinite(value.z)
  );
}

function isViewerCameraState(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    isFiniteVector3(value.eye) &&
    isFiniteVector3(value.center) &&
    isFiniteVector3(value.up)
  );
}

function isLegacyViewerCameraState(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    isFiniteVector3(value.position) &&
    isFiniteVector3(value.rotation)
  );
}

function convertLegacyViewerCameraState(cameraState) {
  const legacyCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000.0);
  legacyCamera.rotation.order = "YXZ";
  legacyCamera.position.set(
    cameraState.position.x,
    cameraState.position.y,
    cameraState.position.z,
  );
  legacyCamera.rotation.set(
    cameraState.rotation.x,
    cameraState.rotation.y,
    cameraState.rotation.z,
  );
  const direction = new THREE.Vector3();
  legacyCamera.getWorldDirection(direction);
  return {
    eye: {
      x: legacyCamera.position.x,
      y: legacyCamera.position.y,
      z: legacyCamera.position.z,
    },
    center: {
      x: legacyCamera.position.x + direction.x,
      y: legacyCamera.position.y + direction.y,
      z: legacyCamera.position.z + direction.z,
    },
    up: {
      x: legacyCamera.up.x,
      y: legacyCamera.up.y,
      z: legacyCamera.up.z,
    },
  };
}

function readPersistedCameraState() {
  if (window.localStorage === undefined) {
    return null;
  }
  try {
    const serializedCameraState = window.localStorage.getItem(
      cameraPersistenceStorageKey,
    );
    if (serializedCameraState === null) {
      return null;
    }
    const cameraState = JSON.parse(serializedCameraState);
    if (isViewerCameraState(cameraState)) {
      return cameraState;
    }
    if (isLegacyViewerCameraState(cameraState)) {
      return convertLegacyViewerCameraState(cameraState);
    }
    return null;
  } catch (_error) {
    return null;
  }
}

function buildCurrentCameraState() {
  const direction = new THREE.Vector3();
  camera.getWorldDirection(direction);
  return {
    eye: {
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    },
    center: {
      x: camera.position.x + direction.x,
      y: camera.position.y + direction.y,
      z: camera.position.z + direction.z,
    },
    up: {
      x: camera.up.x,
      y: camera.up.y,
      z: camera.up.z,
    },
  };
}

function writePersistedCameraState() {
  if (window.localStorage === undefined) {
    return;
  }
  const cameraState = buildCurrentCameraState();
  window.localStorage.setItem(
    cameraPersistenceStorageKey,
    JSON.stringify(cameraState),
  );
}

window.__threejsCameraSyncViewBounds = viewerConfig.meshViewBounds;
function buildDefaultCameraState() {
  const defaultCameraEyeZ =
    __DEFAULT_CAMERA_EYE_Z_JSON__ *
    viewerConfig.meshViewBounds.camera_coordinate_scale;
  const framingCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000.0);
  framingCamera.rotation.order = "YXZ";
  framingCamera.position.set(
    viewerConfig.meshViewBounds.center.x,
    viewerConfig.meshViewBounds.center.y,
    viewerConfig.meshViewBounds.center.z + defaultCameraEyeZ,
  );
  framingCamera.lookAt(
    viewerConfig.meshViewBounds.center.x,
    viewerConfig.meshViewBounds.center.y,
    viewerConfig.meshViewBounds.center.z,
  );
  return {
    eye: {
      x: framingCamera.position.x,
      y: framingCamera.position.y,
      z: framingCamera.position.z,
    },
    center: {
      x: viewerConfig.meshViewBounds.center.x,
      y: viewerConfig.meshViewBounds.center.y,
      z: viewerConfig.meshViewBounds.center.z,
    },
    up: {
      x: framingCamera.up.x,
      y: framingCamera.up.y,
      z: framingCamera.up.z,
    },
  };
}

function applyViewerCameraState(cameraState) {
  camera.position.set(
    cameraState.eye.x,
    cameraState.eye.y,
    cameraState.eye.z,
  );
  camera.up.set(cameraState.up.x, cameraState.up.y, cameraState.up.z);
  camera.lookAt(cameraState.center.x, cameraState.center.y, cameraState.center.z);
}
const initialCameraState = readPersistedCameraState() || buildDefaultCameraState();
applyViewerCameraState(initialCameraState);
controls.update();
controls.addEventListener("change", writePersistedCameraState);
window.addEventListener("beforeunload", writePersistedCameraState);

__CAMERA_SYNC_SCRIPT__

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}

function render() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

function createTrackballMeshCameraControls(camera, domElement) {
  const listeners = [];
  const pointerState = {
    active: false,
    mode: "rotate",
    lastX: 0,
    lastY: 0,
  };

  function emitChange() {
    for (const listener of listeners) {
      listener();
    }
  }

  domElement.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });
  domElement.addEventListener("mousedown", (event) => {
    pointerState.active = true;
    pointerState.mode =
      event.shiftKey || event.button === 1 || event.button === 2
        ? "pan"
        : "rotate";
    pointerState.lastX = event.clientX;
    pointerState.lastY = event.clientY;
  });
  window.addEventListener("mouseup", () => {
    pointerState.active = false;
  });
  window.addEventListener("mousemove", (event) => {
    if (!pointerState.active) {
      return;
    }
    const dx = event.clientX - pointerState.lastX;
    const dy = event.clientY - pointerState.lastY;
    pointerState.lastX = event.clientX;
    pointerState.lastY = event.clientY;
    if (pointerState.mode === "pan") {
      const panScale = 0.002 * viewerConfig.meshViewBounds.camera_coordinate_scale;
      camera.translateX(-dx * panScale);
      camera.translateY(dy * panScale);
    } else {
      camera.rotation.y -= dx * 0.005;
      camera.rotation.x -= dy * 0.005;
    }
    emitChange();
  });
  domElement.addEventListener("wheel", (event) => {
    event.preventDefault();
    camera.translateZ(
      event.deltaY * 0.01 * viewerConfig.meshViewBounds.camera_coordinate_scale,
    );
    emitChange();
  });

  return {
    addEventListener: (type, listener) => {
      if (type !== "change") {
        return;
      }
      listeners.push(listener);
    },
    update: () => {},
  };
}

window.addEventListener("resize", resize);
resize();
render();
