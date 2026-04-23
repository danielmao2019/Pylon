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
const controls = new THREE.TrackballControls(camera, renderer.domElement);
controls.noPan = false;
controls.noRotate = false;
controls.noZoom = false;
controls.staticMoving = true;

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
    isFiniteVector3(value.position) &&
    isFiniteVector3(value.target)
  );
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
    if (!isViewerCameraState(cameraState)) {
      return null;
    }
    return cameraState;
  } catch (_error) {
    return null;
  }
}

function writePersistedCameraState() {
  if (window.localStorage === undefined) {
    return;
  }
  const cameraState = {
    position: {
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    },
    target: {
      x: controls.target.x,
      y: controls.target.y,
      z: controls.target.z,
    },
  };
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
  return {
    position: {
      x: viewerConfig.meshViewBounds.center.x,
      y: viewerConfig.meshViewBounds.center.y,
      z: viewerConfig.meshViewBounds.center.z + defaultCameraEyeZ,
    },
    target: {
      x: viewerConfig.meshViewBounds.center.x,
      y: viewerConfig.meshViewBounds.center.y,
      z: viewerConfig.meshViewBounds.center.z,
    },
  };
}

function applyViewerCameraState(cameraState) {
  camera.position.set(
    cameraState.position.x,
    cameraState.position.y,
    cameraState.position.z,
  );
  controls.target.set(
    cameraState.target.x,
    cameraState.target.y,
    cameraState.target.z,
  );
}
const initialCameraState = readPersistedCameraState() || buildDefaultCameraState();
applyViewerCameraState(initialCameraState);
camera.lookAt(controls.target.x, controls.target.y, controls.target.z);
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
  controls.handleResize();
}

function render() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

window.addEventListener("resize", resize);
resize();
render();
