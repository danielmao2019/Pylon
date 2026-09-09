// Node harness driving the roll-lock clientside callback through a scripted gl3d drag.
//
// The harness stands in for the browser, never for the panel: it builds a jsdom document, constructs the SHIPPED gl-plot3d view controller over it - `createCamera` out of the same `plotly.js` release the Dash app serves - hangs it off a graph div shaped the way Plotly shapes one, and puts left-drag mouse events through the container the controller listens on. The rotation a drag turns the camera through, the time-indexed spline it writes that rotation into, the per-frame `idle` that re-pins that spline and the `recalcMatrix` that resolves the pose a frame draws are all the real ones, so a lock that writes its correction at the wrong timestamp loses here for the reason it loses in the browser rather than for a modelled one.
//
// Three browser capabilities jsdom lacks are supplied. Layout: the container reports the size the renderer would give it, since the controller scales pointer travel by it. The clock: `performance.now` is a counter the harness advances one frame at a time, which is what makes a run reproducible and what puts the frames at the spacing a 60fps browser puts them at. Font metrics: a computed `font-size` comes back in the unit it was written in rather than in pixels, and the controller's wheel listener measures its line height by reading one back, so the pixel resolution a browser does is done here.
//
// Usage: node roll_lock_harness.js '<spec-json>', where the spec carries `node_modules_path`, `source_path`, `graph_id`, `lock_roll`, `eye`, `center`, `up`, `drags`, and `reports_each_drag`. One JSON record per drag is written to stdout.

const fs = require("fs");
const path = require("path");

const spec = JSON.parse(process.argv[2]);

// Milliseconds between one rendered frame and the next, which the harness clock advances by.
const FRAME_MILLISECONDS = 16;
// Rendered frames each pointer move is followed by. The controller draws a pose it samples two frames behind the newest keyframe, so this is what carries the drawn frame past the move just written - and every frame in between runs the controller's own idle over the spline the move went into.
const FRAMES_PER_DRAG = 3;
// The width and height the renderer would lay the container out at. The controller scales pointer travel by the height, so a drag of a given pixel travel turns the camera by an angle this fixes.
const CONTAINER_WIDTH = 800;
const CONTAINER_HEIGHT = 600;
// The pixel the drag opens at, far enough inside the container that the whole scripted drag stays within it.
const POINTER_ORIGIN_X = 400;
const POINTER_ORIGIN_Y = 300;
// The font size a browser resolves a computed `font-size` to, in pixels.
const RESOLVED_FONT_SIZE = "16px";

// The harness clock, read by the shipped controller through `performance.now` for every keyframe timestamp it writes and every frame it draws.
let clock = 0;
globalThis.performance = { now: () => clock };

const { JSDOM, VirtualConsole } = require(path.join(spec.node_modules_path, "jsdom"));
// jsdom catches whatever a DOM event listener throws and reports it here instead of letting it out of `dispatchEvent`, so a callback that aborts on the pose a pointer move handed it would otherwise leave the run reporting a full set of records as if nothing had happened. The callback runs inside the controller's own listener for every move of a drag, so failing the run on that report is what makes such an abort visible. The other thing reported here is jsdom naming a browser API it does not implement - the WebGL context the renderer would draw through, which nothing under test reads - and that is left to pass.
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (error) => {
    if (error.type !== "unhandled-exception") {
        return;
    }
    process.stderr.write(String(error.stack) + "\n");
    process.exit(1);
});
const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    virtualConsole: virtualConsole,
});
const domWindow = dom.window;
// The shipped controller reaches for the browser globals through the bare names, so the jsdom window is installed under those names before it is loaded. `performance` is deliberately not among them: jsdom's own implementation reads the global back and would recur into itself, and the harness clock is what belongs there.
for (const globalName of [
    "document",
    "navigator",
    "Element",
    "HTMLElement",
    "HTMLDivElement",
    "Event",
    "MouseEvent",
    "MutationObserver",
]) {
    globalThis[globalName] = domWindow[globalName];
}
globalThis.window = domWindow;
const resolveJsdomStyle = domWindow.getComputedStyle.bind(domWindow);
globalThis.getComputedStyle = function (element) {
    const style = resolveJsdomStyle(element);
    return {
        getPropertyValue: (propertyName) =>
            propertyName === "font-size" ? RESOLVED_FONT_SIZE : style.getPropertyValue(propertyName),
    };
};

const createCamera = require(path.join(spec.node_modules_path, "plotly.js", "stackgl_modules"))
    .gl_plot3d.createCamera;

// The DOM Plotly leaves behind: `dcc.Graph` renders its component id onto a wrapper div, the Plotly graph div is the `.js-plotly-plot` inside it, and the gl3d scene owns a container inside that which the view controller listens on.
const wrapper = domWindow.document.createElement("div");
wrapper.id = spec.graph_id;
domWindow.document.body.appendChild(wrapper);
const graphDiv = domWindow.document.createElement("div");
graphDiv.className = "js-plotly-plot";
wrapper.appendChild(graphDiv);
const container = domWindow.document.createElement("div");
graphDiv.appendChild(container);
Object.defineProperty(container, "clientWidth", { value: CONTAINER_WIDTH });
Object.defineProperty(container, "clientHeight", { value: CONTAINER_HEIGHT });

// The camera Plotly's own `initializeGLCamera` builds, on the options it builds it with.
const camera = createCamera(container, {
    center: spec.center.slice(),
    eye: spec.eye.slice(),
    up: spec.up.slice(),
    _ortho: false,
    zoomMin: 0.01,
    zoomMax: 100,
    mode: "orbit",
});

// The camera Plotly's own `getCamera` reports, which resolves the pose at the newest keyframe before reading it back.
function readCamera() {
    camera.view.recalcMatrix(camera.view.lastT());
    return {
        up: { x: camera.up[0], y: camera.up[1], z: camera.up[2] },
        center: { x: camera.center[0], y: camera.center[1], z: camera.center[2] },
        eye: { x: camera.eye[0], y: camera.eye[1], z: camera.eye[2] },
        projection: { type: "perspective" },
    };
}

graphDiv._fullLayout = { scene: { _scene: { camera: camera, getCamera: readCamera } } };
globalThis.window.dash_clientside = { no_update: null };
globalThis.Plotly = {
    // A camera a relayout hands the panel reaches the view controller the way Plotly's own `setViewport` hands it over, through the controller's own `lookAt`, so a correction issued from outside the drag is subject to exactly the ordering one issued from inside it is.
    relayout: (targetGraphDiv, update) => {
        const eye = update["scene.camera.eye"];
        const up = update["scene.camera.up"];
        camera.lookAt([eye.x, eye.y, eye.z], camera.center.slice(), [up.x, up.y, up.z]);
        return Promise.resolve();
    },
};

function vectorSubtract(left, right) {
    return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function vectorCross(left, right) {
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ];
}

function vectorDot(left, right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function vectorNormalize(vector) {
    const length = Math.sqrt(vectorDot(vector, vector));
    return [vector[0] / length, vector[1] / length, vector[2] / length];
}

const axis = vectorNormalize(spec.lock_roll);

// Reads the roll-lock invariants off the pose the frame just drawn resolved to - the camera right axis's component along the lock axis, the up vector's side of it, and the polar angle that says how close to the pole the camera stands - which is the pose the controller published for the renderer rather than the one it reports between gestures.
function measureCamera() {
    const eye = camera.view.computedEye.slice();
    const center = camera.view.computedCenter.slice();
    const up = camera.view.computedUp.slice();
    const offset = vectorSubtract(eye, center);
    const forward = vectorNormalize(vectorSubtract(center, eye));
    const cameraRightAxis = vectorNormalize(vectorCross(forward, up));
    const components = eye.concat(up).concat(cameraRightAxis);
    return {
        right_along_axis: vectorDot(cameraRightAxis, axis),
        up_along_axis: vectorDot(vectorNormalize(up), axis),
        up_length: Math.sqrt(vectorDot(up, up)),
        camera_right_axis_length: Math.sqrt(vectorDot(cameraRightAxis, cameraRightAxis)),
        polar: Math.acos(Math.max(-1, Math.min(1, vectorDot(vectorNormalize(offset), axis)))),
        eye: eye,
        up: up,
        camera_right_axis: cameraRightAxis,
        finite: components.every((component) => Number.isFinite(component)),
    };
}

// Draws one frame: the clock advances by a frame, and the controller's own tick re-pins its spline and resolves the pose that frame draws.
function drawFrame() {
    clock += FRAME_MILLISECONDS;
    camera.tick();
}

let pointerX = POINTER_ORIGIN_X;
let pointerY = POINTER_ORIGIN_Y;

function dispatchMouse(type, buttons) {
    container.dispatchEvent(
        new domWindow.MouseEvent(type, {
            button: 0,
            buttons: buttons,
            clientX: pointerX,
            clientY: pointerY,
            bubbles: true,
            cancelable: true,
        }),
    );
}

const callback = eval(fs.readFileSync(spec.source_path, "utf8"))(spec.graph_id, axis);

// Fires the callback on the panel's seeded camera, which is what Dash's initial render reports, then puts one scripted pointer move through the container per spec drag and draws the frames that follow it. `reports_each_drag` is the cadence the panel reports those moves to the callback at: true makes each move a drag of its own, released at the mouse-up gl3d emits `plotly_relayout` on and reported there, and false makes them the pointer moves of one live drag, which the panel reports nothing of until the button comes up. Draining the task queue after each invocation settles the callback's own `Plotly.relayout` promise, which in the browser settles between two user gestures.
async function run() {
    const records = [];
    callback(null);
    await new Promise((settle) => setTimeout(settle, 0));
    for (let frame = 0; frame < FRAMES_PER_DRAG; frame++) {
        drawFrame();
    }
    records.push(measureCamera());
    if (!spec.reports_each_drag) {
        dispatchMouse("mousedown", 1);
    }
    for (const drag of spec.drags) {
        if (spec.reports_each_drag) {
            dispatchMouse("mousedown", 1);
        }
        pointerX += drag.dx;
        pointerY += drag.dy;
        clock += FRAME_MILLISECONDS;
        dispatchMouse("mousemove", 1);
        camera.tick();
        for (let frame = 1; frame < FRAMES_PER_DRAG; frame++) {
            drawFrame();
        }
        if (spec.reports_each_drag) {
            dispatchMouse("mouseup", 0);
            callback(null);
            await new Promise((settle) => setTimeout(settle, 0));
            for (let frame = 0; frame < FRAMES_PER_DRAG; frame++) {
                drawFrame();
            }
        }
        records.push(measureCamera());
    }
    if (!spec.reports_each_drag) {
        dispatchMouse("mouseup", 0);
    }
    process.stdout.write(JSON.stringify(records));
    process.exit(0);
}

run();
