// Node harness driving the roll-lock clientside callback through a scripted gl3d drag.
//
// The harness stands in for the Plotly panel: it owns the gl3d camera the callback reads and the view controller a drag turns that camera through, turns the camera the way an `orbit` left-drag does, hands the turned camera to the callback, applies whatever `Plotly.relayout` the callback issues, and reports the camera the callback left behind. Nothing about the roll lock itself is modelled here: the callback under test is the shipped source, evaluated exactly as the Dash registration inlines it.
//
// The view controller stand-in holds its pose directly where a real one splines its keyframes over time and publishes a pose a frame or two behind. What is under test is which pose each call leaves behind, not how the controller interpolates between them, so the timestamps the callback writes at are carried and never interpolated.
//
// Usage: node roll_lock_harness.js '<spec-json>', where the spec carries `source_path`, `graph_id`, `lock_roll`, `eye`, `center`, `up`, `turns`, and `reports_each_turn`. One JSON record per turn is written to stdout.

const fs = require("fs");

function vectorAdd(left, right) {
    return [left[0] + right[0], left[1] + right[1], left[2] + right[2]];
}

function vectorSubtract(left, right) {
    return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function vectorScale(vector, scalar) {
    return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar];
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

function vectorRotate(vector, axis, angle) {
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    return vectorAdd(
        vectorAdd(vectorScale(vector, cosine), vectorScale(vectorCross(axis, vector), sine)),
        vectorScale(axis, vectorDot(axis, vector) * (1 - cosine)),
    );
}

function vectorToRecord(vector) {
    return { x: vector[0], y: vector[1], z: vector[2] };
}

function recordToVector(record) {
    return [record.x, record.y, record.z];
}

const spec = JSON.parse(process.argv[2]);
const axis = vectorNormalize(spec.lock_roll);

// The panel's own view controller, as much of it as the roll lock touches: the pose it
// publishes, the rotation a drag turns the camera through, and the two calls the lock
// reads and writes that pose with.
const view = {
    computedEye: spec.eye.slice(),
    computedCenter: spec.center.slice(),
    computedUp: spec.up.slice(),
    recalcMatrix: function (time) {},
    lookAt: function (time, eye, center, up) {
        view.computedEye = eye.slice();
        view.computedCenter = center.slice();
        view.computedUp = up.slice();
    },
    // Turns the camera the way a Plotly gl3d `orbit` left-drag does: the eye and the
    // camera up vector rotate together about the camera's own screen axes, so the whole
    // frame is carried rigidly and roll is left free to drift. This is the turn the roll
    // lock exists to correct, so the harness produces it rather than a roll-locked one.
    rotate: function (time, yawRadians, pitchRadians, rollRadians) {
        const center = view.computedCenter;
        const up = view.computedUp;
        const offset = vectorSubtract(view.computedEye, center);
        const cameraUpAxis = vectorNormalize(up);
        const cameraRightAxis = vectorNormalize(
            vectorCross(vectorNormalize(vectorScale(offset, -1)), cameraUpAxis),
        );
        view.computedEye = vectorAdd(
            center,
            vectorRotate(vectorRotate(offset, cameraUpAxis, yawRadians), cameraRightAxis, pitchRadians),
        );
        view.computedUp = vectorNormalize(
            vectorRotate(vectorRotate(up, cameraUpAxis, yawRadians), cameraRightAxis, pitchRadians),
        );
    },
};

function readCamera() {
    return {
        eye: vectorToRecord(view.computedEye),
        center: vectorToRecord(view.computedCenter),
        up: vectorToRecord(view.computedUp),
    };
}

const graphDiv = {
    _fullLayout: { scene: { _scene: { camera: { view: view }, getCamera: readCamera } } },
};
globalThis.document = {
    getElementById: (elementId) =>
        elementId === spec.graph_id
            ? { querySelector: (selector) => (selector === ".js-plotly-plot" ? graphDiv : null) }
            : null,
};
globalThis.window = { dash_clientside: { no_update: null } };
globalThis.Plotly = {
    relayout: (targetGraphDiv, update) => {
        if (update["scene.camera.eye"] !== undefined) {
            view.computedEye = recordToVector(update["scene.camera.eye"]);
        }
        if (update["scene.camera.up"] !== undefined) {
            view.computedUp = recordToVector(update["scene.camera.up"]);
        }
        return Promise.resolve();
    },
};

// Reads the roll-lock invariants off the camera the callback left behind: the camera
// right axis's component along the lock axis, the up vector's side of it, and the polar
// angle that says how close to the pole the camera stands.
function measureCamera() {
    const center = view.computedCenter;
    const eye = view.computedEye;
    const up = view.computedUp;
    const offset = vectorSubtract(eye, center);
    const forward = vectorNormalize(vectorScale(offset, -1));
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

const callback = eval(fs.readFileSync(spec.source_path, "utf8"))(spec.graph_id, axis);

// Fires the callback on the panel's seeded camera, which is what Dash's initial render reports, and turns the camera once per spec turn. `reports_each_turn` is the cadence the panel reports those turns to the callback at: true makes each turn a drag of its own, reported at the mouse-up gl3d emits `plotly_relayout` on, and false makes the turns the pointer moves of one live drag, which the panel reports nothing of until it ends. Draining the task queue after each invocation settles the callback's own `Plotly.relayout` promise, which in the browser settles between two user gestures.
async function run() {
    const records = [];
    callback(null);
    await new Promise((settle) => setTimeout(settle, 0));
    records.push(measureCamera());
    for (const turn of spec.turns) {
        view.rotate(0, turn.yaw, turn.pitch, 0);
        if (spec.reports_each_turn) {
            callback(null);
            await new Promise((settle) => setTimeout(settle, 0));
        }
        records.push(measureCamera());
    }
    process.stdout.write(JSON.stringify(records));
}

run();
