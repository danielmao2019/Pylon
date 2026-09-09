// Roll lock about a caller-supplied axis for a Plotly gl3d panel in `orbit` dragmode.
//
// `orbit` leaves camera roll free: a drag carries `camera.up` wherever the trackball
// takes it, so the camera right axis drifts off the caller's axis. This module puts it
// back after every camera change the panel reports, by re-deriving `up` from the new
// view direction and the caller's axis and writing it with `Plotly.relayout`.
//
// gl3d emits `plotly_relayout` at mouse-up, so the correction lands once per drag. The
// per-pointer-move `plotly_relayouting` event cannot carry it: a `Plotly.relayout` issued
// inside a live drag re-seeds the orbit camera from the layout every frame, so the drag
// stops accumulating and the camera oscillates in place.
//
// The module is a single expression: a factory the Python registration calls with the
// graph id and the unit-length axis, whose result is the Dash clientside callback.
(function (graphId, axis) {
    // Squared length below which a cross product no longer defines a direction: the
    // view direction runs parallel to the roll-lock axis and their cross product
    // collapses, so the camera right axis is carried instead of re-derived.
    const ROLL_LOCK_DEGENERACY_EPSILON_SQUARED = 1e-12;
    // Magnitude of `right . axis` at or below which the camera is already roll-locked
    // and no write is issued. Skipping the redundant write is what stops this module's
    // own `Plotly.relayout` from driving an endless relayout -> correct -> relayout cycle.
    const ROLL_LOCK_VIOLATION_EPSILON = 1e-12;

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

    function vectorLengthSquared(vector) {
        return vectorDot(vector, vector);
    }

    function vectorNormalize(vector) {
        const length = Math.sqrt(vectorLengthSquared(vector));
        console.assert(length > 0, "cannot normalize a zero-length vector", vector);
        return [vector[0] / length, vector[1] / length, vector[2] / length];
    }

    function vectorNegate(vector) {
        return [-vector[0], -vector[1], -vector[2]];
    }

    function recordToVector(record) {
        return [record.x, record.y, record.z];
    }

    // Resolves the gl3d graph div and its live camera, or null while the scene has not
    // mounted yet. `dcc.Graph` renders its component id onto a wrapper div, so the
    // Plotly graph div is the `.js-plotly-plot` inside it. Dash fires the callback on
    // initial render, before the WebGL scene exists, so the unmounted case is owned
    // here rather than handled downstream.
    function resolveMountedScene() {
        const wrapper = document.getElementById(graphId);
        if (wrapper === null) {
            return null;
        }
        const graphDiv = wrapper.querySelector(".js-plotly-plot");
        if (graphDiv === null || graphDiv._fullLayout === undefined || graphDiv._fullLayout.scene === undefined) {
            return null;
        }
        const scene = graphDiv._fullLayout.scene._scene;
        if (scene === undefined || scene === null) {
            return null;
        }
        return { graphDiv: graphDiv, camera: scene.getCamera() };
    }

    // Seeds the per-graph roll-lock state on first use: the carried camera right axis
    // and the in-flight write flag.
    function ensureRollLockState(graphDiv, forward) {
        if (graphDiv.__rollLock !== undefined) {
            return;
        }
        let seedRight = vectorCross(forward, axis);
        if (vectorLengthSquared(seedRight) < ROLL_LOCK_DEGENERACY_EPSILON_SQUARED) {
            // The camera already looks along the lock axis, so every axis perpendicular
            // to the lock axis is an equally valid camera right axis to start from.
            seedRight = vectorCross([1, 0, 0], axis);
            if (vectorLengthSquared(seedRight) < ROLL_LOCK_DEGENERACY_EPSILON_SQUARED) {
                seedRight = vectorCross([0, 1, 0], axis);
            }
        }
        graphDiv.__rollLock = { right: vectorNormalize(seedRight), writing: false };
    }

    // Re-derives the camera right axis from the current view direction and the lock
    // axis, carrying the previous axis through the pole where the cross product
    // collapses or flips sign.
    function resolveCameraRightAxis(graphDiv, forward) {
        const carriedRight = graphDiv.__rollLock.right;
        const rederivedRight = vectorCross(forward, axis);
        if (vectorLengthSquared(rederivedRight) < ROLL_LOCK_DEGENERACY_EPSILON_SQUARED) {
            return carriedRight;
        }
        const normalizedRight = vectorNormalize(rederivedRight);
        // Past a pole the re-derived axis points the opposite way, which would flip the
        // view upside down and bounce the camera off the pole; taking the carried
        // axis's orientation pitches straight through and out the far side.
        if (vectorDot(normalizedRight, carriedRight) < 0) {
            return vectorNegate(normalizedRight);
        }
        return normalizedRight;
    }

    // Writes the roll-locked up axis back to the panel when the live camera violates
    // the lock.
    function applyRollLock(graphDiv, camera) {
        const forward = vectorNormalize(vectorSubtract(recordToVector(camera.center), recordToVector(camera.eye)));
        ensureRollLockState(graphDiv, forward);
        const right = resolveCameraRightAxis(graphDiv, forward);
        graphDiv.__rollLock.right = right;

        const liveRight = vectorNormalize(vectorCross(forward, recordToVector(camera.up)));
        const isPerpendicular = Math.abs(vectorDot(liveRight, axis)) <= ROLL_LOCK_VIOLATION_EPSILON;
        const isSameOrientation = vectorDot(liveRight, right) > 0;
        if (isPerpendicular && isSameOrientation) {
            return;
        }
        if (graphDiv.__rollLock.writing) {
            return;
        }
        const up = vectorNormalize(vectorCross(right, forward));
        graphDiv.__rollLock.writing = true;
        Plotly.relayout(graphDiv, { "scene.camera.up": { x: up[0], y: up[1], z: up[2] } }).then(function () {
            graphDiv.__rollLock.writing = false;
        });
    }

    // Holds the camera right axis perpendicular to the caller's axis after every camera
    // change the panel reports.
    return function (relayoutData) {
        const mounted = resolveMountedScene();
        if (mounted !== null) {
            applyRollLock(mounted.graphDiv, mounted.camera);
        }
        return window.dash_clientside.no_update;
    };
})
