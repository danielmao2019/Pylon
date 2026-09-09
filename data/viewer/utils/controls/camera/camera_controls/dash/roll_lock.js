// Roll lock about a caller-supplied axis for a Plotly gl3d panel in `orbit` dragmode.
//
// `orbit` leaves camera roll free: a drag carries `camera.up` wherever the trackball
// takes it, so the camera right axis drifts off the caller's axis and a drag that
// pitches through a pole hangs the scene upside down. This module puts both back after
// every camera change the panel reports, by clamping the reported camera onto the arc
// that never crosses the caller's axis, re-deriving `up` there, and writing the pose
// with `Plotly.relayout`.
//
// gl3d emits `plotly_relayout` at mouse-up, so the correction lands once per drag. The
// per-pointer-move `plotly_relayouting` event cannot carry it: a `Plotly.relayout` issued
// inside a live drag re-seeds the orbit camera from the layout every frame, so the drag
// stops accumulating and the camera oscillates in place.
//
// The module is a single expression: a factory the Python registration calls with the
// graph id and the unit-length axis, whose result is the Dash clientside callback.
(function (graphId, axis) {
    // Radians the roll-locked camera stops short of the lock axis. Clamping the camera
    // onto this band is what stops it at the pole instead of letting the drag carry the
    // view through, so the view direction never runs parallel to the axis and the cross
    // product that re-derives the camera right axis never collapses.
    const ROLL_LOCK_POLAR_ANGLE_EPSILON = 1e-6;
    // Magnitude of `right . axis` at or below which the camera is already roll-locked
    // and no write is issued. Skipping the redundant write is what stops this module's
    // own `Plotly.relayout` from driving an endless relayout -> correct -> relayout cycle.
    const ROLL_LOCK_VIOLATION_EPSILON = 1e-12;

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

    function vectorToRecord(vector) {
        return { x: vector[0], y: vector[1], z: vector[2] };
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

    // Seeds the per-graph in-flight write flag on first use.
    function ensureRollLockState(graphDiv) {
        if (graphDiv.__rollLock !== undefined) {
            return;
        }
        graphDiv.__rollLock = { writing: false };
    }

    // Resolves the roll-locked pose the panel must carry for the camera it reports: the
    // eye clamped onto the arc that never crosses the lock axis, and the up vector
    // re-derived from that eye, reporting whether the clamp moved the eye at all.
    function resolveRollLockedPose(camera) {
        const center = recordToVector(camera.center);
        const offset = vectorSubtract(recordToVector(camera.eye), center);
        const right = vectorNormalize(vectorCross(vectorNegate(offset), axis));
        if (vectorDot(recordToVector(camera.up), axis) >= 0) {
            const forward = vectorNormalize(vectorNegate(offset));
            return {
                clamped: false,
                eye: vectorAdd(center, offset),
                up: vectorNormalize(vectorCross(right, forward)),
            };
        }
        // The reported up hangs on the far side of the lock axis, which the drag can
        // only have done by carrying the view through a pole. Stop the camera at that
        // pole, on the azimuth it entered from - the one the crossing flipped away.
        const entryRight = vectorNegate(right);
        const entryPerpendicular = vectorCross(entryRight, axis);
        const polarAngle = vectorDot(offset, axis) > 0
            ? ROLL_LOCK_POLAR_ANGLE_EPSILON
            : Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON;
        const radius = Math.sqrt(vectorLengthSquared(offset));
        const clampedOffset = vectorAdd(
            vectorScale(axis, radius * Math.cos(polarAngle)),
            vectorScale(entryPerpendicular, radius * Math.sin(polarAngle)),
        );
        return {
            clamped: true,
            eye: vectorAdd(center, clampedOffset),
            up: vectorNormalize(
                vectorCross(entryRight, vectorNormalize(vectorNegate(clampedOffset))),
            ),
        };
    }

    // Reports whether the panel's live camera already holds the lock, so the redundant
    // write is skipped.
    function isRollLockHeld(camera) {
        const forward = vectorNormalize(vectorSubtract(recordToVector(camera.center), recordToVector(camera.eye)));
        const liveRight = vectorNormalize(vectorCross(forward, recordToVector(camera.up)));
        return Math.abs(vectorDot(liveRight, axis)) <= ROLL_LOCK_VIOLATION_EPSILON;
    }

    // Writes the roll-locked pose back to the panel when the live camera violates the
    // lock.
    function applyRollLock(graphDiv, camera) {
        ensureRollLockState(graphDiv);
        const pose = resolveRollLockedPose(camera);
        if (!pose.clamped && isRollLockHeld(camera)) {
            return;
        }
        if (graphDiv.__rollLock.writing) {
            return;
        }
        graphDiv.__rollLock.writing = true;
        Plotly.relayout(graphDiv, {
            "scene.camera.eye": vectorToRecord(pose.eye),
            "scene.camera.up": vectorToRecord(pose.up),
        }).then(function () {
            graphDiv.__rollLock.writing = false;
        });
    }

    // Holds the camera right axis perpendicular to the caller's axis, and the camera up
    // vector on that axis's own side, after every camera change the panel reports.
    return function (relayoutData) {
        const mounted = resolveMountedScene();
        if (mounted !== null) {
            applyRollLock(mounted.graphDiv, mounted.camera);
        }
        return window.dash_clientside.no_update;
    };
})
