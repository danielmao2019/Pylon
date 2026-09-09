// Roll lock about a caller-supplied axis for a Plotly gl3d panel in `orbit` dragmode.
//
// `orbit` leaves camera roll free: a drag carries `camera.up` wherever the trackball
// takes it, so the camera right axis drifts off the caller's axis and a drag that
// pitches through a pole leaves the scene hanging upside down. This module puts both
// back after every camera change the panel reports, by banding the reported eye away
// from the axis, stopping the camera at the pole the drag carried it through,
// re-deriving `up` there from the view direction and the caller's axis, and writing the
// pose with `Plotly.relayout`.
//
// gl3d emits `plotly_relayout` at mouse-up, so the correction lands once per drag. The
// per-pointer-move `plotly_relayouting` event cannot carry it: a `Plotly.relayout` issued
// inside a live drag re-seeds the orbit camera from the layout every frame, so the drag
// stops accumulating and the camera oscillates in place.
//
// The module is a single expression: a factory the Python registration calls with the
// graph id and the unit-length axis, whose result is the Dash clientside callback.
(function (graphId, axis) {
    // Radians the roll-locked camera stops short of the lock axis. The reported eye is
    // banded into this range before anything derives a camera right axis from it, which
    // is what leaves the view direction never parallel to the axis, so the cross product
    // that re-derives that right axis never collapses.
    const ROLL_LOCK_POLAR_ANGLE_EPSILON = 1e-6;
    // Squared distance between the reported up vector and the roll-locked one at or
    // below which the camera is already roll-locked and no write is issued. Skipping the
    // redundant write is what stops this module's own `Plotly.relayout` from driving an
    // endless relayout -> correct -> relayout cycle.
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

    // A zero-length input is a camera this module cannot describe, so it aborts here
    // rather than dividing and handing the panel a NaN pose it would then report back
    // forever. Every call site below feeds this a vector the banding above already made
    // non-degenerate, so reaching the abort means a camera arrived that the band does
    // not cover, and that is the thing worth seeing.
    function vectorNormalize(vector) {
        const length = Math.sqrt(vectorLengthSquared(vector));
        if (!(length > 0)) {
            throw new Error(
                "cannot normalize a zero-length vector: vector=" + JSON.stringify(vector) + " length=" + length,
            );
        }
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

    // The meridian the fallbacks below stand on, as a unit vector perpendicular to the
    // lock axis. An eye sitting on the axis names no meridian of its own - it is on
    // every meridian at once - and this is the one it is banded onto. Crossing the axis
    // with the world basis vector it leans on least is what keeps this cross product
    // itself clear of the degeneracy it stands in for.
    const ROLL_LOCK_FALLBACK_MERIDIAN = (function () {
        const magnitudes = [Math.abs(axis[0]), Math.abs(axis[1]), Math.abs(axis[2])];
        if (magnitudes[0] <= magnitudes[1] && magnitudes[0] <= magnitudes[2]) {
            return vectorNormalize(vectorCross(axis, [1, 0, 0]));
        }
        if (magnitudes[1] <= magnitudes[2]) {
            return vectorNormalize(vectorCross(axis, [0, 1, 0]));
        }
        return vectorNormalize(vectorCross(axis, [0, 0, 1]));
    })();

    // Builds the eye offset a radius, a polar angle off the lock axis, and a meridian
    // name together.
    function buildOffset(radius, polarAngle, meridian) {
        return vectorAdd(
            vectorScale(axis, radius * Math.cos(polarAngle)),
            vectorScale(meridian, radius * Math.sin(polarAngle)),
        );
    }

    // Resolves the meridian an offset stands on, as a unit vector perpendicular to the
    // lock axis.
    function resolveMeridian(offset) {
        const meridian = vectorSubtract(offset, vectorScale(axis, vectorDot(offset, axis)));
        if (vectorLengthSquared(meridian) === 0) {
            return ROLL_LOCK_FALLBACK_MERIDIAN;
        }
        return vectorNormalize(meridian);
    }

    // Bands an offset's polar angle off the lock axis into the range the roll lock holds
    // the camera in, rebuilding it at the banded angle on its own meridian. Every camera
    // right axis this module derives comes from an offset this has already banded, so
    // the degeneracy that derivation would hit on an eye sitting exactly on the axis is
    // unreachable rather than guarded against afterwards.
    function resolveBandedOffset(offset) {
        const radius = Math.sqrt(vectorLengthSquared(offset));
        const polarAngle = Math.acos(Math.min(Math.max(vectorDot(offset, axis) / radius, -1), 1));
        if (
            polarAngle >= ROLL_LOCK_POLAR_ANGLE_EPSILON
            && polarAngle <= Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON
        ) {
            return offset;
        }
        return buildOffset(
            radius,
            Math.min(
                Math.max(polarAngle, ROLL_LOCK_POLAR_ANGLE_EPSILON),
                Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON,
            ),
            resolveMeridian(offset),
        );
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

    // Resolves the eye the roll lock holds the camera at: the banded eye where the drag
    // stayed on the axis's own side, and the pole the drag entered from where it did
    // not. The reported up hanging on the far side of the axis is what says the drag
    // carried the view through a pole, since a locked camera's up sits on the axis's own
    // side by construction; stopping the camera there is the pitch clamp the panel's own
    // `orbit` rotation does not apply.
    function resolveRollLockedEye(camera) {
        const center = recordToVector(camera.center);
        const offset = resolveBandedOffset(vectorSubtract(recordToVector(camera.eye), center));
        if (vectorDot(recordToVector(camera.up), axis) >= 0) {
            return vectorAdd(center, offset);
        }
        // Past the pole the re-derived right axis points the opposite way, so negating
        // it recovers the meridian the drag entered the pole on, which is the one the
        // camera must be put back onto.
        const entryRight = vectorNegate(vectorNormalize(vectorCross(vectorNegate(offset), axis)));
        const entryMeridian = vectorNormalize(vectorCross(entryRight, axis));
        const polarAngle = vectorDot(offset, axis) > 0
            ? ROLL_LOCK_POLAR_ANGLE_EPSILON
            : Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON;
        const radius = Math.sqrt(vectorLengthSquared(offset));
        return vectorAdd(center, buildOffset(radius, polarAngle, entryMeridian));
    }

    // Resolves the pose the roll lock holds the camera at: the eye above, and the up
    // vector the view direction from that eye and the caller's axis determine.
    function resolveRollLockedPose(camera) {
        const eye = resolveRollLockedEye(camera);
        const forward = vectorNormalize(vectorSubtract(recordToVector(camera.center), eye));
        const right = vectorNormalize(vectorCross(forward, axis));
        return { eye: eye, up: vectorNormalize(vectorCross(right, forward)) };
    }

    // Reports whether the panel's live camera already carries both halves of the lock:
    // its right axis perpendicular to the caller's axis, and its up vector on that
    // axis's own side. Both halves are exactly what the roll-locked up vector is built
    // from, so the reported up matching it is the whole of the question - and it is
    // asked of the up vector alone because the reported eye's own right axis is what
    // collapses on the axis, which is the degeneracy the banding removes.
    function isRollLockHeld(camera, pose) {
        return (
            vectorLengthSquared(vectorSubtract(recordToVector(camera.up), pose.up))
            <= ROLL_LOCK_VIOLATION_EPSILON
        );
    }

    // Writes the roll-locked pose back to the panel when the live camera violates
    // either half of the lock.
    function applyRollLock(graphDiv, camera) {
        ensureRollLockState(graphDiv);
        const pose = resolveRollLockedPose(camera);
        if (isRollLockHeld(camera, pose)) {
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
