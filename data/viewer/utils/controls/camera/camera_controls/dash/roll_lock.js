// Roll lock about a caller-supplied axis for a Plotly gl3d panel in `orbit` dragmode.
//
// `orbit` leaves camera roll free. The panel's own view controller turns the camera about the camera's own screen axes, and successive yaw and pitch turns about a frame that each turn moves compose into roll, so the camera right axis drifts off the caller's axis; the same controller applies no pitch limit, so a drag that carries the view through a pole leaves the scene hanging upside down. This module holds both back: it keeps the camera right axis perpendicular to the caller's axis, and the camera up vector on that axis's own side.
//
// Two things move the camera off that lock, and the module meets each where it happens. A pose the panel is handed - the framing it comes up on, and any camera a later `Plotly.relayout` writes - arrives whole, and is corrected whole when the panel reports it. A drag arrives one pointer move at a time through the view controller's own rotation, and is corrected there: the controller writes each move as a keyframe into a time-indexed spline the renderer samples a frame or two behind, so a pose written back once the move is already in that spline is re-pinned by the controller's own idle before it is ever drawn. Wrapping the controller's `rotate` puts the roll-locked pose into the spline at the same keyframe timestamp the move was written at, which is what holds the horizon level in every rendered frame of a live drag rather than at the drag's end. Rotation is the whole of what that wrapper has to cover: the controller's pan carries the eye and the center together and its wheel zoom moves the eye along the view direction, so both leave the camera frame - and the lock - exactly as they found it.
//
// The module is a single expression: the named factory `createRollLockCallback`, which `_register_dash_roll_lock_callback` calls with the graph id and the unit-length axis, and whose result, the named `rollLockCallback`, is the Dash clientside callback.
(function createRollLockCallback(graphId, axis) {
    // Radians the roll-locked camera stops short of the lock axis. The eye is banded into this range before anything derives a camera right axis from it, which is what leaves the view direction never parallel to the axis, so the cross product that re-derives that right axis never collapses.
    const ROLL_LOCK_POLAR_ANGLE_EPSILON = 1e-6;
    // Squared distance between the reported up vector and the roll-locked one at or below which the camera is already roll-locked and no write is issued. Skipping the redundant write is what stops this module's own `Plotly.relayout` from driving an endless relayout -> correct -> relayout cycle, and what leaves a drag the wrapped rotation already locked reporting a camera this module writes nothing over.
    const ROLL_LOCK_VIOLATION_EPSILON = 1e-12;
    // Radians a drag step turns per unit of the screen-space component the view controller hands its rotation. The controller's own trackball turns by `2 asin(|q|)` for a step of screen length `|q|`, so this keeps the locked panel's drag sensitivity equal to the free panel's.
    const ROLL_LOCK_RADIANS_PER_DRAG_UNIT = 2;

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
    // rather than dividing and handing the panel a NaN pose it would then turn from
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

    // The meridian the fallbacks below stand on, as a unit vector perpendicular to the lock axis. An eye sitting on the axis names no meridian of its own - it is on every meridian at once - and this is the one it is banded onto. Crossing the axis with the world basis vector it leans on least is what keeps this cross product itself clear of the degeneracy it stands in for.
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

    // Builds the eye offset a radius, a polar angle off the lock axis, and a meridian name together.
    function buildOffset(radius, polarAngle, meridian) {
        return vectorAdd(
            vectorScale(axis, radius * Math.cos(polarAngle)),
            vectorScale(meridian, radius * Math.sin(polarAngle)),
        );
    }

    // Resolves the meridian an offset stands on, as a unit vector perpendicular to the lock axis.
    function resolveMeridian(offset) {
        const meridian = vectorSubtract(offset, vectorScale(axis, vectorDot(offset, axis)));
        if (vectorLengthSquared(meridian) === 0) {
            return ROLL_LOCK_FALLBACK_MERIDIAN;
        }
        return vectorNormalize(meridian);
    }

    // Bands an offset's polar angle off the lock axis into the range the roll lock holds the camera in, rebuilding it at the banded angle on its own meridian. Every camera right axis this module derives comes from an offset this has already banded, so the degeneracy that derivation would hit on an eye sitting exactly on the axis is unreachable rather than guarded against afterwards.
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

    // Resolves the eye the roll lock holds the camera at: the banded eye where the turn stayed on the axis's own side, and the pole the turn entered from where it did not. The turned up hanging on the far side of the axis is what says the turn carried the view through a pole, since a locked camera's up sits on the axis's own side by construction; stopping the camera there is the pitch clamp the panel's own `orbit` rotation does not apply.
    function resolveRollLockedEye(camera) {
        const center = recordToVector(camera.center);
        const offset = resolveBandedOffset(vectorSubtract(recordToVector(camera.eye), center));
        if (vectorDot(recordToVector(camera.up), axis) >= 0) {
            return vectorAdd(center, offset);
        }
        // Past the pole the re-derived right axis points the opposite way, so negating it recovers the meridian the turn entered the pole on, which is the one the camera must be put back onto.
        const entryRight = vectorNegate(vectorNormalize(vectorCross(vectorNegate(offset), axis)));
        const entryMeridian = vectorNormalize(vectorCross(entryRight, axis));
        const polarAngle = vectorDot(offset, axis) > 0
            ? ROLL_LOCK_POLAR_ANGLE_EPSILON
            : Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON;
        const radius = Math.sqrt(vectorLengthSquared(offset));
        return vectorAdd(center, buildOffset(radius, polarAngle, entryMeridian));
    }

    // Resolves the pose the roll lock holds the camera at: the eye above, and the up vector the view direction from that eye and the caller's axis determine.
    function resolveRollLockedPose(camera) {
        const eye = resolveRollLockedEye(camera);
        const forward = vectorNormalize(vectorSubtract(recordToVector(camera.center), eye));
        const right = vectorNormalize(vectorCross(forward, axis));
        return { eye: eye, up: vectorNormalize(vectorCross(right, forward)) };
    }

    // Resolves the gl3d panel's scene, or null while it has not mounted yet. `dcc.Graph` renders its component id onto a wrapper div, so the Plotly graph div is the `.js-plotly-plot` inside it. Dash fires the callback on initial render, before the WebGL scene exists, so the unmounted case is owned here rather than handled downstream.
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
        return { graphDiv: graphDiv, scene: scene };
    }

    // Seeds the per-graph in-flight write flag on first use.
    function ensureRollLockState(graphDiv) {
        if (graphDiv.__rollLock !== undefined) {
            return;
        }
        graphDiv.__rollLock = { writing: false };
    }

    // Reports whether the panel's live camera already carries both halves of the lock: its right axis perpendicular to the caller's axis, and its up vector on that axis's own side. Both halves are exactly what the roll-locked up vector is built from, so the reported up matching it is the whole of the question - and it is asked of the up vector alone because the reported eye's own right axis is what collapses on the axis, which is the degeneracy the banding removes.
    function isRollLockHeld(camera, pose) {
        return (
            vectorLengthSquared(vectorSubtract(recordToVector(camera.up), pose.up))
            <= ROLL_LOCK_VIOLATION_EPSILON
        );
    }

    // Writes the roll-locked pose back to the panel when the camera it reports violates either half of the lock.
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

    // Rotates a vector about a unit axis by an angle in radians, right-handed.
    function vectorRotateAboutAxis(vector, unitAxis, angle) {
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        return vectorAdd(
            vectorAdd(vectorScale(vector, cosine), vectorScale(vectorCross(unitAxis, vector), sine)),
            vectorScale(unitAxis, vectorDot(unitAxis, vector) * (1 - cosine)),
        );
    }

    // Turns a roll-locked pose by one drag step. The step's horizontal share yaws the eye about the lock axis and its vertical share pitches it about the camera right axis, the pitch stopping at the polar band, so a horizontal drag holds the elevation and a vertical one holds the azimuth - the motion a roll lock is, rather than the free trackball's motion with its roll taken out afterwards.
    function resolveTurnedPose(eye, center, yaw, pitch) {
        const yawedOffset = vectorRotateAboutAxis(
            resolveBandedOffset(vectorSubtract(eye, center)),
            axis,
            ROLL_LOCK_RADIANS_PER_DRAG_UNIT * yaw,
        );
        const right = vectorNormalize(vectorCross(vectorNegate(yawedOffset), axis));
        const radius = Math.sqrt(vectorLengthSquared(yawedOffset));
        const polarAngle = Math.acos(Math.min(Math.max(vectorDot(yawedOffset, axis) / radius, -1), 1));
        const pitchAngle = Math.min(
            Math.max(
                -ROLL_LOCK_RADIANS_PER_DRAG_UNIT * pitch,
                ROLL_LOCK_POLAR_ANGLE_EPSILON - polarAngle,
            ),
            Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON - polarAngle,
        );
        const offset = vectorRotateAboutAxis(yawedOffset, right, pitchAngle);
        return {
            eye: vectorAdd(center, offset),
            up: vectorNormalize(vectorCross(right, vectorNormalize(vectorNegate(offset)))),
        };
    }

    // Replaces the view controller's rotation with the roll-locked one. The controller's own rotation is never run: it turns the eye about the screen axes of a trackball, and on a pure-horizontal drag that alone moves the eye's polar angle to the lock axis, so keeping its eye and correcting only the up vector would fly the free trackball's path with a level horizon. The drag's screen components instead turn the pose the controller holds at that timestamp through `resolveTurnedPose`, and the turned pose is written back at the same timestamp, so the keyframe the renderer interpolates towards is the locked one.
    function holdRollLock(view) {
        if (view.rollLockHeld === true) {
            return;
        }
        view.rollLockHeld = true;
        view.rotate = function (time, yaw, pitch, roll) {
            view.recalcMatrix(time);
            const center = view.computedCenter.slice();
            const pose = resolveTurnedPose(view.computedEye.slice(), center, yaw, pitch);
            view.lookAt(time, pose.eye, center, pose.up);
        };
    }

    // Re-holds the lock on graphId's gl3d scene each time the graph reports a relayout, its first render included, waiting out the frames before the WebGL scene mounts: the wrapper on the view controller a drag turns the camera through, and the correction of the camera the panel currently reports. A panel that re-renders arrives with a view controller of its own, so the wrapper goes onto whichever one the panel is turning now rather than once and for all.
    function rollLockCallback(relayoutData) {
        const mounted = resolveMountedScene();
        if (mounted === null) {
            window.requestAnimationFrame(function () {
                rollLockCallback(relayoutData);
            });
            return window.dash_clientside.no_update;
        }
        holdRollLock(mounted.scene.camera.view);
        applyRollLock(mounted.graphDiv, mounted.scene.getCamera());
        return window.dash_clientside.no_update;
    }

    return rollLockCallback;
})
