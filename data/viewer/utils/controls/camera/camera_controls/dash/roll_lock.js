// Roll lock about a caller-supplied axis for a Plotly gl3d panel in `orbit` dragmode.
//
// `orbit` leaves camera roll free. The panel's own view controller turns the camera about the camera's own screen axes, and successive yaw and pitch turns about a frame that each turn moves compose into roll, so the camera right axis drifts off the caller's axis; the same controller applies no pitch limit, so a drag that carries the view through a pole leaves the scene hanging upside down. This module holds both back: it keeps the camera right axis perpendicular to the caller's axis as the scene draws it, and the camera up vector on that axis's own side. The caller's axis is a direction in the data's own world frame, while the view controller turns the camera in the scene's normalized space, where Plotly draws each world axis scaled by its aspect ratio over its range - one scale on all three axes under aspectmode "data", a different one per axis under "cube" or "manual" - so the lock holds the camera about the world axis scaled into that space, re-resolved from the scene's layout each time the lock is held on a scene.
//
// Every pose reaches the renderer as a keyframe one of the view's camera controllers - orbital, turntable, matrix - writes through its own `lookAt` into a time-indexed spline the renderer samples a frame or two behind. The view's `lookAt` hands a pose to every controller, and the rotation-mode setter the modebar's "Turntable rotation" button runs writes straight into the newly active controller, so the module wraps each controller's `lookAt` rather than the view's: a drag step, a `Plotly.relayout`, a reset-camera button, a replot and a rotation-mode switch all go into the keyframes already roll-locked, each orbital quaternion in the hemisphere of the keyframe before it, and the renderer never holds an unlocked keyframe to draw. A drag arrives one pointer move at a time through the view's rotation, which the module replaces with the roll-locked turn, written as sub-step keyframes a bounded turn apart so that the frames the renderer interpolates between them stay on the lock too. The locked pose keeps the eye a camera was written with, banded off the poles, and re-derives the up vector from its view direction and the axis whatever up was written, so a camera written upside down is turned upright about its own view direction rather than moved; a camera written with its eye on its center names no view direction at all, so the lock takes the eye offset the pose that write goes into holds at that write's time, about the written center. The layout keeps its own record of the camera, which a relayout, a reset button, a switch to turntable or a Dash figure update writes as it was handed, so the graph's `plotly_relayout` and `plotly_afterplot` events rewrite that record to the roll-locked pose the renderer already draws. A projection switch, on its own or inside a figure update, also builds the scene a new view controller from that record, which the scene's render loop draws before the replot reports itself, so the lock goes onto each view controller where the scene builds it, before it draws a frame, and each replot re-holds the lock on whichever view controller the scene has. A figure update that drops the 3D trace and adds it back builds a new scene altogether, and a mutation observer on the graph div holds the lock on that scene before its first animation frame. Rotation is the whole of what the replaced turn has to cover: the controller's pan carries the eye and the center together and its wheel zoom moves the eye along the view direction, so both leave the camera frame - and the lock - exactly as they found it.
//
// The module is the named function `holdRollLockedGraphs`, which Dash's inline clientside template assigns as the one clientside callback `trackball_camera_controls.py` registers, on the relayoutData of every graph whose pattern-matching component id has type `dash-roll-locked-graph`. Each time any of those graphs reports a relayout - its first render, and a graph a Dash callback adds after the page loaded, included - it walks every matched input in `callback_context.inputs_list[0]`: it resolves the DOM id dash-renderer renders that graph's dict id onto, decodes the unit-length world-frame axis the id's `lock_roll` carries as base64-encoded JSON, and runs the per-graph lock `createRollLockCallback` builds from the two on that graph's relayoutData. The per-graph lock keeps its state on the graph div, its scene and its view controllers, so building it afresh on every run re-holds the lock rather than installing it twice.
function holdRollLockedGraphs(relayoutDataList) {
    // Resolves the DOM id Dash renders a pattern-matching component id onto, the way dash-renderer stringifies one: its keys sorted, each written as its JSON key and its JSON value around a colon, comma-joined inside braces. JSON.stringify writes an object's properties in the order of the key list it is handed, with no whitespace, which is that string for an id whose values are all strings.
    function resolveGraphElementId(graphId) {
        const graphElementId = JSON.stringify(graphId, Object.keys(graphId).sort());
        return graphElementId;
    }

    // Builds the lock one graph holds: the callback that holds its gl3d scene's camera roll about worldAxis, as the scene draws it, through every drag. graphElementId is the DOM id dcc.Graph renders the graph's component id onto; worldAxis is the unit-length lock axis in the data's own world frame, as an [x, y, z] array.
    function createRollLockCallback(graphElementId, worldAxis) {
        // Radians the roll-locked camera stops short of the lock axis. The eye is banded into this range before anything derives a camera right axis from it, which is what leaves the view direction never parallel to the axis, so the cross product that re-derives that right axis never collapses.
        const ROLL_LOCK_POLAR_ANGLE_EPSILON = 1e-6;
        // Squared distance between the reported up vector and the roll-locked one at or below which the camera is already roll-locked and no write is issued. Skipping the redundant write is what stops this module's own `Plotly.relayout` from driving an endless relayout -> correct -> relayout cycle.
        const ROLL_LOCK_VIOLATION_EPSILON = 1e-12;
        // Radians a drag step turns per unit of the screen-space component the view controller hands its rotation. The controller's own trackball turns by `2 asin(|q|)` for a step of screen length `|q|`, so this keeps the locked panel's drag sensitivity equal to the free panel's.
        const ROLL_LOCK_RADIANS_PER_DRAG_UNIT = 2;
        // Largest turn, in radians, one roll-locked drag keyframe takes from the keyframe before it. The renderer interpolates two keyframes component by component, and a frame drawn between two roll-locked poses a turn `theta` apart rolls off the lock by about `theta^2 / 8`, so a fast pointer move written as one keyframe draws frames a degree off the lock; at this bound the in-between frames stay within about 3e-4 of it.
        const ROLL_LOCK_SUB_STEP_RADIANS = 0.05;
        // The lock axis in the mounted scene's normalized space, as a unit [x, y, z] array, and the meridian an eye sitting on it is banded onto, as a unit vector perpendicular to it: such an eye stands on every meridian at once. resolveSceneAxis sets both each time the lock is held on a scene.
        let axis, ROLL_LOCK_FALLBACK_MERIDIAN;

        // Re-holds the lock on graphElementId's gl3d scene each time the roll-lock callback runs, its first render included, waiting out the frames before the WebGL scene mounts. A panel that re-renders arrives with a view controller of its own, so the lock goes onto whichever one the panel is turning now rather than once and for all.
        function rollLockCallback(relayoutData) {
            const mounted = resolveMountedScene();
            if (mounted === null) {
                window.requestAnimationFrame(rollLockCallback.bind(null, relayoutData));
                return window.dash_clientside.no_update;
            }
            holdSceneRollLock(mounted.scene);
            subscribeRollLock(mounted.graphDiv);
            applyRollLock(mounted.graphDiv, mounted.scene.getCamera());
            return window.dash_clientside.no_update;
        }

        // Resolves graphElementId's Plotly graph div and its gl3d scene, or null while the scene has not mounted. `dcc.Graph` renders its component id onto a wrapper div, so the Plotly graph div is the `.js-plotly-plot` inside it; Dash fires the callback on initial render, before the WebGL scene exists.
        function resolveMountedScene() {
            const graphDiv = document.getElementById(graphElementId)?.querySelector(".js-plotly-plot") ?? null;
            if (
                graphDiv === null
                || graphDiv._fullLayout === undefined
                || graphDiv._fullLayout.scene === undefined
                || graphDiv._fullLayout.scene._scene === undefined
                || graphDiv._fullLayout.scene._scene === null
            ) {
                return null;
            }
            const mountedScene = { graphDiv: graphDiv, scene: graphDiv._fullLayout.scene._scene };
            return mountedScene;
        }

        // Holds the lock on one gl3d scene: on the view controller it turns now, and on each one it builds later from the camera the layout stores, before that controller draws a frame. A projection switch - on its own or inside a Dash figure update - disposes the scene's plot and builds it again, and the rebuilt plot's render loop draws the new view controller from its next animation frame, well before the replot reports itself through `plotly_afterplot`; so the lock goes onto that controller where the scene builds it, in `initializeGLCamera`, which gl3d's `initializeGLPlot` calls on the scene itself, so the scene's own property takes precedence over the method its prototype carries.
        function holdSceneRollLock(scene) {
            resolveSceneAxis(scene);
            holdRollLock(scene.camera.view);
            if (scene.rollLockHeld === true) {
                return;
            }
            scene.rollLockHeld = true;
            const sceneInitializeGLCamera = scene.initializeGLCamera;

            // Builds the scene's camera through its own initializeGLCamera, as a projection switch does, then holds the lock on the new view controller in the same call.
            function rollLockedInitializeGLCamera() {
                sceneInitializeGLCamera.call(scene);
                holdRollLock(scene.camera.view);
                return;
            }

            scene.initializeGLCamera = rollLockedInitializeGLCamera;
            return;
        }

        // Resolves the lock axis in the scene's normalized space, where Plotly draws each world axis scaled by its aspect ratio over its range, so worldAxis stays upright on screen under any aspect: the camera turns in that space, and the scale is the same on all three axes under aspectmode "data" but not under "cube" or "manual". Crossing the axis with the world basis vector it leans on least keeps the fallback meridian's cross product itself clear of the degeneracy it stands in for.
        function resolveSceneAxis(scene) {
            const sceneScale = [
                scene.fullSceneLayout.aspectratio.x / (scene.fullSceneLayout.xaxis.range[1] - scene.fullSceneLayout.xaxis.range[0]),
                scene.fullSceneLayout.aspectratio.y / (scene.fullSceneLayout.yaxis.range[1] - scene.fullSceneLayout.yaxis.range[0]),
                scene.fullSceneLayout.aspectratio.z / (scene.fullSceneLayout.zaxis.range[1] - scene.fullSceneLayout.zaxis.range[0]),
            ];
            axis = vectorNormalize([
                worldAxis[0] * sceneScale[0],
                worldAxis[1] * sceneScale[1],
                worldAxis[2] * sceneScale[2],
            ]);
            let leastBasis;
            if (Math.abs(axis[0]) <= Math.abs(axis[1]) && Math.abs(axis[0]) <= Math.abs(axis[2])) {
                leastBasis = [1, 0, 0];
            } else if (Math.abs(axis[1]) <= Math.abs(axis[2])) {
                leastBasis = [0, 1, 0];
            } else {
                leastBasis = [0, 0, 1];
            }
            ROLL_LOCK_FALLBACK_MERIDIAN = vectorNormalize(vectorCross(axis, leastBasis));
            return;
        }

        // Holds the lock on one gl3d scene's view controller, once per view controller, since a replotted graph arrives with a view controller of its own: wraps every camera controller's lookAt, and replaces the view's rotation with the roll-locked turn. The view's own rotation is never run: it turns the eye about the screen axes of a trackball, and on a pure-horizontal drag that alone moves the eye's polar angle to the lock axis, so keeping its eye and correcting only the up vector would fly the free trackball's path with a level horizon.
        function holdRollLock(view) {
            if (view.rollLockHeld === true) {
                return;
            }
            view.rollLockHeld = true;
            for (const controller of view._controllerList) {
                holdControllerRollLock(controller);
            }
            // A view controller a scene built from a rolled stored camera draws that camera until something writes it a new pose; writing its own pose back through the wrapped controllers puts it on the lock from its first frame when the scene has just built it, and from its next one otherwise.
            view.recalcMatrix(view.lastT());
            view.lookAt(view.lastT(), view.computedEye.slice(), view.computedCenter.slice(), view.computedUp.slice());

            // Turns the camera by one drag step, as yaw about axis plus pitch about the camera right axis, written as roll-locked sub-step keyframes at evenly spaced times from the view's newest keyframe to the step's time, each turned from the pose the view holds at its own time. The pure roll a horizontal wheel scroll hands in as `roll` is dropped. Right after a switch to turntable the newest keyframe sits half a second ahead, and sub-steps timed before it are dropped by the controllers the way Plotly drops any write older than its newest keyframe.
            function rollLockedRotate(time, yaw, pitch, roll) {
                const subStepCount = Math.max(
                    1,
                    Math.ceil(ROLL_LOCK_RADIANS_PER_DRAG_UNIT * Math.hypot(yaw, pitch) / ROLL_LOCK_SUB_STEP_RADIANS),
                );
                const startTime = view.lastT();
                for (let subStep = 1; subStep <= subStepCount; subStep += 1) {
                    const subStepTime = startTime + (time - startTime) * subStep / subStepCount;
                    view.recalcMatrix(subStepTime);
                    const [eye, center] = [view.computedEye.slice(), view.computedCenter.slice()];
                    const turnedPose = resolveTurnedPose(eye, center, yaw / subStepCount, pitch / subStepCount);
                    view.lookAt(subStepTime, turnedPose.eye, center, turnedPose.up);
                }
                return;
            }

            view.rotate = rollLockedRotate;
            return;
        }

        // Wraps one camera controller's lookAt so every pose written into its keyframes goes in roll-locked - a drag step, a relayout, a reset-camera button, a replot, and a rotation-mode switch writing into the newly active controller directly.
        function holdControllerRollLock(controller) {
            const controllerLookAt = controller.lookAt;

            // Writes one pose into the controller's keyframes on the lock, taking the same optional eye and center the controller's own lookAt does, filling a missing one from the controller's pose at that time, and an eye written onto its center from the eye offset that pose holds; the written up is the one thing the lock never keeps, since it re-derives the up from the view direction and the axis. Only the orbital controller keeps its rotation as quaternion keyframes; the turntable controller keeps angles and the matrix controller whole matrices, which have no second hemisphere to land in.
            function rollLockedLookAt(time, eye, center, up) {
                controller.recalcMatrix(time);
                [eye, center] = [(eye || controller.computedEye).slice(), (center || controller.computedCenter).slice()];
                const heldEye = resolveHeldEye(eye, center, controller, time);
                const rollLockedPose = resolveRollLockedPose(heldEye, center);
                const keyframeCount = controller.rotation && controller.rotation._time.length;
                controllerLookAt.call(controller, time, rollLockedPose.eye, center, rollLockedPose.up);
                if (controller.rotation !== undefined && controller.rotation._time.length > keyframeCount) {
                    alignRotationKeyframeHemisphere(controller.rotation);
                }
                return;
            }

            controller.lookAt = rollLockedLookAt;
            return;
        }

        // Negates the newest rotation keyframe's quaternion when it sits in the opposite hemisphere from the keyframe before it. `q` and `-q` name one rotation and the controller's lookAt stores whichever its matrix-to-quaternion step lands on, but the renderer interpolates the keyframes component by component, so between `q` and `-q` the frames it draws swing through unrelated orientations. The vector stores each keyframe's four components contiguously, so the newest keyframe is the last four entries of its state and the one before it the four ahead of those.
        function alignRotationKeyframeHemisphere(rotation) {
            if (
                rotation._state[rotation._state.length - 8] * rotation._state[rotation._state.length - 4]
                + rotation._state[rotation._state.length - 7] * rotation._state[rotation._state.length - 3]
                + rotation._state[rotation._state.length - 6] * rotation._state[rotation._state.length - 2]
                + rotation._state[rotation._state.length - 5] * rotation._state[rotation._state.length - 1]
                >= 0
            ) {
                return;
            }
            [
                rotation._state[rotation._state.length - 4],
                rotation._state[rotation._state.length - 3],
                rotation._state[rotation._state.length - 2],
                rotation._state[rotation._state.length - 1],
            ] = [
                -rotation._state[rotation._state.length - 4],
                -rotation._state[rotation._state.length - 3],
                -rotation._state[rotation._state.length - 2],
                -rotation._state[rotation._state.length - 1],
            ];
            return;
        }

        // Subscribes the lock once to graphDiv's plotly_relayout and plotly_afterplot events and to the scenes Plotly mounts inside it, so the camera the layout stores - which a relayout, a reset-camera button, a switch to turntable or a Dash figure update writes as it was handed - is rewritten to the roll-locked pose the renderer already draws. A relayout event hands the lock the camera it wrote rather than the scene's, since the wrapped controllers have already locked the one the scene reports.
        function subscribeRollLock(graphDiv) {
            if (graphDiv.__rollLock !== undefined) {
                return;
            }
            // `writing` is the in-flight flag applyRollLock holds while its own relayout lands; `pending` records whether a camera write arrived while it was in flight.
            graphDiv.__rollLock = { writing: false, pending: false };

            // Rewrites the camera one relayout event wrote into the layout onto the lock.
            function rewriteWrittenCamera(eventData) {
                const writtenCamera = resolveWrittenCamera(graphDiv, eventData);
                if (writtenCamera === null) {
                    return;
                }
                applyRollLock(graphDiv, writtenCamera);
                return;
            }

            graphDiv.on("plotly_relayout", rewriteWrittenCamera);

            // Re-holds the lock after each replot, since a Dash figure update reports its camera to no relayout event, and a projection switch rebuilds the scene's view controller from the camera the layout stores; a scene the graph built anew, rather than rebuilt, is held here too. The event fires after the replot rebuilt the full layout from the layout input, so the camera read there is the one this replot stored.
            function rewriteReplottedCamera() {
                const mounted = resolveMountedScene();
                if (mounted === null) {
                    return;
                }
                holdSceneRollLock(mounted.scene);
                applyRollLock(graphDiv, graphDiv._fullLayout.scene.camera);
                return;
            }

            graphDiv.on("plotly_afterplot", rewriteReplottedCamera);

            // Holds the lock on a gl3d scene Plotly builds anew inside graphDiv, as a figure update that drops the 3D trace and adds it back does, from a mutation observer that runs before that scene's first animation frame.
            function holdRebuiltScene() {
                const mounted = resolveMountedScene();
                if (mounted === null) {
                    return;
                }
                holdSceneRollLock(mounted.scene);
                return;
            }

            // Runs as Plotly.react's synchronous part ends, before the rebuilt scene's first animation frame, where plotly_afterplot runs later.
            new MutationObserver(holdRebuiltScene).observe(graphDiv, { childList: true, subtree: true });
            return;
        }

        // Resolves the camera one relayout event wrote into graphDiv's layout, or null when it wrote none. A drag, pan or zoom ends by saving its camera into the layout input and into a full layout object the graph has since replaced, and reports that camera whole in the event. A `Plotly.relayout` that writes the camera by key path - a reset-camera button, this module's own correction - rebuilds the full layout, and so does a rotation-mode switch, whose switch to turntable also re-seats the stored camera's up on world +Z without reporting it; both are read from the full layout. An event that writes neither - the empty relayout a wheel zoom opens with - leaves the stored camera as it was, and the full layout, which such an event need not rebuild, can still hold one a drag has since replaced.
        function resolveWrittenCamera(graphDiv, eventData) {
            if (eventData["scene.camera"] !== undefined) {
                return eventData["scene.camera"];
            }
            for (const key of Object.keys(eventData)) {
                if (key.indexOf("scene.camera.") === 0 || key === "scene.dragmode") {
                    return graphDiv._fullLayout.scene.camera;
                }
            }
            return null;
        }

        // Writes the roll-locked pose back to the graph when the camera it reports sits off the lock. The reported up matching the roll-locked one is the whole of the question: both halves of the lock - the right axis perpendicular to the caller's axis, the up vector on that axis's own side - are exactly what the roll-locked up is built from. A camera reported with its eye on its center has no view direction to sit on the lock with: the wrapped controllers already hold the scene's view at its last eye offset about that center, so its record is rewritten to that pose whatever up it reports.
        async function applyRollLock(graphDiv, camera) {
            const eye = recordToVector(camera.eye);
            const center = recordToVector(camera.center);
            const up = recordToVector(camera.up);
            const heldEye = resolveHeldEye(
                eye,
                center,
                graphDiv._fullLayout.scene._scene.camera.view,
                graphDiv._fullLayout.scene._scene.camera.view.lastT(),
            );
            const rollLockedPose = resolveRollLockedPose(heldEye, center);
            const upDistance = vectorSubtract(up, rollLockedPose.up);
            if (heldEye === eye && vectorDot(upDistance, upDistance) <= ROLL_LOCK_VIOLATION_EPSILON) {
                return;
            }
            if (graphDiv.__rollLock.writing) {
                // Camera writes fired back to back reach the layout one after another while a correction is already in flight, and each of their events leaves the record to that correction; the record the last of them left is the one read back once the correction lands, rather than left as it was written.
                graphDiv.__rollLock.pending = true;
                return;
            }
            graphDiv.__rollLock.writing = true;
            await Plotly.relayout(graphDiv, {
                "scene.camera.eye": vectorToRecord(rollLockedPose.eye),
                "scene.camera.up": vectorToRecord(rollLockedPose.up),
            });
            graphDiv.__rollLock.writing = false;
            if (graphDiv.__rollLock.pending === true) {
                graphDiv.__rollLock.pending = false;
                applyRollLock(graphDiv, graphDiv._fullLayout.scene.camera);
            }
            return;
        }

        // Resolves the eye the lock holds a written camera from: the eye it was written with, or - when that eye coincides with its center, so the camera carries no view direction to lock - the written center moved by the eye offset heldPose holds at time, heldPose being the camera controller the camera is written into or the scene's view. The lock thereby keeps the view direction and the distance the scene last held and takes the written center alone.
        function resolveHeldEye(eye, center, heldPose, time) {
            const offset = vectorSubtract(eye, center);
            if (vectorDot(offset, offset) !== 0) {
                return eye;
            }
            heldPose.recalcMatrix(time);
            const heldEye = vectorAdd(center, vectorSubtract(heldPose.computedEye, heldPose.computedCenter));
            return heldEye;
        }

        // Turns a roll-locked pose by one drag step's yaw about axis and pitch about the camera right axis, the pitch stopping at the polar band, so a horizontal drag holds the elevation and a vertical one holds the azimuth - the motion a roll lock is, rather than the free trackball's motion with its roll taken out afterwards.
        function resolveTurnedPose(eye, center, yaw, pitch) {
            const bandedOffset = resolveBandedOffset(vectorSubtract(eye, center));
            const yawedOffset = vectorRotateAboutAxis(bandedOffset, axis, ROLL_LOCK_RADIANS_PER_DRAG_UNIT * yaw);
            const right = vectorNormalize(vectorCross(vectorScale(yawedOffset, -1), axis));
            const polarAngle = Math.acos(
                Math.min(Math.max(vectorDot(yawedOffset, axis) / Math.sqrt(vectorDot(yawedOffset, yawedOffset)), -1), 1),
            );
            const pitchAngle = Math.min(
                Math.max(-ROLL_LOCK_RADIANS_PER_DRAG_UNIT * pitch, ROLL_LOCK_POLAR_ANGLE_EPSILON - polarAngle),
                Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON - polarAngle,
            );
            const turnedOffset = vectorRotateAboutAxis(yawedOffset, right, pitchAngle);
            const turnedUp = vectorNormalize(vectorCross(right, vectorScale(turnedOffset, -1)));
            const turnedPose = { eye: vectorAdd(center, turnedOffset), up: turnedUp };
            return turnedPose;
        }

        // Resolves the pose the lock holds a camera at: its eye where it was written, banded off the poles, and the up vector the view direction from that eye and axis determine, whatever up was written. A camera written with its up on the far side of the axis therefore keeps its eye and is turned upright about its own view direction.
        function resolveRollLockedPose(eye, center) {
            const bandedOffset = resolveBandedOffset(vectorSubtract(eye, center));
            const rollLockedEye = vectorAdd(center, bandedOffset);
            const forward = vectorNormalize(vectorSubtract(center, rollLockedEye));
            const right = vectorNormalize(vectorCross(forward, axis));
            const rollLockedPose = { eye: rollLockedEye, up: vectorNormalize(vectorCross(right, forward)) };
            return rollLockedPose;
        }

        // Bands an eye offset's polar angle off axis into [ROLL_LOCK_POLAR_ANGLE_EPSILON, pi - ROLL_LOCK_POLAR_ANGLE_EPSILON], rebuilding it at the banded angle on its own meridian. Every camera right axis this module derives comes from an offset this has already banded, so the degeneracy that derivation would hit on an eye sitting exactly on the axis is unreachable rather than guarded against afterwards.
        function resolveBandedOffset(offset) {
            const radius = Math.sqrt(vectorDot(offset, offset));
            const polarAngle = Math.acos(Math.min(Math.max(vectorDot(offset, axis) / radius, -1), 1));
            if (polarAngle >= ROLL_LOCK_POLAR_ANGLE_EPSILON && polarAngle <= Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON) {
                return offset;
            }
            const meridian = resolveMeridian(offset);
            const bandedPolarAngle = Math.min(
                Math.max(polarAngle, ROLL_LOCK_POLAR_ANGLE_EPSILON),
                Math.PI - ROLL_LOCK_POLAR_ANGLE_EPSILON,
            );
            const bandedOffset = vectorAdd(
                vectorScale(axis, radius * Math.cos(bandedPolarAngle)),
                vectorScale(meridian, radius * Math.sin(bandedPolarAngle)),
            );
            return bandedOffset;
        }

        // Resolves the meridian an eye offset stands on, as a unit vector perpendicular to axis; an offset on axis stands on every meridian at once.
        function resolveMeridian(offset) {
            const meridian = vectorSubtract(offset, vectorScale(axis, vectorDot(offset, axis)));
            if (vectorDot(meridian, meridian) === 0) {
                return ROLL_LOCK_FALLBACK_MERIDIAN;
            }
            const unitMeridian = vectorNormalize(meridian);
            return unitMeridian;
        }

        // Rotates a vector about a unit axis by an angle in radians, right-handed (Rodrigues' rotation), the way a drag's yaw and pitch turn the eye offset.
        function vectorRotateAboutAxis(vector, unitAxis, angle) {
            const [cosine, sine] = [Math.cos(angle), Math.sin(angle)];
            const rotated = vectorAdd(
                vectorAdd(vectorScale(vector, cosine), vectorScale(vectorCross(unitAxis, vector), sine)),
                vectorScale(unitAxis, vectorDot(unitAxis, vector) * (1 - cosine)),
            );
            return rotated;
        }

        // Adds two [x, y, z] arrays.
        function vectorAdd(left, right) {
            const sum = [left[0] + right[0], left[1] + right[1], left[2] + right[2]];
            return sum;
        }

        // Subtracts one [x, y, z] array from another.
        function vectorSubtract(left, right) {
            const difference = [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
            return difference;
        }

        // Scales an [x, y, z] array by a scalar, negation included.
        function vectorScale(vector, scalar) {
            const scaled = [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar];
            return scaled;
        }

        // Resolves the dot product of two [x, y, z] arrays, a squared length when both are one vector.
        function vectorDot(left, right) {
            const dot = left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
            return dot;
        }

        // Resolves the right-handed cross product of two [x, y, z] arrays.
        function vectorCross(left, right) {
            const cross = [
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            ];
            return cross;
        }

        // Scales an [x, y, z] array to unit length. A zero-length input is a camera the polar band does not cover, so it aborts here rather than handing the panel a NaN pose it would then turn from forever.
        function vectorNormalize(vector) {
            const length = Math.sqrt(vectorDot(vector, vector));
            if (!(length > 0)) {
                throw new Error(
                    "cannot normalize a zero-length vector: vector=" + JSON.stringify(vector) + " length=" + length,
                );
            }
            const unit = [vector[0] / length, vector[1] / length, vector[2] / length];
            return unit;
        }

        // Converts a Plotly {x, y, z} camera record to an [x, y, z] array.
        function recordToVector(record) {
            const vector = [record.x, record.y, record.z];
            return vector;
        }

        // Converts an [x, y, z] array to a Plotly {x, y, z} camera record.
        function vectorToRecord(vector) {
            const record = { x: vector[0], y: vector[1], z: vector[2] };
            return record;
        }

        return rollLockCallback;
    }

    // Holds every graph the roll-locked pattern matches, one { id, property, value } input per graph, about the world-frame axis its own component id carries.
    for (const input of window.dash_clientside.callback_context.inputs_list[0]) {
        const graphElementId = resolveGraphElementId(input.id);
        // The unit-length axis create_dash_trackball_camera_controls normalized and base64-encoded, so no id value holds a "." Dash escapes in output ids.
        const worldAxis = JSON.parse(atob(input.id.lock_roll));
        const rollLockCallback = createRollLockCallback(graphElementId, worldAxis);
        rollLockCallback(input.value);
    }
    return window.dash_clientside.no_update;
}
