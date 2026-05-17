(function () {
  const params = new URLSearchParams(window.location.search);
  const label = params.get("label") || "Point cloud";
  const filePath = params.get("file");
  const autoloadUrl = params.get("autoloadUrl");
  const cameraPosesFilePath = params.get("cameraPosesFile");
  const cameraConvention = params.get("cameraConvention");
  const cameraExtrinsicsFormat = params.get("cameraExtrinsicsFormat");
  const isMissing = params.get("missing") === "1";
  const title = document.getElementById("viewer-title");
  const status = document.getElementById("viewer-status");
  const canvas = document.getElementById("pointcloud-canvas");
  const context = canvas.getContext("2d", { alpha: false });
  let pendingDrawFrame = null;
  let pendingCameraStatePublishFrame = null;
  const state = {
    orientation: quaternionFromYawPitch(0.7, -0.45),
    scale: 1,
    dragging: false,
    dragMode: "rotate",
    lastX: 0,
    lastY: 0,
    lastTrackballVector: null,
    offsetX: 0,
    offsetY: 0,
    offsetZ: 0,
    points: null,
    bounds: null,
    cameraVisualizations: null,
    showCameras: params.get("showCameras") !== "0",
  };
  const background = { r: 248, g: 250, b: 252, a: 255 };

  title.textContent = label;

  function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      scheduleDraw();
    }
  }

  function load() {
    if (isMissing) {
      status.textContent = "Placeholder: no materialized artifact for this selection";
      paintPlaceholder();
      return;
    }
    const url = buildPayloadUrl();
    if (!url) {
      status.textContent = "Placeholder: no materialized artifact for this selection";
      paintPlaceholder();
      return;
    }
    fetch(url)
      .then((response) => {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.arrayBuffer();
      })
      .then((buffer) => {
        const parsed = parsePly(buffer);
        state.points = parsed.points;
        state.bounds = parsed.bounds;
        status.textContent = "Loaded " + parsed.count.toLocaleString() + " points";
        resizeCanvas();
        scheduleDraw();
        writeCameraStateForTests();
        loadCameraPoses()
          .then((cameraCount) => {
            if (cameraCount > 0) {
              status.textContent =
                "Loaded " +
                parsed.count.toLocaleString() +
                " points and " +
                cameraCount.toLocaleString() +
                " cameras";
            }
            scheduleDraw();
            writeCameraStateForTests();
          })
          .catch((error) => {
            status.textContent =
              "Loaded " +
              parsed.count.toLocaleString() +
              " points; unable to load cameras (" +
              error.message +
              ")";
          });
      })
      .catch((error) => {
        status.textContent = "Placeholder: unable to load artifact (" + error.message + ")";
        paintPlaceholder();
      });
  }

  function buildPayloadUrl() {
    if (filePath) {
      return "/shared-files?" + new URLSearchParams({ path: filePath }).toString();
    }
    if (autoloadUrl) {
      return autoloadUrl;
    }
    return "";
  }

  function loadCameraPoses() {
    if (!cameraPosesFilePath) {
      state.cameraVisualizations = [];
      return Promise.resolve(0);
    }
    assertSupportedCameraPosesConvention();
    const url =
      "/shared-files?" +
      new URLSearchParams({ path: cameraPosesFilePath }).toString();
    return fetch(url)
      .then((response) => {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.text();
      })
      .then((text) => {
        state.cameraVisualizations = parseOpenCvC2wCameraPoses(text);
        return state.cameraVisualizations.length;
      });
  }

  function assertSupportedCameraPosesConvention() {
    if (cameraConvention !== "opencv" || cameraExtrinsicsFormat !== "c2w") {
      throw new Error(
        "unsupported camera pose convention " +
          cameraConvention +
          " / " +
          cameraExtrinsicsFormat
      );
    }
  }

  function parsePly(buffer) {
    const headerBytes = buffer.slice(0, Math.min(buffer.byteLength, 1048576));
    const headerText = new TextDecoder("utf-8").decode(headerBytes);
    const endIndex = headerText.indexOf("end_header");
    if (endIndex < 0) {
      throw new Error("PLY header is missing end_header");
    }
    const headerPrefix = headerText.slice(0, endIndex);
    const rawHeader = headerText.slice(0, endIndex + "end_header".length);
    let dataOffset = new TextEncoder().encode(rawHeader).length;
    const bytes = new Uint8Array(buffer);
    while (dataOffset < bytes.length && (bytes[dataOffset] === 10 || bytes[dataOffset] === 13)) {
      dataOffset += 1;
    }
    const header = readHeader(headerPrefix);
    if (header.format === "ascii") {
      return parseAscii(buffer, dataOffset, header);
    }
    if (header.format === "binary_little_endian") {
      return parseBinaryLittleEndian(buffer, dataOffset, header);
    }
    throw new Error("unsupported PLY format " + header.format);
  }

  function readHeader(headerText) {
    const lines = headerText.split(/\r?\n/);
    let format = "";
    let vertexCount = 0;
    let inVertex = false;
    const properties = [];
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length === 0 || parts[0] === "") {
        continue;
      }
      if (parts[0] === "format") {
        format = parts[1];
      } else if (parts[0] === "element") {
        inVertex = parts[1] === "vertex";
        if (inVertex) {
          vertexCount = Number(parts[2]);
        }
      } else if (parts[0] === "property" && inVertex) {
        if (parts[1] === "list") {
          throw new Error("vertex list properties are not supported");
        }
        properties.push({ type: parts[1], name: parts[2] });
      }
    }
    if (!format) {
      throw new Error("PLY format is missing");
    }
    if (!Number.isFinite(vertexCount) || vertexCount < 1) {
      throw new Error("PLY vertex count is invalid");
    }
    return { format: format, vertexCount: vertexCount, properties: properties };
  }

  function parseAscii(buffer, dataOffset, header) {
    const dataText = new TextDecoder("utf-8").decode(buffer.slice(dataOffset));
    const lines = dataText.trim().split(/\r?\n/);
    const indices = propertyIndices(header.properties);
    const points = allocatePoints(header.vertexCount);
    const bounds = freshBounds();
    for (let index = 0; index < header.vertexCount; index += 1) {
      const parts = lines[index].trim().split(/\s+/);
      const x = Number(parts[indices.x]);
      const y = Number(parts[indices.y]);
      const z = Number(parts[indices.z]);
      const color = readAsciiColor(parts, indices);
      writePoint(points, bounds, index, x, y, z, color);
    }
    return { points: points, bounds: bounds, count: header.vertexCount };
  }

  function parseBinaryLittleEndian(buffer, dataOffset, header) {
    const view = new DataView(buffer);
    const offsets = propertyOffsets(header.properties);
    const points = allocatePoints(header.vertexCount);
    const bounds = freshBounds();
    for (let index = 0; index < header.vertexCount; index += 1) {
      const base = dataOffset + index * offsets.stride;
      const x = readBinary(view, base + offsets.x.offset, offsets.x.type);
      const y = readBinary(view, base + offsets.y.offset, offsets.y.type);
      const z = readBinary(view, base + offsets.z.offset, offsets.z.type);
      const color = readBinaryColor(view, base, offsets);
      writePoint(points, bounds, index, x, y, z, color);
    }
    return { points: points, bounds: bounds, count: header.vertexCount };
  }

  function propertyIndices(properties) {
    const names = properties.map((property) => property.name);
    const x = names.indexOf("x");
    const y = names.indexOf("y");
    const z = names.indexOf("z");
    if (x < 0 || y < 0 || z < 0) {
      throw new Error("PLY vertex coordinates are missing");
    }
    return {
      x: x,
      y: y,
      z: z,
      red: names.indexOf("red"),
      green: names.indexOf("green"),
      blue: names.indexOf("blue"),
    };
  }

  function propertyOffsets(properties) {
    const offsets = {};
    let offset = 0;
    for (const property of properties) {
      offsets[property.name] = { offset: offset, type: property.type };
      offset += typeSize(property.type);
    }
    if (!offsets.x || !offsets.y || !offsets.z) {
      throw new Error("PLY vertex coordinates are missing");
    }
    offsets.stride = offset;
    return offsets;
  }

  function typeSize(type) {
    const sizes = {
      char: 1,
      int8: 1,
      uchar: 1,
      uint8: 1,
      short: 2,
      int16: 2,
      ushort: 2,
      uint16: 2,
      int: 4,
      int32: 4,
      uint: 4,
      uint32: 4,
      float: 4,
      float32: 4,
      double: 8,
      float64: 8,
    };
    if (!Object.prototype.hasOwnProperty.call(sizes, type)) {
      throw new Error("unsupported PLY scalar type " + type);
    }
    return sizes[type];
  }

  function readBinary(view, offset, type) {
    if (type === "char" || type === "int8") {
      return view.getInt8(offset);
    }
    if (type === "uchar" || type === "uint8") {
      return view.getUint8(offset);
    }
    if (type === "short" || type === "int16") {
      return view.getInt16(offset, true);
    }
    if (type === "ushort" || type === "uint16") {
      return view.getUint16(offset, true);
    }
    if (type === "int" || type === "int32") {
      return view.getInt32(offset, true);
    }
    if (type === "uint" || type === "uint32") {
      return view.getUint32(offset, true);
    }
    if (type === "float" || type === "float32") {
      return view.getFloat32(offset, true);
    }
    if (type === "double" || type === "float64") {
      return view.getFloat64(offset, true);
    }
    throw new Error("unsupported PLY scalar type " + type);
  }

  function allocatePoints(count) {
    return {
      x: new Float32Array(count),
      y: new Float32Array(count),
      z: new Float32Array(count),
      r: new Uint8ClampedArray(count),
      g: new Uint8ClampedArray(count),
      b: new Uint8ClampedArray(count),
    };
  }

  function freshBounds() {
    return {
      minX: Infinity,
      maxX: -Infinity,
      minY: Infinity,
      maxY: -Infinity,
      minZ: Infinity,
      maxZ: -Infinity,
    };
  }

  function readAsciiColor(parts, indices) {
    if (indices.red < 0 || indices.green < 0 || indices.blue < 0) {
      return { r: 180, g: 184, b: 192 };
    }
    return {
      r: Number(parts[indices.red]),
      g: Number(parts[indices.green]),
      b: Number(parts[indices.blue]),
    };
  }

  function readBinaryColor(view, base, offsets) {
    if (!offsets.red || !offsets.green || !offsets.blue) {
      return { r: 180, g: 184, b: 192 };
    }
    return {
      r: readBinary(view, base + offsets.red.offset, offsets.red.type),
      g: readBinary(view, base + offsets.green.offset, offsets.green.type),
      b: readBinary(view, base + offsets.blue.offset, offsets.blue.type),
    };
  }

  function writePoint(points, bounds, index, x, y, z, color) {
    points.x[index] = x;
    points.y[index] = y;
    points.z[index] = z;
    points.r[index] = color.r;
    points.g[index] = color.g;
    points.b[index] = color.b;
    bounds.minX = Math.min(bounds.minX, x);
    bounds.maxX = Math.max(bounds.maxX, x);
    bounds.minY = Math.min(bounds.minY, y);
    bounds.maxY = Math.max(bounds.maxY, y);
    bounds.minZ = Math.min(bounds.minZ, z);
    bounds.maxZ = Math.max(bounds.maxZ, z);
  }

  function draw() {
    if (!state.points || !state.bounds || canvas.width < 1 || canvas.height < 1) {
      return;
    }
    const image = context.createImageData(canvas.width, canvas.height);
    paintImageBackground(image);
    const depthBuffer = new Float32Array(canvas.width * canvas.height);
    depthBuffer.fill(-Infinity);
    const points = state.points;
    const bounds = state.bounds;
    const rotationMatrix = quaternionToMatrix(state.orientation);
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    const centerZ = (bounds.minZ + bounds.maxZ) / 2;
    const rangeX = Math.max(0.001, bounds.maxX - bounds.minX);
    const rangeY = Math.max(0.001, bounds.maxY - bounds.minY);
    const rangeZ = Math.max(0.001, bounds.maxZ - bounds.minZ);
    const span = Math.max(rangeX, rangeY, rangeZ);
    const scale =
      (0.82 * state.scale * Math.min(canvas.width, canvas.height)) / span;
    for (let index = 0; index < points.x.length; index += 1) {
      const x = points.x[index] - centerX + state.offsetX;
      const y = points.y[index] - centerY + state.offsetY;
      const z = points.z[index] - centerZ + state.offsetZ;
      const projectedX =
        rotationMatrix[0][0] * x +
        rotationMatrix[0][1] * y +
        rotationMatrix[0][2] * z;
      const projectedY =
        rotationMatrix[1][0] * x +
        rotationMatrix[1][1] * y +
        rotationMatrix[1][2] * z;
      const depth =
        rotationMatrix[2][0] * x +
        rotationMatrix[2][1] * y +
        rotationMatrix[2][2] * z;
      const px = Math.round(canvas.width / 2 + projectedX * scale);
      const py = Math.round(canvas.height / 2 - projectedY * scale);
      plotPoint(
        image,
        depthBuffer,
        px,
        py,
        depth,
        points.r[index],
        points.g[index],
        points.b[index]
      );
    }
    context.putImageData(image, 0, 0);
    drawCameraVisualizations({
      bounds: bounds,
      rotationMatrix: rotationMatrix,
      scale: scale,
    });
  }

  function parseOpenCvC2wCameraPoses(text) {
    const cameras = [];
    const lines = text.split(/\r?\n/);
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const line = lines[lineIndex].trim();
      if (!line) {
        continue;
      }
      const values = line.split(/\s+/).map((value) => Number(value));
      if (values.length !== 16 || values.some((value) => !Number.isFinite(value))) {
        throw new Error("invalid OpenCV c2w camera pose row " + lineIndex);
      }
      const matrix = [
        values.slice(0, 4),
        values.slice(4, 8),
        values.slice(8, 12),
        values.slice(12, 16),
      ];
      cameras.push(createCameraVisualizationFromOpenCvC2w(matrix));
    }
    return cameras;
  }

  function createCameraVisualizationFromOpenCvC2w(matrix) {
    const center = {
      x: matrix[0][3],
      y: matrix[1][3],
      z: matrix[2][3],
    };
    const right = normalizeCameraBasisVector({
      x: matrix[0][0],
      y: matrix[1][0],
      z: matrix[2][0],
    });
    const down = normalizeCameraBasisVector({
      x: matrix[0][1],
      y: matrix[1][1],
      z: matrix[2][1],
    });
    const forward = normalizeCameraBasisVector({
      x: matrix[0][2],
      y: matrix[1][2],
      z: matrix[2][2],
    });
    return {
      center: center,
      right: right,
      forward: forward,
      up: scaleVector(down, -1),
    };
  }

  function drawCameraVisualizations(args) {
    const { bounds, rotationMatrix, scale } = args;
    const cameras = state.cameraVisualizations;
    if (!state.showCameras || cameras === null || cameras.length === 0) {
      window.__pointCloudViewerCameraOverlayDrawCount = 0;
      return;
    }
    let lineCount = 0;
    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = Math.max(2, 2 * (window.devicePixelRatio || 1));
    for (const camera of cameras) {
      for (const line of buildCameraVisualizationLines({ camera, bounds })) {
        drawProjectedLine({
          line,
          bounds,
          rotationMatrix,
          scale,
        });
        lineCount += 1;
      }
    }
    context.restore();
    window.__pointCloudViewerCameraOverlayDrawCount = lineCount;
  }

  function buildCameraVisualizationLines(args) {
    const { camera, bounds } = args;
    const span = sceneSpan(bounds);
    const axisLength = 0.035 * span;
    const frustumDepth = 0.055 * span;
    const frustumHalfSize = 0.5 * frustumDepth;
    const frustumCenter = addVectors(
      camera.center,
      scaleVector(camera.forward, frustumDepth)
    );
    const frustumPoints = [
      addVectors(
        addVectors(frustumCenter, scaleVector(camera.right, -frustumHalfSize)),
        scaleVector(camera.up, frustumHalfSize)
      ),
      addVectors(
        addVectors(frustumCenter, scaleVector(camera.right, frustumHalfSize)),
        scaleVector(camera.up, frustumHalfSize)
      ),
      addVectors(
        addVectors(frustumCenter, scaleVector(camera.right, frustumHalfSize)),
        scaleVector(camera.up, -frustumHalfSize)
      ),
      addVectors(
        addVectors(frustumCenter, scaleVector(camera.right, -frustumHalfSize)),
        scaleVector(camera.up, -frustumHalfSize)
      ),
    ];
    const lines = [
      {
        start: camera.center,
        end: addVectors(camera.center, scaleVector(camera.right, axisLength)),
        color: "#ef4444",
      },
      {
        start: camera.center,
        end: addVectors(camera.center, scaleVector(camera.forward, axisLength)),
        color: "#22c55e",
      },
      {
        start: camera.center,
        end: addVectors(camera.center, scaleVector(camera.up, axisLength)),
        color: "#3b82f6",
      },
    ];
    for (const frustumPoint of frustumPoints) {
      lines.push({
        start: camera.center,
        end: frustumPoint,
        color: "#f59e0b",
      });
    }
    for (let index = 0; index < frustumPoints.length; index += 1) {
      lines.push({
        start: frustumPoints[index],
        end: frustumPoints[(index + 1) % frustumPoints.length],
        color: "#f59e0b",
      });
    }
    return lines;
  }

  function sceneSpan(bounds) {
    const rangeX = Math.max(0.001, bounds.maxX - bounds.minX);
    const rangeY = Math.max(0.001, bounds.maxY - bounds.minY);
    const rangeZ = Math.max(0.001, bounds.maxZ - bounds.minZ);
    return Math.max(rangeX, rangeY, rangeZ);
  }

  function drawProjectedLine(args) {
    const { line, bounds, rotationMatrix, scale } = args;
    const start = projectWorldPoint({
      point: line.start,
      bounds,
      rotationMatrix,
      scale,
    });
    const end = projectWorldPoint({
      point: line.end,
      bounds,
      rotationMatrix,
      scale,
    });
    context.strokeStyle = line.color;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  }

  function projectWorldPoint(args) {
    const { point, bounds, rotationMatrix, scale } = args;
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    const centerZ = (bounds.minZ + bounds.maxZ) / 2;
    const x = point.x - centerX + state.offsetX;
    const y = point.y - centerY + state.offsetY;
    const z = point.z - centerZ + state.offsetZ;
    const projectedX =
      rotationMatrix[0][0] * x +
      rotationMatrix[0][1] * y +
      rotationMatrix[0][2] * z;
    const projectedY =
      rotationMatrix[1][0] * x +
      rotationMatrix[1][1] * y +
      rotationMatrix[1][2] * z;
    return {
      x: canvas.width / 2 + projectedX * scale,
      y: canvas.height / 2 - projectedY * scale,
    };
  }

  function scheduleDraw() {
    if (pendingDrawFrame !== null) {
      return;
    }
    pendingDrawFrame = window.requestAnimationFrame(() => {
      pendingDrawFrame = null;
      draw();
    });
  }

  function paintImageBackground(image) {
    for (let offset = 0; offset < image.data.length; offset += 4) {
      image.data[offset] = background.r;
      image.data[offset + 1] = background.g;
      image.data[offset + 2] = background.b;
      image.data[offset + 3] = background.a;
    }
  }

  function plotPoint(image, depthBuffer, x, y, depth, r, g, b) {
    if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) {
      return;
    }
    const pixelIndex = y * canvas.width + x;
    if (depth < depthBuffer[pixelIndex]) {
      return;
    }
    depthBuffer[pixelIndex] = depth;
    const offset = pixelIndex * 4;
    image.data[offset] = r;
    image.data[offset + 1] = g;
    image.data[offset + 2] = b;
    image.data[offset + 3] = 255;
  }

  function paintPlaceholder() {
    resizeCanvas();
    context.fillStyle = "#f8fafc";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#64748b";
    context.font = "16px system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText("Placeholder", canvas.width / 2, canvas.height / 2 - 8);
    context.fillText("No display artifact is available", canvas.width / 2, canvas.height / 2 + 18);
  }

  function buildCameraState() {
    const rotationMatrix = quaternionToMatrix(state.orientation);
    const forward = {
      x: rotationMatrix[2][0],
      y: rotationMatrix[2][1],
      z: rotationMatrix[2][2],
    };
    const yaw = Math.atan2(forward.x, forward.z);
    const pitch = Math.asin(clamp(forward.y, -1, 1));
    return {
      intrinsics: {
        projection: "orthographic-canvas",
        scale: state.scale,
      },
      extrinsics: {
        rotation: {
          yaw: yaw,
          pitch: pitch,
          roll: 0,
        },
        orientation: {
          x: state.orientation.x,
          y: state.orientation.y,
          z: state.orientation.z,
          w: state.orientation.w,
        },
        translation: {
          x: state.offsetX,
          y: state.offsetY,
          z: state.offsetZ,
        },
        view_up: {
          x: rotationMatrix[1][0],
          y: rotationMatrix[1][1],
          z: rotationMatrix[1][2],
        },
      },
      convention: "trackball_canvas_pointcloud",
      name: label,
      id: filePath || autoloadUrl || null,
    };
  }

  function isFiniteObjectVector(value) {
    return (
      value !== null &&
      typeof value === "object" &&
      Number.isFinite(value.x) &&
      Number.isFinite(value.y) &&
      Number.isFinite(value.z)
    );
  }

  function isPointCloudCameraState(cameraState) {
    return (
      cameraState !== null &&
      typeof cameraState === "object" &&
      cameraState.convention === "trackball_canvas_pointcloud" &&
      cameraState.intrinsics !== null &&
      typeof cameraState.intrinsics === "object" &&
      Number.isFinite(cameraState.intrinsics.scale) &&
      cameraState.extrinsics !== null &&
      typeof cameraState.extrinsics === "object" &&
      cameraState.extrinsics.rotation !== null &&
      typeof cameraState.extrinsics.rotation === "object" &&
      Number.isFinite(cameraState.extrinsics.rotation.yaw) &&
      Number.isFinite(cameraState.extrinsics.rotation.pitch) &&
      isFiniteObjectVector(cameraState.extrinsics.translation) &&
      isFiniteQuaternion(cameraState.extrinsics.orientation)
    );
  }

  function isFiniteQuaternion(value) {
    return (
      value !== null &&
      typeof value === "object" &&
      Number.isFinite(value.x) &&
      Number.isFinite(value.y) &&
      Number.isFinite(value.z) &&
      Number.isFinite(value.w)
    );
  }

  function isOpenCvCameraState(cameraState) {
    return (
      cameraState !== null &&
      typeof cameraState === "object" &&
      cameraState.convention === "opencv" &&
      cameraState.extrinsics !== null &&
      typeof cameraState.extrinsics === "object" &&
      isFiniteMatrix4x4(cameraState.extrinsics.matrix)
    );
  }

  function isFiniteMatrix4x4(value) {
    return (
      Array.isArray(value) &&
      value.length === 4 &&
      value.every(
        (row) =>
          Array.isArray(row) &&
          row.length === 4 &&
          row.every((item) => Number.isFinite(item))
      )
    );
  }

  function applyCameraState(cameraState) {
    const rendererCameraState = convertToPointCloudCameraState(cameraState);
    if (rendererCameraState === null) {
      return;
    }
    state.orientation = resolveCameraStateOrientation(rendererCameraState);
    state.scale = rendererCameraState.intrinsics.scale;
    state.offsetX = rendererCameraState.extrinsics.translation.x;
    state.offsetY = rendererCameraState.extrinsics.translation.y;
    state.offsetZ = rendererCameraState.extrinsics.translation.z;
    scheduleDraw();
    writeCameraStateForTests();
  }

  function convertToPointCloudCameraState(cameraState) {
    if (isPointCloudCameraState(cameraState)) {
      return cameraState;
    }
    if (isOpenCvCameraState(cameraState)) {
      return convertOpenCvCameraState(cameraState);
    }
    return null;
  }

  function convertOpenCvCameraState(cameraState) {
    const matrix = cameraState.extrinsics.matrix;
    const right = normalizeCameraBasisVector({
      x: matrix[0][0],
      y: matrix[1][0],
      z: matrix[2][0],
    });
    const down = normalizeCameraBasisVector({
      x: matrix[0][1],
      y: matrix[1][1],
      z: matrix[2][1],
    });
    const forward = normalizeCameraBasisVector({
      x: matrix[0][2],
      y: matrix[1][2],
      z: matrix[2][2],
    });
    const up = scaleVector(down, -1);
    const yaw = Math.atan2(forward.x, forward.z);
    const pitch = Math.asin(clamp(forward.y, -1, 1));
    return {
      intrinsics: {
        projection: "orthographic-canvas",
        scale: 1,
      },
      extrinsics: {
        rotation: {
          yaw: yaw,
          pitch: pitch,
          roll: 0,
        },
        orientation: quaternionFromMatrix([
          [right.x, right.y, right.z],
          [down.x, down.y, down.z],
          [forward.x, forward.y, forward.z],
        ]),
        translation: {
          x: 0,
          y: 0,
          z: 0,
        },
        view_up: up,
      },
      convention: "trackball_canvas_pointcloud",
      name: cameraState.name,
      id: cameraState.id,
    };
  }

  function resolveCameraStateOrientation(cameraState) {
    if (
      cameraState.convention === "trackball_canvas_pointcloud" &&
      isFiniteQuaternion(cameraState.extrinsics.orientation)
    ) {
      return normalizeQuaternion(cameraState.extrinsics.orientation);
    }
    return quaternionFromYawPitch(
      cameraState.extrinsics.rotation.yaw,
      cameraState.extrinsics.rotation.pitch
    );
  }

  function clamp(value, minValue, maxValue) {
    return Math.min(Math.max(value, minValue), maxValue);
  }

  function writeCameraStateForTests() {
    window.__pointCloudViewerCameraState = buildCameraState();
    window.__pointCloudViewerShowCameras = state.showCameras;
    window.__pointCloudViewerCameraVisualizationCount =
      state.cameraVisualizations === null ? 0 : state.cameraVisualizations.length;
  }

  function publishCameraState() {
    const cameraState = buildCameraState();
    window.__pointCloudViewerCameraState = cameraState;
    if (window.parent !== window) {
      window.parent.postMessage(
        {
          cameraState: cameraState,
          type: "trackball-camera-state-change",
        },
        window.location.origin
      );
    }
  }

  function scheduleCameraStatePublish() {
    if (pendingCameraStatePublishFrame !== null) {
      return;
    }
    pendingCameraStatePublishFrame = window.requestAnimationFrame(() => {
      pendingCameraStatePublishFrame = null;
      publishCameraState();
    });
  }

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) {
      return;
    }
    const message = event.data;
    if (
      message === null ||
      typeof message !== "object" ||
      (message.type !== "trackball-camera-state" &&
        message.type !== "show-cameras-state")
    ) {
      return;
    }
    if (message.type === "show-cameras-state") {
      state.showCameras = message.showCameras === true;
      writeCameraStateForTests();
      scheduleDraw();
      return;
    }
    applyCameraState(message.cameraState);
  });

  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });
  canvas.addEventListener("mousedown", (event) => {
    state.dragging = true;
    state.dragMode =
      event.shiftKey || event.button === 1 || event.button === 2
        ? "pan"
        : "rotate";
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    state.lastTrackballVector = mapPointerToTrackball(
      event.clientX,
      event.clientY
    );
  });
  window.addEventListener("mouseup", () => {
    state.dragging = false;
    state.lastTrackballVector = null;
  });
  window.addEventListener("mousemove", (event) => {
    if (!state.dragging) {
      return;
    }
    const dx = event.clientX - state.lastX;
    const dy = event.clientY - state.lastY;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    if (state.dragMode === "pan") {
      const rotationMatrix = quaternionToMatrix(state.orientation);
      const panScale = canvas.width > 0 ? 2 / canvas.width / state.scale : 0;
      state.offsetX +=
        (rotationMatrix[0][0] * dx - rotationMatrix[1][0] * dy) * panScale;
      state.offsetY +=
        (rotationMatrix[0][1] * dx - rotationMatrix[1][1] * dy) * panScale;
      state.offsetZ +=
        (rotationMatrix[0][2] * dx - rotationMatrix[1][2] * dy) * panScale;
    } else {
      const nextTrackballVector = mapPointerToTrackball(
        event.clientX,
        event.clientY
      );
      state.orientation = rotateTrackballOrientation({
        orientation: state.orientation,
        previousVector: state.lastTrackballVector,
        nextVector: nextTrackballVector,
      });
      state.lastTrackballVector = nextTrackballVector;
    }
    writeCameraStateForTests();
    scheduleDraw();
    scheduleCameraStatePublish();
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.scale *= event.deltaY < 0 ? 1.08 : 0.92;
    if (!Number.isFinite(state.scale) || state.scale <= 0) {
      state.scale = 1;
    }
    writeCameraStateForTests();
    scheduleDraw();
    scheduleCameraStatePublish();
  });
  window.addEventListener("resize", resizeCanvas);

  resizeCanvas();
  writeCameraStateForTests();
  load();

  function mapPointerToTrackball(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const shortestSide = Math.max(1, Math.min(rect.width, rect.height));
    const x = (2 * (clientX - rect.left) - rect.width) / shortestSide;
    const y = (rect.height - 2 * (clientY - rect.top)) / shortestSide;
    const radiusSquared = x * x + y * y;
    if (radiusSquared <= 1) {
      return normalizeVector({
        x: x,
        y: y,
        z: Math.sqrt(1 - radiusSquared),
      });
    }
    return normalizeVector({
      x: x,
      y: y,
      z: 0,
    });
  }

  function rotateTrackballOrientation(args) {
    const { orientation, previousVector, nextVector } = args;
    if (previousVector === null) {
      return orientation;
    }
    const rotationAxis = crossVectors(nextVector, previousVector);
    const rotationAxisNorm = vectorNorm(rotationAxis);
    if (!Number.isFinite(rotationAxisNorm) || rotationAxisNorm < 1e-8) {
      return orientation;
    }
    const rotationAngle =
      2.8 *
      Math.atan2(
        rotationAxisNorm,
        clamp(dotVectors(previousVector, nextVector), -1, 1)
      );
    const deltaQuaternion = quaternionFromAxisAngle(
      scaleVector(rotationAxis, 1 / rotationAxisNorm),
      rotationAngle
    );
    return normalizeQuaternion(
      multiplyQuaternions(deltaQuaternion, orientation)
    );
  }

  function quaternionFromYawPitch(yaw, pitch) {
    const yawCos = Math.cos(yaw);
    const yawSin = Math.sin(yaw);
    const pitchCos = Math.cos(pitch);
    const pitchSin = Math.sin(pitch);
    return quaternionFromMatrix([
      [yawCos, 0, -yawSin],
      [-yawSin * pitchSin, pitchCos, -yawCos * pitchSin],
      [yawSin * pitchCos, pitchSin, yawCos * pitchCos],
    ]);
  }

  function quaternionToMatrix(quaternion) {
    const normalizedQuaternion = normalizeQuaternion(quaternion);
    const x = normalizedQuaternion.x;
    const y = normalizedQuaternion.y;
    const z = normalizedQuaternion.z;
    const w = normalizedQuaternion.w;
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const wx = w * x;
    const wy = w * y;
    const wz = w * z;
    return [
      [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
      [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
      [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ];
  }

  function quaternionFromMatrix(matrix) {
    const trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    if (trace > 0) {
      const scale = Math.sqrt(trace + 1) * 2;
      return normalizeQuaternion({
        w: 0.25 * scale,
        x: (matrix[2][1] - matrix[1][2]) / scale,
        y: (matrix[0][2] - matrix[2][0]) / scale,
        z: (matrix[1][0] - matrix[0][1]) / scale,
      });
    }
    if (matrix[0][0] > matrix[1][1] && matrix[0][0] > matrix[2][2]) {
      const scale =
        Math.sqrt(1 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2;
      return normalizeQuaternion({
        w: (matrix[2][1] - matrix[1][2]) / scale,
        x: 0.25 * scale,
        y: (matrix[0][1] + matrix[1][0]) / scale,
        z: (matrix[0][2] + matrix[2][0]) / scale,
      });
    }
    if (matrix[1][1] > matrix[2][2]) {
      const scale =
        Math.sqrt(1 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2;
      return normalizeQuaternion({
        w: (matrix[0][2] - matrix[2][0]) / scale,
        x: (matrix[0][1] + matrix[1][0]) / scale,
        y: 0.25 * scale,
        z: (matrix[1][2] + matrix[2][1]) / scale,
      });
    }
    const scale =
      Math.sqrt(1 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2;
    return normalizeQuaternion({
      w: (matrix[1][0] - matrix[0][1]) / scale,
      x: (matrix[0][2] + matrix[2][0]) / scale,
      y: (matrix[1][2] + matrix[2][1]) / scale,
      z: 0.25 * scale,
    });
  }

  function quaternionFromAxisAngle(axis, angle) {
    const halfAngle = angle / 2;
    const sinHalfAngle = Math.sin(halfAngle);
    return {
      x: axis.x * sinHalfAngle,
      y: axis.y * sinHalfAngle,
      z: axis.z * sinHalfAngle,
      w: Math.cos(halfAngle),
    };
  }

  function multiplyQuaternions(left, right) {
    return {
      x:
        left.w * right.x +
        left.x * right.w +
        left.y * right.z -
        left.z * right.y,
      y:
        left.w * right.y -
        left.x * right.z +
        left.y * right.w +
        left.z * right.x,
      z:
        left.w * right.z +
        left.x * right.y -
        left.y * right.x +
        left.z * right.w,
      w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
    };
  }

  function normalizeQuaternion(quaternion) {
    const norm = Math.hypot(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
    if (!Number.isFinite(norm) || norm <= 0) {
      return { x: 0, y: 0, z: 0, w: 1 };
    }
    return {
      x: quaternion.x / norm,
      y: quaternion.y / norm,
      z: quaternion.z / norm,
      w: quaternion.w / norm,
    };
  }

  function normalizeVector(vector) {
    const norm = vectorNorm(vector);
    if (!Number.isFinite(norm) || norm <= 0) {
      return { x: 0, y: 0, z: 1 };
    }
    return scaleVector(vector, 1 / norm);
  }

  function normalizeCameraBasisVector(vector) {
    const norm = vectorNorm(vector);
    if (!Number.isFinite(norm) || norm <= 0) {
      throw new Error("invalid camera basis vector");
    }
    return scaleVector(vector, 1 / norm);
  }

  function crossVectors(left, right) {
    return {
      x: left.y * right.z - left.z * right.y,
      y: left.z * right.x - left.x * right.z,
      z: left.x * right.y - left.y * right.x,
    };
  }

  function dotVectors(left, right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
  }

  function scaleVector(vector, scale) {
    return {
      x: vector.x * scale,
      y: vector.y * scale,
      z: vector.z * scale,
    };
  }

  function addVectors(left, right) {
    return {
      x: left.x + right.x,
      y: left.y + right.y,
      z: left.z + right.z,
    };
  }

  function vectorNorm(vector) {
    return Math.hypot(vector.x, vector.y, vector.z);
  }
})();
