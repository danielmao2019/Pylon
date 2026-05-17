export function renderLayeredDisplayContainer({
  basePanel,
  originalOverlayPanel,
  cameraPanel,
  baseDisplaySlotId,
  showOriginalOverlay,
  showCameras,
  onOriginalOverlayVisibilityChange,
}: {
  basePanel: HTMLElement;
  originalOverlayPanel?: HTMLElement;
  cameraPanel?: HTMLElement;
  baseDisplaySlotId: string;
  showOriginalOverlay?: boolean;
  showCameras?: boolean;
  onOriginalOverlayVisibilityChange?: (
    showOriginalOverlay: boolean,
  ) => void;
}): HTMLElement {
  // Generic container owns layer composition; route/view state owns visibility values.
  if (!(basePanel instanceof HTMLElement)) {
    throw new Error("layered display base panel is missing");
  }
  if (baseDisplaySlotId.length === 0) {
    throw new Error("layered display base slot id is missing");
  }

  const container = document.createElement("div");
  container.className = "layered-display-container";
  container.dataset.baseDisplaySlotId = baseDisplaySlotId;

  const baseBody = basePanel.querySelector(".display-panel__body");
  if (!(baseBody instanceof HTMLElement)) {
    throw new Error("display panel body is missing");
  }
  baseBody.classList.add("display-panel__body--layered");

  const baseLayer = baseBody.firstElementChild;
  if (!(baseLayer instanceof HTMLElement)) {
    throw new Error("display panel rendered element is missing");
  }
  baseLayer.classList.add(
    "layered-display-container__layer",
    "layered-display-container__layer--base",
  );
  baseLayer.style.position = "absolute";
  baseLayer.style.inset = "0";
  baseLayer.style.width = "100%";
  baseLayer.style.height = "100%";

  let originalOverlayToggle: HTMLElement | undefined;
  if (originalOverlayPanel !== undefined) {
    if (typeof showOriginalOverlay !== "boolean") {
      throw new Error("original-overlay visibility must be boolean");
    }
    if (typeof onOriginalOverlayVisibilityChange !== "function") {
      throw new Error("original-overlay visibility handler is missing");
    }
    const originalOverlayBody = originalOverlayPanel.querySelector(
      ".display-panel__body",
    );
    if (!(originalOverlayBody instanceof HTMLElement)) {
      throw new Error("display panel body is missing");
    }
    const originalOverlayLayer = originalOverlayBody.firstElementChild;
    if (!(originalOverlayLayer instanceof HTMLElement)) {
      throw new Error("display panel rendered element is missing");
    }
    originalOverlayLayer.remove();
    originalOverlayLayer.classList.add(
      "layered-display-container__layer",
      "layered-display-container__layer--original-overlay",
    );
    originalOverlayLayer.style.position = "absolute";
    originalOverlayLayer.style.inset = "0";
    originalOverlayLayer.style.width = "100%";
    originalOverlayLayer.style.height = "100%";
    originalOverlayLayer.hidden = !showOriginalOverlay;
    baseBody.append(originalOverlayLayer);

    const toggle = document.createElement("label");
    toggle.className = "layered-display-container__toggle";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = showOriginalOverlay;
    const onChange = onOriginalOverlayVisibilityChange;
    input.addEventListener("change", () => {
      originalOverlayLayer.hidden = !input.checked;
      onChange(input.checked);
    });
    const label = document.createElement("span");
    label.textContent = "Original Overlay";
    toggle.append(input, label);
    originalOverlayToggle = toggle;
  }

  if (cameraPanel !== undefined) {
    if (typeof showCameras !== "boolean") {
      throw new Error("show-cameras visibility must be boolean");
    }
    const cameraBody = cameraPanel.querySelector(".display-panel__body");
    if (!(cameraBody instanceof HTMLElement)) {
      throw new Error("display panel body is missing");
    }
    const cameraLayer = cameraBody.firstElementChild;
    if (!(cameraLayer instanceof HTMLElement)) {
      throw new Error("display panel rendered element is missing");
    }
    cameraLayer.remove();
    cameraLayer.classList.add(
      "layered-display-container__layer",
      "layered-display-container__layer--camera",
    );
    cameraLayer.style.position = "absolute";
    cameraLayer.style.inset = "0";
    cameraLayer.style.width = "100%";
    cameraLayer.style.height = "100%";
    cameraLayer.hidden = !showCameras;
    cameraLayer.dataset.showCameras = showCameras ? "true" : "false";
    baseBody.append(cameraLayer);
  }

  container.append(basePanel);
  if (originalOverlayToggle !== undefined) {
    container.append(originalOverlayToggle);
  }
  return container;
}
