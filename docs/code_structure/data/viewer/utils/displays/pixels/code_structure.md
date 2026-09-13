# Data Viewer Pixels Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/pixels/dash/apis.py`

```text
apis.py
├── from typing import Dict, Tuple
├── import torch
├── from dash import dcc
├── from data.viewer.utils.displays.pixels.dash.core_pixels_display import create_dash_pixels_display
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── from data.viewer.utils.displays.utils.heatmap_colors import map_scalars_to_rgb
├── from utils.io.image import load_image
├── DEFAULT_COLOR_IMAGE_INTERPOLATION = "linear"                # color images: linear interpolation smooths between RGB samples, appropriate for natural-image content
├── DEFAULT_DEPTH_IMAGE_INTERPOLATION = "nearest"               # depth images: nearest preserves exact metric depth samples; linear would invent midpoint depths that don't exist in the data
├── DEFAULT_EDGE_IMAGE_INTERPOLATION = "nearest"                # edge images: nearest preserves edge crispness; linear would smooth edges and defeat their purpose
├── DEFAULT_NORMAL_IMAGE_INTERPOLATION = "nearest"              # normal images: nearest preserves unit-length normal vectors; linear interpolation between normals produces non-unit results
├── DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION = "nearest"        # segmentation images: nearest preserves class-id integrity; linear would invent fractional class ids
├── DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION = "nearest"  # instance-surrogate images: nearest preserves class-id integrity (same reason as segmentation)
├── def create_color_image_display(color_image_path: str, image_interpolation: str = DEFAULT_COLOR_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Builds a Dash color-image display from an image path, defaulting to linear interpolation.
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_depth_image_display(depth_image_path: str, image_interpolation: str = DEFAULT_DEPTH_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Builds a Dash depth-image display from a depth-map path, colorizing it through the heatmap palette.
│   ├── calls _map_depth_image_to_rgb
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_edge_image_display(edge_image_path: str, image_interpolation: str = DEFAULT_EDGE_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Builds a Dash edge-image display from an edge-map path, colorizing it to RGB.
│   ├── calls _map_edge_image_to_rgb
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_normal_image_display(normal_image_path: str, image_interpolation: str = DEFAULT_NORMAL_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Builds a Dash normal-image display from a normal-map path, colorizing the normal vectors to RGB.
│   ├── calls _map_normal_image_to_rgb
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_segmentation_image_display(segmentation_image_path: str, image_interpolation: str = DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Renders the backend-colorized segmentation image display.
│   ├── impls reads segmentation image tensor from segmentation_image_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_image))
│   ├── calls _map_segmentation_image_to_rgb(segmentation_image_path=segmentation_image_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_instance_surrogate_image_display(image_path: str, image_interpolation: str = DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Renders the backend-colorized instance-surrogate image display.
│   ├── impls builds integer instance-surrogate class-id image from offset-magnitude quantile bins
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(instance_surrogate_class_id_image))
│   ├── calls _map_instance_surrogate_image_to_rgb(image_path=image_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def _map_depth_image_to_rgb(depth_image_path: str) -> torch.Tensor
│   ├── # Maps the depth image to RGB through the continuous heatmap palette for Dash display.
│   ├── impls assert isinstance(depth_image_path, str)
│   ├── calls load_image(filepath=depth_image_path, normalization=None)
│   ├── if depth_image.ndim == 3
│   │   └── impls depth_image = depth_image[0]
│   ├── impls depth_scalars = depth_image.to(torch.float64)
│   ├── impls depth_scalars = depth_scalars - float(depth_scalars.min().item())
│   ├── calls map_scalars_to_rgb(scalars=depth_scalars)                                           → rgb_image
│   └── return rgb_image
├── def _map_edge_image_to_rgb(edge_image_path: str) -> torch.Tensor
│   ├── # Maps the edge image to RGB for Dash display.
│   ├── impls assert isinstance(edge_image_path, str)
│   ├── calls load_image(filepath=edge_image_path, normalization=None)
│   ├── if edge_image.ndim == 3
│   │   └── impls edge_image = edge_image[0]
│   ├── impls edge_float = edge_image.to(torch.float64)
│   ├── impls edge_min = float(edge_float.min().item())
│   ├── impls edge_max = float(edge_float.max().item())
│   ├── impls normalized = (edge_float - edge_min) / max(edge_max - edge_min, 1e-12)
│   ├── impls gray = (normalized * 255.0).round().clamp(min=0.0, max=255.0).to(torch.uint8)
│   ├── impls gray_hwc = gray.unsqueeze(-1)
│   ├── impls rgb_image = gray_hwc.repeat(1, 1, 3)
│   └── return rgb_image
├── def _map_normal_image_to_rgb(normal_image_path: str) -> torch.Tensor
│   ├── # Maps the normal vectors to RGB for Dash display.
│   ├── impls assert isinstance(normal_image_path, str)
│   ├── calls load_image(filepath=normal_image_path, normalization=None)
│   ├── impls normal_float = normal_image.to(torch.float64)
│   ├── impls normal_float = normal_float / 127.5 - 1.0  # decodes the stored bytes back to normal components in [-1, 1]
│   ├── impls normals_normalized = (normal_float + 1.0) / 2.0
│   ├── impls normals_normalized = normals_normalized.clamp(min=0.0, max=1.0)
│   ├── impls rgb = (normals_normalized * 255.0).round().clamp(min=0.0, max=255.0).to(torch.uint8)
│   ├── impls rgb_image = rgb.permute(1, 2, 0)
│   └── return rgb_image
├── def _map_segmentation_image_to_rgb(segmentation_image_path: str, class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> torch.Tensor
│   ├── # Maps the segmentation image's per-pixel class ids to RGB via the class-to-RGB mapping for Dash display.
│   ├── impls assert isinstance(segmentation_image_path, str)
│   ├── impls assert isinstance(class_id_to_rgb, dict)
│   ├── calls load_image(filepath=segmentation_image_path, normalization=None)
│   ├── impls segmentation_image = that loaded image cast to torch.int64
│   ├── impls assert segmentation_image.ndim == 2
│   ├── impls height, width = segmentation_image.shape
│   ├── impls rgb_image = a uint8 zeros tensor of shape (height, width, 3)
│   ├── for each class_id, color in class_id_to_rgb.items()
│   │   └── impls rgb_image[segmentation_image == class_id] = color  # a class id with no mapping entry stays black
│   └── return rgb_image  # an HWC uint8 RGB image
└── def _map_instance_surrogate_image_to_rgb(image_path: str, class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> torch.Tensor
    ├── # Maps the instance-surrogate offset image to RGB via the class-to-RGB mapping for Dash display.
    ├── impls assert isinstance(image_path, str)
    ├── impls assert isinstance(class_id_to_rgb, dict)
    ├── calls load_image(filepath=image_path, normalization=None)
    ├── impls assert instance_surrogate is a 3-D tensor whose first dimension is at least 2
    ├── impls y_offset = instance_surrogate[0].to(torch.float64)
    ├── impls x_offset = instance_surrogate[1].to(torch.float64)
    ├── impls magnitude = torch.sqrt(y_offset**2 + x_offset**2)
    ├── impls class_id_image = torch.zeros_like(magnitude, dtype=torch.int64)
    ├── impls percentiles = torch.quantile(magnitude.reshape(-1), torch.linspace(0, 1, 20, dtype=torch.float64))
    ├── for bin_index in range(len(percentiles) - 1)
    │   ├── if bin_index == len(percentiles) - 2
    │   │   └── impls mask = magnitude >= percentiles[bin_index]
    │   ├── else
    │   │   └── impls mask = (magnitude >= percentiles[bin_index]) & (magnitude < percentiles[bin_index + 1])
    │   └── impls class_id_image[mask] = bin_index + 1
    ├── impls height, width = class_id_image.shape
    ├── impls rgb_image = a uint8 zeros tensor of shape (height, width, 3)
    ├── for each class_id, color in class_id_to_rgb.items()
    │   └── impls rgb_image[class_id_image == class_id] = color  # a class id with no mapping entry stays black
    └── return rgb_image  # an HWC uint8 RGB image
```

`data/viewer/utils/displays/pixels/dash/core_pixels_display.py`

```text
core_pixels_display.py
├── from typing import Any
├── import plotly.graph_objects as go
├── import torch
├── from dash import dcc
└── def create_dash_pixels_display(image: Any, image_interpolation: str) -> dcc.Graph
    ├── # Renders a Dash pixel-image display element from the resolved interpolation choice; modality-agnostic.
    ├── if isinstance(image, torch.Tensor)
    │   └── impls image_array = image.detach().cpu().numpy()
    ├── else
    │   └── impls image_array = image
    ├── impls assert image_array has shape [H, W, 3]
    ├── impls zsmooth = False when image_interpolation is "nearest", "fast" otherwise  # the caller's per-modality interpolation choice
    ├── impls figure = a go.Figure over a go.Image trace of image_array carrying that zsmooth
    ├── impls hide figure's axes, letting the image fill the cell at its own aspect ratio
    ├── impls display = dcc.Graph(figure=figure)
    └── return display  # the pixel display element
```

### Backend schemas

`data/viewer/utils/displays/pixels/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class PixelDisplayResponse(DisplayResponse)
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class DepthImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "depth_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class EdgeImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "edge_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class NormalImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "normal_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── class SegmentationImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "segmentation_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── class InstanceSurrogateImageDisplayResponse(PixelDisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "instance_surrogate_image"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

### Backend

`data/viewer/utils/displays/pixels/ts/backend/apis.py`

```text
apis.py
├── import torch
├── from data.viewer.utils.displays.pixels.ts.backend.core_pixels_display import create_pixels_display_response_core
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_image_display_response
│   ├── # intentional thin wrapper: passes color image directly to core response
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_depth_image_display_response
│   ├── # maps depth image to color image before core response
│   ├── calls _map_depth_image_to_rgb
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_edge_image_display_response
│   ├── # maps edge image to color image before core response
│   ├── calls _map_edge_image_to_rgb
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_normal_image_display_response
│   ├── # maps normal image to color image before core response
│   ├── calls _map_normal_image_to_rgb
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_segmentation_image_display_response
│   ├── # Creates a segmentation image response from a class-labeled image resource.
│   ├── impls reads segmentation image tensor from segmentation_image_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_image))
│   ├── calls _map_segmentation_image_to_rgb(segmentation_image_path=segmentation_image_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_segmentation_image_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_instance_surrogate_image_display_response
│   ├── # maps instance-surrogate image to color image before core response
│   ├── impls builds integer instance-surrogate class-id image from offset-magnitude quantile bins
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(instance_surrogate_class_id_image))
│   ├── calls _map_instance_surrogate_image_to_rgb(image_path=image_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_instance_surrogate_image_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_pixels_display_response_core
│   └── return
├── def _map_depth_image_to_rgb
│   └── # Writes a backend-colorized image resource by mapping the depth image through the continuous heatmap palette.
├── def _map_edge_image_to_rgb
│   └── # Writes a backend-colorized image resource by mapping the edge image to RGB.
├── def _map_normal_image_to_rgb
│   └── # Writes a backend-colorized image resource by mapping the normal vectors to RGB.
├── def _map_segmentation_image_to_rgb
│   └── # Writes a backend-colorized image resource by applying the class-to-RGB mapping to the segmentation image.
├── def _build_segmentation_image_meta_info
│   ├── # Builds factual class/color metadata from the class-to-RGB mapping.
│   ├── impls stores `class_id_to_rgb`
│   └── return
├── def _map_instance_surrogate_image_to_rgb
│   └── # Writes a backend-colorized image resource by applying the class-to-RGB mapping to the instance-surrogate class-id image.
└── def _build_instance_surrogate_image_meta_info
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`data/viewer/utils/displays/pixels/ts/backend/core_pixels_display.py`

```text
core_pixels_display.py
└── def create_pixels_display_response_core
    ├── # Creates a pixel-image display response from the loadable image resource path and caller-provided display metadata.
    ├── impls builds frontend resource url
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

### Frontend

`data/viewer/utils/displays/pixels/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
├── interface PixelDisplayResponse extends DisplayResponse
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface DepthImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "depth_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface EdgeImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "edge_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface NormalImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "normal_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
├── interface SegmentationImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "segmentation_image"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── interface InstanceSurrogateImageDisplayResponse extends PixelDisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "instance_surrogate_image"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

`data/viewer/utils/displays/pixels/ts/frontend/apis.ts`

```text
apis.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { ColorImageDisplayResponse, DepthImageDisplayResponse, EdgeImageDisplayResponse, InstanceSurrogateImageDisplayResponse, NormalImageDisplayResponse, SegmentationImageDisplayResponse } from "./types/display_response";
├── import { renderPixelsDisplay } from "./core_pixels_display";
├── import { registerRasterLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── const DEFAULT_COLOR_IMAGE_INTERPOLATION = "linear"                # color images: linear interpolation smooths between RGB samples, appropriate for natural-image content
├── const DEFAULT_DEPTH_IMAGE_INTERPOLATION = "nearest"               # depth images: nearest preserves exact metric depth samples; linear would invent midpoint depths that don't exist in the data
├── const DEFAULT_EDGE_IMAGE_INTERPOLATION = "nearest"                # edge images: nearest preserves edge crispness; linear would smooth edges and defeat their purpose
├── const DEFAULT_NORMAL_IMAGE_INTERPOLATION = "nearest"              # normal images: nearest preserves unit-length normal vectors; linear interpolation between normals produces non-unit results
├── const DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION = "nearest"        # segmentation images: nearest preserves class-id integrity; linear would invent fractional class ids
├── const DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION = "nearest"  # instance-surrogate images: nearest preserves class-id integrity (same reason as segmentation)
├── function renderColorImageDisplay({ displayResponse, imageInterpolation = DEFAULT_COLOR_IMAGE_INTERPOLATION }: { displayResponse: ColorImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders a color-image display, defaulting to linear interpolation for natural-image content.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderDepthImageDisplay({ displayResponse, imageInterpolation = DEFAULT_DEPTH_IMAGE_INTERPOLATION }: { displayResponse: DepthImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders a depth-image display, defaulting to nearest interpolation to preserve exact metric depths.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderEdgeImageDisplay({ displayResponse, imageInterpolation = DEFAULT_EDGE_IMAGE_INTERPOLATION }: { displayResponse: EdgeImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders an edge-image display, defaulting to nearest interpolation to preserve edge crispness.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderNormalImageDisplay({ displayResponse, imageInterpolation = DEFAULT_NORMAL_IMAGE_INTERPOLATION }: { displayResponse: NormalImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders a normal-image display, defaulting to nearest interpolation to preserve unit-length normals.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderSegmentationImageDisplay({ displayResponse, imageInterpolation = DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION }: { displayResponse: SegmentationImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders the backend-colorized segmentation display and legend derived from meta_info.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderInstanceSurrogateImageDisplay({ displayResponse, imageInterpolation = DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION }: { displayResponse: InstanceSurrogateImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders the backend-colorized image display and legend derived from meta_info.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
└── impls registerRasterLayerRenderer({ displayKind: "color_image", layerRenderer: renderColorImageDisplay })  # module-load self-registration of the raster color-image layer renderer
```

`data/viewer/utils/displays/pixels/ts/frontend/core_pixels_display.ts`

```text
core_pixels_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { PixelDisplayResponse } from "./types/display_response";
└── function renderPixelsDisplay({ displayResponse, imageInterpolation }: { displayResponse: PixelDisplayResponse; imageInterpolation: string }): LeafVNode
    ├── # Renders a self-contained pixel-image display element from the resolved interpolation choice; modality-agnostic.
    └── return LeafVNode keyed by displayResponse.url
```
