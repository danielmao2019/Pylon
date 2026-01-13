"""Base class for image classification datasets."""

from typing import Dict, Any, Optional, List
import torch
import numpy as np
from data.datasets.base_dataset import BaseDataset


class BaseImgClsDataset(BaseDataset):
    """Base class for image classification datasets.

    This class provides the standard INPUT_NAMES, LABEL_NAMES, and display_datapoint
    method for image classification datasets. Concrete dataset classes should inherit
    from this class to automatically get appropriate display functionality.

    Expected data structure:
    - inputs: {'image': torch.Tensor}  # Shape: (C, H, W) or (H, W) for grayscale
    - labels: {'label': torch.Tensor}  # Shape: () for single label or (num_classes,) for multi-label
    """

    INPUT_NAMES = ['image']
    LABEL_NAMES = ['label']

    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> 'html.Div':
        """Display image classification datapoint.

        Args:
            datapoint: Dictionary containing 'inputs' and 'labels'
            class_labels: Optional dictionary mapping label names to lists of class names
            camera_state: Not used for 2D images
            settings_3d: Not used for 2D images

        Returns:
            Dash HTML component for visualization
        """
        import dash.html as html
        import dash.dcc as dcc
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Extract image and label
        inputs = datapoint.get('inputs', {})
        labels = datapoint.get('labels', {})

        image = inputs.get('image')
        label = labels.get('label')

        if image is None:
            return html.Div("No image data available")

        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image

        # Handle different image formats
        if image_np.ndim == 2:
            # Grayscale image (H, W)
            height, width = image_np.shape
            channels = 1
            display_image = image_np
        elif image_np.ndim == 3:
            if image_np.shape[0] in [1, 3, 4]:
                # Channel-first format (C, H, W)
                channels, height, width = image_np.shape
                if channels == 1:
                    display_image = image_np[0]
                elif channels == 3:
                    # RGB - transpose to (H, W, C)
                    display_image = np.transpose(image_np, (1, 2, 0))
                elif channels == 4:
                    # RGBA - transpose to (H, W, C)
                    display_image = np.transpose(image_np, (1, 2, 0))
                else:
                    return html.Div(f"Unsupported number of channels: {channels}")
            else:
                # Assume channel-last format (H, W, C)
                height, width, channels = image_np.shape
                display_image = image_np
        else:
            return html.Div(f"Unsupported image shape: {image_np.shape}")

        # Normalize image to [0, 1] if needed
        if display_image.min() < 0 or display_image.max() > 1:
            display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-8)

        # Create figure
        fig = make_subplots(rows=1, cols=1)

        # Add image
        if channels == 1 or display_image.ndim == 2:
            # Grayscale
            fig.add_trace(
                go.Heatmap(
                    z=display_image,
                    colorscale='gray',
                    showscale=False
                ),
                row=1, col=1
            )
        else:
            # Color image
            fig.add_trace(
                go.Image(z=display_image * 255),
                row=1, col=1
            )

        # Update layout
        title_parts = [f"Image ({height}x{width}x{channels})"]

        # Add label information
        if label is not None:
            if isinstance(label, torch.Tensor):
                label_val = label.item() if label.numel() == 1 else label.tolist()
            else:
                label_val = label

            # Add class name if available
            if class_labels and 'label' in class_labels and isinstance(label_val, (int, np.integer)):
                if 0 <= label_val < len(class_labels['label']):
                    class_name = class_labels['label'][label_val]
                    title_parts.append(f"Class: {class_name} ({label_val})")
                else:
                    title_parts.append(f"Label: {label_val}")
            else:
                title_parts.append(f"Label: {label_val}")

        fig.update_layout(
            title=" | ".join(title_parts),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x"),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return html.Div([
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': True, 'displaylogo': False}
            )
        ])