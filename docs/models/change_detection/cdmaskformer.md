# CDMaskFormer - Change Detection using MaskFormer

This repository contains an implementation of CDMaskFormer, a model for remote sensing change detection based on the MaskFormer architecture. The model is designed to detect changes between two images using a mask-based approach, leveraging the strengths of the MaskFormer framework for semantic segmentation.

## Model Architecture

CDMaskFormer consists of the following key components:

1. **Backbone**: A standard CNN backbone (e.g., ResNet50) used to extract features from input images.
2. **CDMask**: The core component that processes features from the backbone and produces change detection masks:
   - **TFF (Temporal Feature Fusion)**: Modules that fuse and compare features from two images at different scales
   - **SaELayer (Spatial and channel Attention)**: Attention mechanism for better feature learning
   - **MaskFormerHead**: Generates mask predictions and classifications using:
     - Pixel Decoder: Based on MSDeformAttnPixelDecoder, upsamples and refines features
     - Multi-Scale Masked Transformer Decoder: Predicts mask embeddings and classifications

## Usage

To use the CDMaskFormer model in your change detection pipeline:

```python
from models.change_detection.cdmaskformer.models.cdmaskformer import CDMaskFormer

# Initialize the model
backbone_name = 'resnet50'
backbone_args = {'pretrained': True}
cdmask_args = {
    'num_classes': 1,
    'num_queries': 10,
    'dec_layers': 6
}

model = CDMaskFormer(backbone_name, backbone_args, cdmask_args)

# Process a pair of images
output = model({
    'img_1': first_image,  # [B, C, H, W]
    'img_2': second_image  # [B, C, H, W]
})

# During training, output is a dict with keys 'pred_logits' and 'pred_masks'
# During inference, output is a tensor of shape [B, C, H, W] representing the change map
```

## Loss Function

The model uses a specialized criterion that combines:

1. Binary cross-entropy loss for mask classification
2. Dice loss for mask prediction quality

The CDMaskFormerCriterion inherits from SingleTaskCriterion and implements the standard interface used across the Pylon framework.

```python
from criteria.vision_2d.change_detection import CDMaskFormerCriterion

criterion = CDMaskFormerCriterion(
    num_classes=1,
    ignore_value=255,
    dice_weight=1.0,
    ce_weight=1.0
)

# Compute loss
loss = criterion(model_outputs, targets)
```

## Implementation Details

- The model is designed to handle both training and inference modes
- During training, it outputs the raw predictions for loss computation
- During inference, it performs semantic inference to produce the final change map
- The Hungarian Matcher is used to match predictions with ground truth
- Custom TFF modules combine features from both images to detect changes

## References

- MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation (https://arxiv.org/abs/2107.06278)
- Deformable Attention used in the pixel decoder (https://arxiv.org/abs/2010.04159)
