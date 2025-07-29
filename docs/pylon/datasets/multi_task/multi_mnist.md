# Multi-MNIST Dataset

## Overview

The Multi-MNIST dataset is a synthetic dataset created by overlaying two MNIST digits to create a multi-task learning problem. Each image contains two overlapping digits, and the tasks are to classify both the top-left and bottom-right digits. This dataset is commonly used for evaluating multi-task learning algorithms on a controlled problem.

## Dataset Information

- **Type**: Synthetic overlapping digit images
- **Tasks**: 
  - Top-left digit classification (10 classes)
  - Bottom-right digit classification (10 classes)
- **Dataset Size**:
  - Train: 60,000 images
  - Test: 10,000 images
- **Image Size**: 36×36 pixels

## Data Structure

The dataset consists of grayscale images with two overlapping MNIST digits and corresponding class labels.

### Input Format

- `image`: Grayscale image with two overlapping digits

### Label Format

- `top_left_digit`: Class label for the top-left digit (0-9)
- `bottom_right_digit`: Class label for the bottom-right digit (0-9)

## Usage in Pylon

```python
from data.datasets.multi_task_datasets import MultiMNISTDataset

# Create dataset
dataset = MultiMNISTDataset(
    data_root="/path/to/MNIST",  # Path to original MNIST dataset
    split="train"
)

# Get a sample
inputs, labels, meta_info = dataset[0]

# inputs contains 'image'
# labels contains 'top_left_digit' and 'bottom_right_digit'

# Visualization example
dataset.visualize(output_dir="./visualization")
```

## Data Preparation

1. The Multi-MNIST dataset is generated on-the-fly from the original MNIST dataset
2. Download the MNIST dataset (automatically downloaded by torchvision)
3. Create a soft link to the MNIST directory in your Pylon project (if needed):

```bash
ln -s /path/to/MNIST <Pylon_path>/data/datasets/soft_links/MNIST
```

## Implementation Details

- **Image Generation**:
  - Two random MNIST digits are selected
  - The first digit is placed in the top-left corner
  - The second digit is placed in the bottom-right corner
  - The digits overlap in the center of the image
- **Offset Parameters**:
  - The amount of overlap can be controlled by adjusting the offset parameters
  - Default offset is 4 pixels, creating a moderate overlap
- **Data Augmentation**:
  - No additional augmentation is applied by default
  - The synthetic nature of the dataset already provides sufficient variability

## Research Papers Using This Dataset

- [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)
- [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704)
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
- [Uncertainty-Guided Continual Learning with Bayesian Neural Networks](https://arxiv.org/abs/1906.02425)
- [Auxiliary Task Reweighting for Minimum-data Learning](https://papers.nips.cc/paper/2020/hash/95f2b84de5660ddf45c8a34933a2e66f-Abstract.html)
