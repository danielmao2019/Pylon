# Pylon

Pylon is a deep learning library built on top of PyTorch, primarily focused on computer vision research.

## Motivation and Vision

While there are already many popular libraries such as `openmmlab`, `detectron`, and `huggingface`, they are difficult to expand and transfer between. As a result, research project code bases are isolated from each other and hard to be benchmarked in a unified framework.

We aim to utilize Python native objects as much as possible, design the APIs of abstract base classes, and encourage users to inherit from our base classes for project-specific use cases. We do our best in design to minimize the effort for users in writing subclasses and provide classic examples in different areas such as image classification, object detection, semantic/instance segmentation, domain adaptation, etc.

We maintain flexible API to models defined in existing framework like `mmdetection` and `detectron2`, however, we regularize the training and evaluation pipelines by our datasets, transforms, collate functions, criteria, metrics, and trainer classes.

## Naming

The name of this library is inspired by the protoss structure [Pylon](https://starcraft.fandom.com/wiki/Pylon) from the video game StarCraft II. Pylon serves as a fundamental structure for protoss players as it generates power fields in which other protoss units and structures could be deployed.

## Documentation Structure

The documentation is organized as follows:

- **General Documentation**
  - [Environment Setup](environment_setup.md) - How to set up the development environment
  - [Index](index.md) - Documentation navigation

- **Models**
  - Change Detection
    - [I3PE Model](models/change_detection/i3pe.md)
    - [Siamese KPConv Model](models/change_detection/siamese_kpconv.md)
    - [FTN Model](models/change_detection/ftn.md)
  
- **Datasets**
  - Change Detection
    - [SLPCCD Dataset](datasets/change_detection/slpccd.md)
  
- **Optimizers**
  - Multi-Task Optimizers
    - [PCGrad Derivation](optimizers/multi_task_optimizers/pcgrad_derivation.md)

### Testing

```bash
# Run tests
pytest tests/
```

## Contributors

* Dayou (Daniel) Mao, {[daniel.mao@uwaterloo.ca](mailto:daniel.mao@uwaterloo.ca)}.

    Cheriton School of Computer Science, University of Waterloo

    Vision and Image Processing Research Group, University of Waterloo

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
