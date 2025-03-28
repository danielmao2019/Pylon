# Pylon

Pylon is a deep learning library built on top of PyTorch, primarily focused on computer vision research.

## Motivation and Vision

While there are already many popular libraries such as `openmmlab`, `detectron`, and `huggingface`, they are difficult to expand and transfer between. As a result, research project code bases are isolated from each other and hard to be benchmarked in a unified framework.

We aim to utilize Python native objects as much as possible, design the APIs of abstract base classes, and encourage users to inherit from our base classes for project-specific use cases. We do our best in design to minimize the effort for users in writing subclasses and provide classic examples in different areas such as image classification, object detection, semantic/instance segmentation, domain adaptation, etc.

We maintain flexible API to models defined in existing framework like `mmdetection` and `detectron2`, however, we regularize the training and evaluation pipelines by our datasets, transforms, collate functions, criteria, metrics, and trainer classes.

## Naming

The name of this library is inspired by the protoss structure [Pylon](https://starcraft.fandom.com/wiki/Pylon) from the video game StarCraft II. Pylon serves as a fundamental structure for protoss players as it generates power fields in which other protoss units and structures could be deployed.

## Documentation

All documentation can be found in the [docs](docs/) directory. For environment setup instructions, see [docs/environment_setup.md](docs/environment_setup.md).

## Contributors

* Dayou (Daniel) Mao, {[daniel.mao@uwaterloo.ca](mailto:daniel.mao@uwaterloo.ca)}.

    Cheriton School of Computer Science, University of Waterloo

    Vision and Image Processing Research Group, University of Waterloo
