# Pylon

Pylon is a deep learning library built on top of PyTorch, primarily focused on computer vision research.

## Motivation and Vision

While there are already many popular libraries such as `openmmlab`, `detectron`, and `huggingface`, they are difficult to expand and transfer between. As a result, research project code bases are isolated from each other and hard to be benchmarked in a unified framework.

We aim to utilize Python native objects as much as possible, design the APIs of abstract base classes, and encourage users to inherit from our base classes for project-specific use cases. We do our best in design to minimize the effort for users in writing subclasses and provide classic examples in different areas such as image classification, object detection, semantic/instance segmentation, domain adaptation, etc.

We maintain flexible API to models defined in existing framework like `mmdetection` and `detectron2`, however, we regularize the training and evaluation pipelines by our datasets, transforms, collate functions, criteria, metrics, and trainer classes.

## 🚀 Enterprise-Grade Type Safety

Pylon sets itself apart with **100% comprehensive type annotations** across the entire codebase - a rarity in the deep learning ecosystem. While most frameworks leave you guessing about tensor shapes and function signatures, Pylon provides crystal-clear type hints that make development a breeze.

### Why This Matters

- **🎯 Catch Errors Before They Run**: Your IDE catches type mismatches instantly, preventing hours of debugging runtime crashes
- **📖 Self-Documenting Code**: Every function signature tells you exactly what it expects and returns - no more diving through documentation
- **⚡ Accelerated Development**: IntelliSense and auto-completion work flawlessly, suggesting the right methods and parameters as you type
- **🔧 Refactor with Confidence**: Type safety ensures your changes don't break existing code - the type checker has your back
- **🏢 Production-Ready**: Enterprise teams can build on Pylon knowing the codebase meets the highest standards of code quality

### What We've Achieved

- **163+ functions** with complete type annotations
- **Every module covered**: From low-level utilities to high-level training loops
- **Complex types handled**: Union types, Generics, TypeVars, and Protocol types where appropriate
- **IDE-first development**: Optimized for PyCharm, VSCode, and other modern development environments

```python
# Example: Crystal-clear function signatures throughout Pylon
def grid_subsample(
    points: torch.Tensor,
    lengths: torch.Tensor,
    voxel_size: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Every function tells you exactly what it needs and what you'll get back."""
    ...
```

This isn't just about following best practices - it's about respecting your time as a researcher and developer. When you build on Pylon, you're building on a foundation of clarity and reliability that lets you focus on what matters: your research.

## Naming

The name of this library is inspired by the protoss structure [Pylon](https://starcraft.fandom.com/wiki/Pylon) from the video game StarCraft II. Pylon serves as a fundamental structure for protoss players as it generates power fields in which other protoss units and structures could be deployed.

## Documentation

All documentation can be found in the [docs](docs/) directory. For environment setup instructions, see [docs/environment_setup.md](docs/environment_setup.md).

## Contributors

* Dayou (Daniel) Mao, {[daniel.mao@uwaterloo.ca](mailto:daniel.mao@uwaterloo.ca)}.

    Cheriton School of Computer Science, University of Waterloo

    Vision and Image Processing Research Group, University of Waterloo
