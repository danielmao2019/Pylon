# criteria — folder structure

## Code folder structure

```text
criteria/
├── __init__.py        # exposes BaseCriterion and the criteria subpackages
├── base_criterion.py  # BaseCriterion: the criterion root — an async thread-backed loss buffer plus the abstract __call__ / summarize contract
├── common/            # domain-agnostic losses
├── diffusion/         # diffusion-model criteria
├── vision_2d/         # 2D vision criteria
├── vision_3d/         # 3D vision criteria
└── wrappers/          # single-task, multi-task, and third-party criterion wrappers
```

## Tests folder structure

```text
tests/criteria/
├── base_criterion/
│   └── test_base_criterion_async_buffer.py  # BaseCriterion: async buffer throughput, thread safety, the add_to_buffer assertions, and the disabled-buffer path.
├── common/     # tests of the domain-agnostic losses
├── vision_2d/  # tests of the 2D vision criteria
├── vision_3d/  # tests of the 3D vision criteria
└── wrappers/   # tests of the criterion wrappers
```
