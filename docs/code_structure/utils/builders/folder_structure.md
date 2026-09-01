# Utils Builders Folder Structure

## Code folder structure

```text
utils/builders/
├── __init__.py           # builders package API surface
├── builder.py            # the generic {class, args} config builder
└── scheduler_builder.py  # scheduler-specific builder
```

## Tests folder structure

```text
tests/utils/builders/
├── test_builder1.py  # build_from_config tests
└── test_builder2.py  # build_from_config tests
```
