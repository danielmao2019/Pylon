# utils/dtypes — folder structure

## Code folder structure

```text
utils/
└── dtypes.py  # the conceptual dtype universe, the table of the dtype each system stores each conceptual dtype in, the lossless cast every reader, writer and constructor decides through, and the colour range mapping every consumer reads a colour at its own range through
```

## Tests folder structure

```text
tests/utils/
└── test_dtypes.py  # the tables against the widths each system actually carries, the cast that refuses a loss, and the range mapping between colour conventions
```
