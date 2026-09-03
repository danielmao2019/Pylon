# runners/trainers — tests structure

## Tests implementation structure

`tests/runners/trainers/test_interrupt_and_resume.py`

```text
test_interrupt_and_resume.py
├── from pathlib import Path
├── from typing import Dict
├── def test_interrupt_and_resume() -> None
│   ├── # Interrupting a run mid-training and re-running resumes from the last finished-epoch checkpoint rather than recomputing it.
│   ├── calls _prepare_workspace
│   ├── calls _build_config
│   └── impls run the trainer twice, asserting the second run resumes from the saved checkpoint
├── def _prepare_workspace(base: Path) -> Dict[str, str]
│   ├── # Prepares the temp workspace directories for the interrupt-and-resume test.
│   └── return  # the workspace path mapping
└── def _build_config(base: Path, work_dir: str) -> dict
    ├── # Builds the SupervisedSingleTaskTrainer config for the interrupt-and-resume test.
    └── return  # the trainer config dict
```
