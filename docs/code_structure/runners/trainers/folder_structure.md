# runners/trainers — folder structure

## Code folder structure

```text
runners/trainers/
├── __init__.py                          # trainer package API surface
├── base_trainer.py                      # BaseTrainer: abstract train/val/test epoch loop, init, checkpointing, resume
├── supervised_single_task_trainer.py    # single-task supervised trainer (one optimizer, scalar loss)
├── supervised_multi_task_trainer.py     # multi-task supervised trainer
├── multi_stage_trainer.py               # multi-stage trainer with continuous epoch numbering
├── multi_val_dataset_trainer.py         # supervised trainer over multiple validation datasets
├── gan_trainers/                        # GAN trainer variants (base, GAN, CSA-CDGAN)
└── pcr_trainers/                        # point-cloud-registration buffer trainer
```

## Tests folder structure

```text
tests/runners/
├── trainers/
│   └── test_interrupt_and_resume.py     # interrupt-and-resume equivalence of SupervisedSingleTaskTrainer
├── test_multi_stage_trainer.py          # MultiStageTrainer vs SupervisedSingleTaskTrainer equivalence
└── test_parallel_trainer.py             # sequential vs parallel validation equivalence
```
