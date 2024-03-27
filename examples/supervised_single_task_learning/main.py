import sys
sys.path.append("../..")
from trainers import SupervisedSingleTaskTrainer
from .config import config


if __name__ == "__main__":
    SupervisedSingleTaskTrainer(config).train()
