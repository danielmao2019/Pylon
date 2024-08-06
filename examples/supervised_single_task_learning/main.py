import sys
sys.path.append("../..")
from runners import SupervisedSingleTaskTrainer
from .config import config


if __name__ == "__main__":
    SupervisedSingleTaskTrainer(config).train()
