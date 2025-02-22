import os


def get_work_dir(config_file: str) -> str:
    return os.path.join("./logs", os.path.splitext(os.path.relpath(config_file, start="./configs"))[0])


def get_config(work_dir: str) -> str:
    return os.path.join("./configs", os.path.relpath(work_dir, "./logs")) + ".py"
