import os


def check_read_file(path: str) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isfile(path), f"{path=}"
    return path


def check_write_file(path: str) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isdir(os.path.dirname(path)), f"{path=}"
    return path


def check_read_dir(path: str) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isdir(path), f"{path=}"
    return path


def check_write_dir(path: str) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isdir(path), f"{path=}"
    return path
