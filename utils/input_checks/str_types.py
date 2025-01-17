from typing import List, Union, Any, Optional
import os


def check_read_file(
    path: Any,
    ext: Optional[Union[str, List[str]]] = None,
) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isfile(path), f"{path=}"
    if ext is not None:
        if type(ext) == str:
            ext = [ext]
        assert type(ext) == list, f"{type(ext)=}"
        assert all([type(elem) == str for elem in ext]), f"{ext=}"
        assert any([path.endswith(e) for e in ext]), f"{path=}, {ext=}"
    return path


def check_write_file(path: Any) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isdir(os.path.dirname(path)), f"{path=}"
    return path


def check_read_dir(path: Any) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isdir(path), f"{path=}"
    return path


def check_write_dir(path: Any) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isdir(path), f"{path=}"
    return path
