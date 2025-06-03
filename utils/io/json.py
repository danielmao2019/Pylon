from typing import Any
import os
import json
import jsbeautifier
from utils.ops import apply_tensor_op


def serialize_tensor(obj: Any):
    return apply_tensor_op(func=lambda x: x.detach().tolist(), inputs=obj)


def save_json(obj: Any, filepath: str) -> None:
    assert (
        os.path.dirname(filepath) == "" or
        os.path.isdir(os.path.dirname(filepath))
    ), f"{filepath=}, {os.path.dirname(filepath)=}"
    obj = serialize_tensor(obj)
    try:
        with open(filepath, mode='w') as f:
            f.write(jsbeautifier.beautify(json.dumps(obj), jsbeautifier.default_options()))
    except PermissionError:
        raise PermissionError(f"No write permission for file: {filepath}")
    except OSError as e:
        raise OSError(f"Failed to write to file {filepath}: {str(e)}")
