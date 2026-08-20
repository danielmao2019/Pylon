import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np
import pytest

from utils.io.chumpy import load_chumpy


class Ch:
    """Stands in for chumpy's own array class while a test pickle is written."""

    def __init__(self, x: Any) -> None:
        """Store the payload the pickled state will carry.

        Args:
            x: The value pickled as this instance's only attribute — a
                ``numpy.ndarray`` for a chumpy array, anything else for a
                payload carrying no array.

        Returns:
            None.
        """
        self.x = x


class Plain:
    """An ordinary class whose instances the loader must leave untouched."""

    def __init__(self, value: int) -> None:
        """Store the payload the pickled state will carry.

        Args:
            value: The value pickled as this instance's only attribute.

        Returns:
            None.
        """
        self.value = value


def _write_pickle_naming_chumpy(payload: Dict[str, Any], filepath: Path) -> None:
    """Write a pickle of payload whose ``Ch`` references name ``chumpy.ch.Ch``.

    Protocol 2 writes a class reference as the newline-delimited ``GLOBAL``
    opcode, so rewriting the module name in the serialized bytes produces a
    stream naming a class that cannot be imported — exactly what a chumpy pickle
    is in this environment.

    Args:
        payload: Mapping to pickle; every ``Ch`` instance in it ends up naming
            chumpy's own class in the written stream.
        filepath: Path of the ``.pkl`` file to write.

    Returns:
        None.
    """
    stream = pickle.dumps(payload, protocol=2)
    stream = stream.replace(f"c{Ch.__module__}\nCh\n".encode(), b"cchumpy.ch\nCh\n")
    filepath.write_bytes(stream)


def _chumpy_module_keys() -> Set[str]:
    """Collect the ``sys.modules`` keys naming the chumpy package.

    Args:
        None.

    Returns:
        The set of module names that are ``chumpy`` itself or one of its
        submodules.
    """
    return {key for key in sys.modules if key == "chumpy" or key.startswith("chumpy.")}


def test_a_chumpy_pickled_array_loads_as_the_array_it_carries(tmp_path: Path) -> None:
    """A file naming a chumpy class loads without chumpy installed, as the numpy array the pickled state carries.

    Args:
        tmp_path: pytest-provided directory the test pickle is written into.

    Returns:
        None.
    """
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    filepath = tmp_path / "model.pkl"
    _write_pickle_naming_chumpy(payload={"v_template": Ch(array)}, filepath=filepath)

    loaded = load_chumpy(filepath=str(filepath))

    assert isinstance(
        loaded["v_template"], np.ndarray
    ), f"{type(loaded['v_template'])=}"
    assert np.array_equal(loaded["v_template"], array), f"{loaded['v_template']=}"


def test_a_non_chumpy_value_survives_untouched(tmp_path: Path) -> None:
    """Only chumpy's own references are substituted, so every other value comes back as the stream produced it.

    Args:
        tmp_path: pytest-provided directory the test pickle is written into.

    Returns:
        None.
    """
    filepath = tmp_path / "model.pkl"
    _write_pickle_naming_chumpy(
        payload={"v_template": Ch(np.zeros(3)), "meta": Plain(7)}, filepath=filepath
    )

    loaded = load_chumpy(filepath=str(filepath))

    assert isinstance(loaded["meta"], Plain), f"{type(loaded['meta'])=}"
    assert loaded["meta"].value == 7, f"{loaded['meta'].value=}"


def test_the_load_leaves_sys_modules_untouched(tmp_path: Path) -> None:
    """The substitution is confined to the unpickler, so no chumpy entry is planted in the interpreter's module table.

    Args:
        tmp_path: pytest-provided directory the test pickle is written into.

    Returns:
        None.
    """
    filepath = tmp_path / "model.pkl"
    _write_pickle_naming_chumpy(
        payload={"v_template": Ch(np.zeros(3))}, filepath=filepath
    )
    keys_before = _chumpy_module_keys()

    load_chumpy(filepath=str(filepath))

    assert (
        _chumpy_module_keys() == keys_before
    ), f"{_chumpy_module_keys()=} {keys_before=}"


def test_a_chumpy_payload_carrying_no_array_is_rejected(tmp_path: Path) -> None:
    """A chumpy value whose state holds no array raises rather than resolving to something invented.

    Args:
        tmp_path: pytest-provided directory the test pickle is written into.

    Returns:
        None.
    """
    filepath = tmp_path / "model.pkl"
    _write_pickle_naming_chumpy(
        payload={"v_template": Ch("no array here")}, filepath=filepath
    )

    with pytest.raises(TypeError):
        load_chumpy(filepath=str(filepath))
