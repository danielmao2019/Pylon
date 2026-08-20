"""chumpy-pickled file I/O.

Reads a file pickled against ``chumpy`` as plain numpy arrays, without
``chumpy`` installed. The substitution of chumpy's own classes is confined to a
``pickle.Unpickler`` subclass, so nothing is planted in the interpreter's module
table and no other load in the process is affected.
"""

import pickle
from typing import Any, Dict, Optional

import numpy as np

# The package whose classes the stream names and this project does not install.
_CHUMPY_ROOT_MODULE = "chumpy"


def load_chumpy(filepath: str) -> Dict[str, Any]:
    """Load a chumpy-pickled file as plain numpy arrays.

    ``chumpy`` is an abandoned Python-2-era package that neither builds nor
    imports against a modern numpy, so the classes its pickled streams name are
    resolved to a stand-in rather than imported.

    Args:
        filepath: Path to the ``.pkl`` file pickled against chumpy.

    Returns:
        The unpickled mapping from entry name to value, each of the file's
        chumpy arrays now a ``numpy.ndarray``.
    """

    def _validate_inputs() -> None:
        assert filepath.endswith(".pkl"), (
            "Expected a .pkl file, the one form this loader reads. " f"{filepath=}"
        )

    _validate_inputs()

    if filepath.endswith(".pkl"):
        return _load_chumpy_from_pkl(filepath=filepath)
    assert 0, "Should not reach here. " f"{filepath=}"


def _load_chumpy_from_pkl(filepath: str) -> Dict[str, Any]:
    """Read the pickle with every chumpy class the stream names resolved to ``_ChumpyArray``, then resolve those stand-ins to the arrays they carry.

    Args:
        filepath: Path to the ``.pkl`` file pickled against chumpy.

    Returns:
        The unpickled mapping from entry name to value, carrying no stand-in
        beyond this function: every ``_ChumpyArray`` is replaced by its
        ``numpy.ndarray``, and every other value is left as the stream produced
        it (a scipy sparse entry stays sparse).
    """
    with open(filepath, "rb") as pickle_file:
        unpickler = _ChumpyUnpickler(pickle_file, encoding="latin1")
        model = unpickler.load()

    for name, value in list(model.items()):
        if isinstance(value, _ChumpyArray):
            model[name] = np.asarray(value)
    return model


class _ChumpyUnpickler(pickle.Unpickler):
    """Unpickler resolving chumpy's own classes to ``_ChumpyArray``, confining the substitution to this load rather than to global state."""

    def find_class(self, module: str, name: str) -> Any:
        """Resolve one class reference the stream names, substituting the stand-in for chumpy's own.

        Args:
            module: Module name the pickled stream carries for the reference.
            name: Class name the pickled stream carries for the reference.

        Returns:
            ``_ChumpyArray`` when the reference names a chumpy class, otherwise
            whatever the base unpickler resolves the reference to.
        """
        if module == _CHUMPY_ROOT_MODULE or module.startswith(
            f"{_CHUMPY_ROOT_MODULE}."
        ):
            return _ChumpyArray
        return super().find_class(module, name)


class _ChumpyArray:
    """Stand-in for one chumpy array while unpickling: it takes the pickled state and yields the numpy array already inside it."""

    def __setstate__(self, state: Any) -> None:
        """Restore the pickled chumpy state onto the stand-in.

        Args:
            state: The pickled ``__dict__`` payload (a dict of attribute name to
                value), or a non-dict payload, which is stored under the
                ``_state`` key.

        Returns:
            None.
        """
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Yield the stored numpy array, the hook that makes the stand-in coercible.

        Args:
            dtype: Numpy dtype to cast the returned array to, or ``None`` to
                yield the stored array as-is.

        Returns:
            The first ``numpy.ndarray`` among the stand-in's stored attributes,
            cast to ``dtype`` when one is given.
        """
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                return value.astype(dtype) if dtype is not None else value
        raise TypeError(
            "The pickled chumpy state carries no numpy array. "
            f"{sorted(self.__dict__.keys())=}"
        )
