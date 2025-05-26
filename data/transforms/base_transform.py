from typing import Any


class BaseTransform:

    def __call__(self, *args) -> Any:
        r"""This method implements the default __call__ method for concrete classes. It assumes that
        `_call_single_` has been implemented and applies `_call_single_` on each of the input arguments.
        """
        assert hasattr(self, '_call_single_')
        result = [self._call_single_(arg) for arg in args]
        if len(result) == 1:
            return result[0]
        else:
            return result
