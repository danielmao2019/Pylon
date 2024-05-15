from typing import List, Union
import torch


class BaseTransform:

    def __call__(self, *args) -> Union[torch.Tensor, List[torch.Tensor]]:
        r"""This method implements the default __call__ method for concrete classes. It assumes that
        `_call_single_` has been implemented and applies `_call_single_` on each of the input arguments.
        """
        assert hasattr(self, '_call_single_')
        assert all(type(arg) == torch.Tensor for arg in args)
        result = [self._call_single_(arg) for arg in args]
        assert all(type(elem) == torch.Tensor for elem in result)
        if len(result) == 1:
            return result[0]
        else:
            return result
