from typing import Any


class ConstantLambda:

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, cur_iter: int) -> int:
        return 1
