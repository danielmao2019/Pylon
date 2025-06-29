class WarmupLambda:

    def __init__(self, steps: int) -> None:
        assert type(steps) == int, f"{type(steps)=}"
        assert steps > 0
        self.steps = steps

    def __call__(self, cur_iter: int) -> float:
        return min(cur_iter / self.steps, 1)
