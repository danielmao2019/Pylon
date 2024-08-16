from typing import Union
import pytest
from .builder import build_from_config
import copy


class _TestObj:
    pass


class TestObj(_TestObj):

    def __init__(self, attr0: Union[int, _TestObj], attr1: Union[int, _TestObj]) -> None:
        self.attr0 = copy.deepcopy(attr0)
        self.attr1 = copy.deepcopy(attr1)

    def __eq__(self, other) -> bool:
        return (
            type(self) == type(other) and
            self.attr0 == other.attr0 and
            self.attr1 == other.attr1
        )

    def __str__(self) -> str:
        return str({
            'attr0': str(self.attr0),
            'attr1': str(self.attr1),
        })


@pytest.mark.parametrize("config, expected", [
    (
        0, 0,
    ),
    (
        {'class': TestObj, 'args': {'attr0': 0, 'attr1': 1}},
        TestObj(0, 1),
    ),
    (
        {
            'class': TestObj,
            'args': {
                'attr0': {'class': TestObj, 'args': {'attr0': 0, 'attr1': 1}},
                'attr1': 1,
            },
        },
        TestObj(TestObj(0, 1), 1),
    ),
    (
        {
            'class': TestObj,
            'args': {
                'attr0': {'class': TestObj, 'args': {'attr0': 0, 'attr1': 1}},
                'attr1': {'class': TestObj, 'args': {'attr0': 0, 'attr1': 1}},
            },
        },
        TestObj(TestObj(0, 1), TestObj(0, 1)),
    ),
])
def test_build_from_config(config, expected) -> None:
    assert build_from_config(config) == expected
