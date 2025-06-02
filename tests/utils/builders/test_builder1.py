from typing import Union, Optional
import pytest
import copy
from utils.builders.builder import build_from_config


class _DummyObj:
    pass


class DummyObj(_DummyObj):

    def __init__(self, attr0: Union[int, _DummyObj], attr1: Union[int, _DummyObj], attr2: Optional[Union[int, _DummyObj]] = 0) -> None:
        self.attr0 = copy.deepcopy(attr0)
        self.attr1 = copy.deepcopy(attr1)
        self.attr2 = copy.deepcopy(attr2)

    def __eq__(self, other) -> bool:
        return (
            type(self) == type(other) and
            self.attr0 == other.attr0 and
            self.attr1 == other.attr1 and
            self.attr2 == other.attr2
        )

    def __str__(self) -> str:
        return str({
            'attr0': str(self.attr0),
            'attr1': str(self.attr1),
            'attr2': str(self.attr2),
        })


@pytest.mark.parametrize("config, kwargs, expected", [
    (
        0, {}, 0,
    ),
    (
        {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
        {},
        DummyObj(0, 1),
    ),
    (
        {
            'class': DummyObj,
            'args': {
                'attr0': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
                'attr1': 1,
            },
        },
        {},
        DummyObj(DummyObj(0, 1), 1),
    ),
    (
        {
            'class': DummyObj,
            'args': {
                'attr0': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
                'attr1': DummyObj(0, 1),
            },
        },
        {},
        DummyObj(DummyObj(0, 1), DummyObj(0, 1)),
    ),
    (
        {
            'class': DummyObj,
            'args': {
                'attr0': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
                'attr1': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
            },
        },
        {},
        DummyObj(DummyObj(0, 1), DummyObj(0, 1)),
    ),
    (
        {
            'class': DummyObj,
            'args': {
                'attr0': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
                'attr1': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
            },
        },
        {'attr2': 2},
        DummyObj(DummyObj(0, 1), DummyObj(0, 1), 2),
    ),
    (
        {
            'class': DummyObj,
            'args': {
                'attr0': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
                'attr1': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
            },
        },
        {'attr2': DummyObj(0, 1)},
        DummyObj(DummyObj(0, 1), DummyObj(0, 1), DummyObj(0, 1)),
    ),
    (
        {
            'class': DummyObj,
            'args': {
                'attr0': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
                'attr1': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}},
            },
        },
        {'attr2': {'class': DummyObj, 'args': {'attr0': 0, 'attr1': 1}}},
        DummyObj(DummyObj(0, 1), DummyObj(0, 1), DummyObj(0, 1)),
    ),
])
def test_build_from_config(config, kwargs, expected) -> None:
    assert build_from_config(config, **kwargs) == expected
