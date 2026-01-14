"""Shared enumerations and type helpers for manager jobs."""

from enum import Enum, unique
from typing import Dict


@unique
class RunnerKind(str, Enum):
    """Enumerates the known runner categories handled by the manager."""

    TRAINER = "trainer"
    EVALUATOR = "evaluator"
    NERFSTUDIO = "nerfstudio"
    NERFSTUDIO_GENERATION = "nerfstudio_generation"
    DENSE_POINT_CLOUD = "dense_point_cloud"
    SPARSE_POINT_CLOUD = "sparse_point_cloud"
    PIPELINE = "pipeline"

    @classmethod
    def register(cls, name: str, value: str) -> "RunnerKind":
        assert (
            isinstance(name, str) and name.isidentifier() and name.isupper()
        ), "RunnerKind names must be uppercase identifiers"
        assert isinstance(value, str) and value, "RunnerKind values must be non-empty"
        if name in cls.__members__:
            member = cls[name]
            assert (
                member.value == value
            ), f"RunnerKind {name} already registered with value {member.value}"
            return member
        if value in cls._value2member_map_:
            existing = cls._value2member_map_[value]
            assert (
                existing.name == name
            ), f"RunnerKind value {value} already registered as {existing.name}"
            return existing
        member = str.__new__(cls, value)
        member._name_ = name
        member._value_ = value
        cls._member_names_.append(name)
        cls._member_map_[name] = member
        cls._value2member_map_[value] = member
        return member


def register_runner_kind(name: str, value: str) -> RunnerKind:
    return RunnerKind.register(name=name, value=value)


def registered_runner_kinds() -> Dict[str, RunnerKind]:
    return dict(RunnerKind.__members__)
