from typing import Dict, Optional, Tuple, Union

import torch


class PointCloud:

    def __init__(
        self,
        xyz: Optional[torch.Tensor] = None,
        data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        assert xyz is None or isinstance(xyz, torch.Tensor), f"{type(xyz)=}"
        assert data is None or isinstance(
            data, dict
        ), f"Expected dict, got {type(data)}"
        if data is not None:
            assert all(
                isinstance(key, str) for key in data.keys()
            ), f"Non-string point cloud field names: {tuple(data.keys())}"

        if xyz is None:
            assert data is not None
            assert 'xyz' in data, f"{data.keys()=}"
            xyz = data['xyz']
        else:
            assert data is None or (
                'xyz' not in data
            ), "Do not redundantly provide coordinate field when xyz arg is set"

        assert isinstance(xyz, torch.Tensor), f"{type(xyz)=}"
        super().__setattr__('_fields', {})
        super().__setattr__('_length', xyz.shape[0])
        super().__setattr__('_device', xyz.device)
        self._validate_field(name='xyz', value=xyz)
        super().__setattr__('_xyz', xyz)

        if data is not None:
            for key, value in data.items():
                if key == 'xyz':
                    continue
                self._assert_field_name_valid(name=key)
                self._validate_field(name=key, value=value)
                self._fields[key] = value

    def _validate_field(self, name: str, value: torch.Tensor) -> None:
        assert isinstance(
            value, torch.Tensor
        ), f"{name} must be torch.Tensor, got {type(value)}"
        assert value.ndim >= 1, f"{name} must be at least 1D, got shape {value.shape}"
        assert (
            value.shape[0] > 0
        ), f"{name} must have at least one point, got {value.shape[0]}"
        assert (
            value.shape[0] == self._length
        ), f"{name} length mismatch: {value.shape=}, expected {self._length}"
        assert (
            value.device == self._device
        ), f"{name} device mismatch: {value.device=} vs {self._device=}"
        if name == 'xyz':
            self.validate_xyz_tensor(value)
        elif name == 'rgb':
            self.validate_rgb_tensor(value)
        elif name == 'indices':
            assert value.dtype == torch.int64, f"{value.dtype=}"

    @staticmethod
    def validate_xyz_tensor(xyz: torch.Tensor) -> None:
        assert isinstance(xyz, torch.Tensor), f"{type(xyz)=}"
        assert xyz.ndim == 2, f"{xyz.shape=}"
        assert xyz.shape[1] == 3, f"{xyz.shape=}"
        assert xyz.is_floating_point(), f"{xyz.dtype=}"
        assert not torch.isnan(xyz).any(), "xyz tensor contains NaN"
        assert not torch.isinf(xyz).any(), "xyz tensor contains Inf"

    @staticmethod
    def validate_rgb_tensor(rgb: torch.Tensor) -> None:
        assert isinstance(rgb, torch.Tensor), f"{type(rgb)=}"
        assert rgb.ndim == 2, f"{rgb.shape=}"
        assert rgb.shape[1] == 3, f"{rgb.shape=}"
        assert not torch.isnan(rgb).any(), "rgb tensor contains NaN"
        assert not torch.isinf(rgb).any(), "rgb tensor contains Inf"

    def _assert_field_name_valid(self, name: str) -> None:
        assert isinstance(name, str), f"{type(name)=}"
        assert not name.startswith(
            '_'
        ), f"Field names cannot start with underscore: {name}"
        assert name not in (
            'device',
            'num_points',
            'field_names',
        ), f"Reserved attribute name: {name}"

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_points(self) -> int:
        return self._length

    def __len__(self) -> int:
        return self._length

    def field_names(self) -> Tuple[str, ...]:
        return ('xyz',) + tuple(self._fields.keys())

    def __getattr__(self, name: str) -> torch.Tensor:
        assert all(
            key in self.__dict__ for key in ('_xyz', '_fields', '_length', '_device')
        ), "PointCloud accessed before initialization"
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: torch.Tensor) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        self._assert_field_name_valid(name=name)
        self._validate_field(name=name, value=value)
        if name == 'xyz':
            super().__setattr__('_xyz', value)
        else:
            self._fields[name] = value

    def __getstate__(self) -> dict:
        return {
            '_xyz': self._xyz,
            '_fields': self._fields,
            '_length': self._length,
            '_device': self._device,
        }

    def __setstate__(self, state: dict) -> None:
        # Input validations
        assert isinstance(state, dict), f"{type(state)=}"
        assert '_xyz' in state, f"{state.keys()=}"
        assert '_fields' in state, f"{state.keys()=}"
        assert '_length' in state, f"{state.keys()=}"
        assert '_device' in state, f"{state.keys()=}"

        super().__setattr__('_xyz', state['_xyz'])
        super().__setattr__('_fields', state['_fields'])
        super().__setattr__('_length', state['_length'])
        super().__setattr__('_device', state['_device'])
