import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement

from models.three_d.gspl.helpers import (
    FreezableParameterDict,
    HasNewGetters,
    HasVanillaGetters,
    build_scaling_rotation,
    eval_sh,
    inverse_sigmoid,
    strip_symmetric,
)


class BaseGSPLModel(nn.Module, ABC):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gaussians = self.setup_gaussians_container()

    @staticmethod
    def setup_gaussians_container():
        return nn.ParameterDict()

    @abstractmethod
    def get_property_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def property_names(self) -> Tuple[str, ...]:
        return self.get_property_names()

    def get_property(self, name: str) -> torch.Tensor:
        """Get single raw property"""
        return self.gaussians[name]

    def get_properties(self) -> Dict[str, torch.Tensor]:
        """Get all raw properties as a dict"""
        return {name: self.gaussians[name] for name in self.property_names}

    def set_property(self, name: str, value: torch.Tensor):
        """Set single raw property"""
        self.gaussians[name] = value

    def set_properties(self, properties: Dict[str, torch.Tensor]):
        """
        Set all raw properties.
        This setter will not update optimizers.
        """

        for name in self.property_names:
            self.gaussians[name] = properties[name]

    def update_properties(
        self, properties: Dict[str, torch.Tensor], strict: bool = True
    ):
        """
        Replace part of the properties by those provided in `properties`
        """
        for name in properties:
            if name not in self.gaussians and strict is True:
                raise RuntimeError("`{}` is not a property".format(name))
            self.gaussians[name] = properties[name]

    @property
    def properties(self) -> Dict[str, torch.Tensor]:
        return self.get_properties()

    @properties.setter
    def properties(self, properties: Dict[str, torch.Tensor]):
        """
        this setter will not update optimizers
        """

        self.set_properties(properties)

    def get_n_gaussians(self) -> int:
        return self.gaussians[next(iter(self.gaussians))].shape[0]

    @property
    def n_gaussians(self) -> int:
        return self.get_n_gaussians()

    def freeze(self):
        self.gaussians = FreezableParameterDict(self.gaussians, new_requires_grad=False)

    @abstractmethod
    def setup_from_number(self, n: int, *args, **kwargs):
        pass

    @abstractmethod
    def setup_from_tensors(self, tensors: Dict[str, torch.Tensor], *args, **kwargs):
        pass


class InstantiatableConfig:
    def instantiate(self, *args, **kwargs) -> Any:
        pass


class Gaussian(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> BaseGSPLModel:
        raise NotImplementedError()


@dataclass
class VanillaGaussian(Gaussian):

    sh_degree: int = 3

    def instantiate(self, *args, **kwargs) -> "GSPLModel":
        return GSPLModel(self)


class GSPLModel(
    BaseGSPLModel,
    HasNewGetters,
    HasVanillaGetters,
):
    def __init__(self, config: VanillaGaussian) -> None:
        super().__init__()
        self.config = config

        names = [
            "means",
            "shs_dc",
            "shs_rest",
            "opacities",
            "scales",
            "rotations",
        ] + self.get_extra_property_names()
        self._names = tuple(names)

        self.is_pre_activated = False

        # TODO: is it suitable to place `active_sh_degree` in gaussian model?
        self.register_buffer(
            "_active_sh_degree", torch.tensor(0, dtype=torch.uint8), persistent=True
        )

    def get_extra_property_names(self):
        return []

    def before_setup_set_properties_from_number(
        self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs
    ):
        pass

    def setup_from_number(self, n: int, *args, **kwargs):
        means = torch.zeros((n, 3))
        shs = torch.zeros((n, 3, (self.max_sh_degree + 1) ** 2))
        shs_dc = shs[:, :, 0:1].transpose(1, 2).contiguous()
        shs_rest = shs[:, :, 1:].transpose(1, 2).contiguous()
        scales = torch.zeros((n, 3))
        rotations = torch.zeros((n, 4))
        opacities = torch.zeros((n, 1))

        means = nn.Parameter(means.requires_grad_(True))
        shs_dc = nn.Parameter(shs_dc.requires_grad_(True))
        shs_rest = nn.Parameter(shs_rest.requires_grad_(True))
        scales = nn.Parameter(scales.requires_grad_(True))
        rotations = nn.Parameter(rotations.requires_grad_(True))
        opacities = nn.Parameter(opacities.requires_grad_(True))

        property_dict = {
            "means": means,
            "shs_dc": shs_dc,
            "shs_rest": shs_rest,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }
        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        self.set_properties(property_dict)

        self.active_sh_degree = 0

    def setup_from_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        active_sh_degree: int = -1,
        *args,
        **kwargs,
    ):
        """
        Args:
            tensors
            active_sh_degree: -1 means use maximum sh_degree
            *args,
            **kwargs
        """

        # detect sh_degree
        if "shs_rest" in tensors:
            shs_rest_dims = tensors["shs_rest"].shape[1]
            sh_degree = -1
            for i in range(4):
                if shs_rest_dims == (i + 1) ** 2 - 1:
                    sh_degree = i
                    break
            assert sh_degree >= 0, f"can not get sh_degree from `shs_rest`"
        else:
            sh_degree = self.config.sh_degree

        # update sh_degree
        # self.config.sh_degree = sh_degree

        # TODO: may be should enable changing sh_degree
        # validate sh_degree
        assert self.config.sh_degree == sh_degree, "sh_degree not match"

        # initialize by number
        n_gaussians = tensors[list(tensors.keys())[0]].shape[0]
        self.setup_from_number(n_gaussians)

        unused_properties = list(tensors.keys())
        unmet_properties = list(self.property_names)

        # copy from tensor
        property_names = self.property_names

        with torch.no_grad():
            for i in tensors:
                if i not in property_names:
                    continue
                self.get_property(i).copy_(tensors[i])

                unused_properties.remove(i)
                unmet_properties.remove(i)

        if active_sh_degree == -1:
            active_sh_degree = sh_degree
        self.active_sh_degree = min(active_sh_degree, sh_degree)

        return unused_properties, unmet_properties

    def get_property_names(self) -> Tuple[str, ...]:
        return self._names

    def get_max_sh_degree(self) -> int:
        return self.config.sh_degree

    @property
    def max_sh_degree(self) -> int:
        return self.config.sh_degree

    def get_active_sh_degree(self) -> int:
        return self._active_sh_degree.item()

    def set_active_sh_degree(self, v):
        self._active_sh_degree.fill_(v)

    @property
    def active_sh_degree(self) -> int:
        return self._active_sh_degree.item()

    @active_sh_degree.setter
    def active_sh_degree(self, v: int):
        self._active_sh_degree.fill_(v)

    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(opacities)

    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return inverse_sigmoid(opacities)

    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.exp(scales)

    def scale_inverse_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.log(scales)

    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(rotations)

    def rotation_inverse_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return rotations

    @staticmethod
    def _return_as_is(v):
        return v

    def _get_shs_from_dict(self) -> torch.Tensor:
        return self.gaussians["shs"]

    def pre_activate_all_properties(self):
        self.is_pre_activated = True

        self.scales = self.get_scales()
        self.rotations = self.get_rotations()
        self.opacities = self.get_opacities()

        # concat `shs_dc` and `shs_rest` and store it to dict, then remove `shs_dc` and `shs_rest`
        names = list(self._names)
        ## concat
        self.gaussians["shs"] = self.get_shs()
        names.append("shs")
        ## remove `shs_dc`
        del self.gaussians["shs_dc"]
        names.remove("shs_dc")
        ## remove `shs_rest`
        del self.gaussians["shs_rest"]
        names.remove("shs_rest")
        ## replace `get_shs` interface
        self.get_shs = self._get_shs_from_dict
        ## replace `names`
        self._names = tuple(names)

        self.scale_activation = self._return_as_is
        self.scale_inverse_activation = self._return_as_is
        self.rotation_activation = self._return_as_is
        self.rotation_inverse_activation = self._return_as_is
        self.opacity_activation = self._return_as_is
        self.opacity_inverse_activation = self._return_as_is

    def get_non_pre_activated_properties(self):
        if self.is_pre_activated is True:
            activated_properties = self.properties
            keys = list(activated_properties.keys())
            non_pre_activated_properties = {}
            non_pre_activated_properties["scales"] = torch.log(
                activated_properties["scales"]
            )
            keys.remove("scales")
            non_pre_activated_properties["opacities"] = inverse_sigmoid(
                activated_properties["opacities"]
            )
            keys.remove("opacities")
            non_pre_activated_properties["shs_dc"] = activated_properties["shs"][
                :, :1, :
            ]
            non_pre_activated_properties["shs_rest"] = activated_properties["shs"][
                :, 1:, :
            ]
            keys.remove("shs")

            for key in keys:
                non_pre_activated_properties[key] = activated_properties[key]

            return non_pre_activated_properties
        else:
            return self.properties

    # below getters are declared for the compatibility purpose

    @property
    def get_scaling(self):
        return self.scale_activation(self.gaussians["scales"])

    @property
    def get_rotation(self):
        return self.rotation_activation(self.gaussians["rotations"])

    @property
    def get_xyz(self):
        return self.gaussians["means"]

    @property
    def get_features(self):
        return self.get_shs()

    @property
    def get_opacity(self):
        return self.opacity_activation(self.gaussians["opacities"])

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self.build_covariance_from_scaling_rotation(
            self.get_scales(),
            scaling_modifier,
            self.get_rotations(),
        )


@dataclass
class GaussianPlyUtils:
    """
    Load parameters from ply;
    Save to ply;
    """

    sh_degrees: int
    xyz: Union[np.ndarray, torch.Tensor]  # [n, 3]
    opacities: Union[np.ndarray, torch.Tensor]  # [n, 1]
    features_dc: Union[np.ndarray, torch.Tensor]  # ndarray[n, 3, 1], or tensor[n, 1, 3]
    features_rest: Union[
        np.ndarray, torch.Tensor
    ]  # ndarray[n, 3, 15], or tensor[n, 15, 3]; NOTE: this is features_rest actually!
    scales: Union[np.ndarray, torch.Tensor]  # [n, 3]
    rotations: Union[np.ndarray, torch.Tensor]  # [n, 4]

    @staticmethod
    def detect_sh_degree_from_shs_rest(shs_rest: torch.Tensor):
        assert isinstance(shs_rest, torch.Tensor)
        return SHS_REST_DIM_TO_DEGREE[shs_rest.shape[-2]]

    @staticmethod
    def load_array_from_plyelement(plyelement, name_prefix: str, required: bool = True):
        names = [
            p.name for p in plyelement.properties if p.name.startswith(name_prefix)
        ]
        if len(names) == 0:
            if required is True:
                raise RuntimeError(f"'{name_prefix}' not found in ply")
            return np.empty((plyelement["x"].shape[0], 0))
        names = sorted(names, key=lambda x: int(x.split('_')[-1]))
        v_list = []
        for idx, attr_name in enumerate(names):
            v_list.append(np.asarray(plyelement[attr_name]))

        return np.stack(v_list, axis=1)

    @classmethod
    def load_from_ply(cls, path: str, sh_degrees: int = -1):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        features_rest = cls.load_array_from_plyelement(
            plydata.elements[0], "f_rest_", required=False
        ).reshape((xyz.shape[0], 3, -1))
        if sh_degrees >= 0:
            assert (
                features_rest.shape[-1] == (sh_degrees + 1) ** 2 - 1
            )  # TODO: remove such a assertion
        else:
            # auto determine sh_degrees
            features_rest_dims = features_rest.shape[-1]
            for i in range(4):
                if features_rest_dims == (i + 1) ** 2 - 1:
                    sh_degrees = i
                    break
            assert sh_degrees >= 0, f"invalid sh_degrees={sh_degrees}"

        scales = cls.load_array_from_plyelement(plydata.elements[0], "scale_")
        rots = cls.load_array_from_plyelement(plydata.elements[0], "rot_")

        return cls(
            sh_degrees=sh_degrees,
            xyz=xyz,
            opacities=opacities,
            features_dc=features_dc,
            features_rest=features_rest,
            scales=scales,
            rotations=rots,
        )

    @classmethod
    def load_from_model_properties(cls, properties, sh_degree: int = -1):
        if sh_degree < 0:
            sh_degree = cls.detect_sh_degree_from_shs_rest(properties["shs_rest"])

        init_args = {
            "sh_degrees": sh_degree,
        }

        for name_in_model, name_in_dataclass in [
            ("means", "xyz"),
            ("shs_dc", "features_dc"),
            ("shs_rest", "features_rest"),
            ("scales", "scales"),
            ("rotations", "rotations"),
            ("opacities", "opacities"),
        ]:
            init_args[name_in_dataclass] = properties[name_in_model].detach()

        return cls(**init_args)

    @classmethod
    def load_from_model(cls, model):
        return cls.load_from_model_properties(
            model.properties, sh_degree=model.max_sh_degree
        )

    @classmethod
    def load_from_state_dict(cls, state_dict):
        if "gaussian_model.gaussians.means" in state_dict:
            return cls.load_from_new_state_dict(state_dict)
        return cls.load_from_old_state_dict(state_dict)

    @classmethod
    def load_from_new_state_dict(cls, state_dict):
        prefix = "gaussian_model.gaussians."

        init_args = {
            "sh_degrees": cls.detect_sh_degree_from_shs_rest(
                state_dict["{}shs_rest".format(prefix)]
            ),
        }

        for name_in_dict, name_in_dataclass in [
            ("means", "xyz"),
            ("shs_dc", "features_dc"),
            ("shs_rest", "features_rest"),
            ("scales", "scales"),
            ("rotations", "rotations"),
            ("opacities", "opacities"),
        ]:
            init_args[name_in_dataclass] = state_dict[
                "{}{}".format(prefix, name_in_dict)
            ]

        return cls(**init_args)

    @classmethod
    def load_from_old_state_dict(cls, state_dict):
        key_prefix = "gaussian_model._"

        init_args = {
            "sh_degrees": cls.detect_sh_degree_from_shs_rest(
                state_dict["{}features_rest".format(key_prefix)]
            ),
        }
        for name_in_dict, name_in_dataclass in [
            ("xyz", "xyz"),
            ("features_dc", "features_dc"),
            ("features_rest", "features_rest"),
            ("scaling", "scales"),
            ("rotation", "rotations"),
            ("opacity", "opacities"),
        ]:
            init_args[name_in_dataclass] = state_dict[
                "{}{}".format(key_prefix, name_in_dict)
            ]

        return cls(**init_args)

    def to_parameter_structure(self):
        assert isinstance(self.xyz, np.ndarray) is True
        return GaussianPlyUtils(
            sh_degrees=self.sh_degrees,
            xyz=torch.tensor(self.xyz, dtype=torch.float),
            opacities=torch.tensor(self.opacities, dtype=torch.float),
            features_dc=torch.tensor(self.features_dc, dtype=torch.float).transpose(
                1, 2
            ),
            features_rest=torch.tensor(self.features_rest, dtype=torch.float).transpose(
                1, 2
            ),
            scales=torch.tensor(self.scales, dtype=torch.float),
            rotations=torch.tensor(self.rotations, dtype=torch.float),
        )

    @torch.no_grad()
    def to_ply_format(self):
        assert isinstance(self.xyz, torch.Tensor) is True
        return GaussianPlyUtils(
            sh_degrees=self.sh_degrees,
            xyz=self.xyz.cpu().numpy(),
            opacities=self.opacities.cpu().numpy(),
            features_dc=self.features_dc.transpose(1, 2).cpu().numpy(),
            features_rest=self.features_rest.transpose(1, 2).cpu().numpy(),
            scales=self.scales.cpu().numpy(),
            rotations=self.rotations.cpu().numpy(),
        )

    def save_to_ply(self, path: str, with_colors: bool = False):
        assert isinstance(self.xyz, np.ndarray) is True

        gaussian = self

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except:
            pass

        xyz = gaussian.xyz
        f_dc = gaussian.features_dc.reshape((gaussian.features_dc.shape[0], -1))
        # TODO: change sh degree
        if gaussian.sh_degrees > 0:
            f_rest = gaussian.features_rest.reshape(
                (gaussian.features_rest.shape[0], -1)
            )
        else:
            f_rest = np.zeros((f_dc.shape[0], 0))
        opacities = gaussian.opacities
        scale = gaussian.scales
        rotation = gaussian.rotations

        # xyz
        dtype_full = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ]
        attribute_list = [
            ("x", xyz[..., 0]),
            ("y", xyz[..., 1]),
            ("z", xyz[..., 2]),
        ]

        def add_attribute(name_prefix, value):
            for i in range(value.shape[-1]):
                name = "{}_{}".format(name_prefix, i)
                dtype_full.append((name, "f4"))
                attribute_list.append((name, value[..., i]))

        # shs_dc
        add_attribute("f_dc", f_dc)
        # shs_rest
        add_attribute("f_rest", f_rest)
        # opacities
        dtype_full.append(("opacity", "f4"))
        attribute_list.append(("opacity", opacities.squeeze(-1)))
        # scales
        add_attribute("scale", scale)
        # rotations
        add_attribute("rot", rotation)

        if with_colors is True:
            rgbs = np.clip((eval_sh(0, self.features_dc, None) + 0.5), 0.0, 1.0)
            rgbs = (rgbs * 255).astype(np.uint8)

            dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            attribute_list += [
                ("red", rgbs[..., 0]),
                ("green", rgbs[..., 1]),
                ("blue", rgbs[..., 2]),
            ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        for k, v in attribute_list:
            elements[k] = v
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
