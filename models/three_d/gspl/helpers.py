from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class HasMeanGetter:
    _mean_name = "means"

    def get_means(self) -> torch.Tensor:
        return self.gaussians[self._mean_name]

    @property
    def means(self) -> torch.Tensor:
        return self.gaussians[self._mean_name]

    @means.setter
    def means(self, v):
        self.gaussians[self._mean_name] = v


class HasScaleGetter(ABC):
    _scale_name = "scales"

    @abstractmethod
    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def scale_inverse_activation(self, scales: torch.Tensor) -> torch.Tensor:
        pass

    def get_scales(self) -> torch.Tensor:
        """Return activated scales"""
        return self.scale_activation(self.scales)

    @property
    def scales(self) -> torch.Tensor:
        """Return raw scales"""
        return self.gaussians[self._scale_name]

    @scales.setter
    def scales(self, v):
        """Set raw scales"""
        self.gaussians[self._scale_name] = v


class HasRotationGetter(ABC):
    _rotation_name = "rotations"

    @abstractmethod
    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def rotation_inverse_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        pass

    def get_rotations(self) -> torch.Tensor:
        """Return activated rotations"""
        return self.rotation_activation(self.rotations)

    @property
    def rotations(self) -> torch.Tensor:
        """Return raw rotations"""
        return self.gaussians[self._rotation_name]

    @rotations.setter
    def rotations(self, v):
        """Set raw rotations"""
        self.gaussians[self._rotation_name] = v


class HasOpacityGetter(ABC):
    _opacity_name = "opacities"

    @abstractmethod
    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        pass

    def get_opacities(self) -> torch.Tensor:
        """Return activated opacities"""
        return self.opacity_activation(self.opacities)

    @property
    def opacities(self) -> torch.Tensor:
        """Return raw opacities"""
        return self.gaussians[self._opacity_name]

    @opacities.setter
    def opacities(self, v):
        """Set raw opacities"""
        self.gaussians[self._opacity_name] = v


class HasSHs(ABC):
    _shs_dc_name = "shs_dc"
    _shs_rest_name = "shs_rest"

    # shs_dc

    def get_shs_dc(self) -> torch.Tensor:
        return self.gaussians[self._shs_dc_name]

    @property
    def shs_dc(self) -> torch.Tensor:
        return self.gaussians[self._shs_dc_name]

    @shs_dc.setter
    def shs_dc(self, v):
        self.gaussians[self._shs_dc_name] = v

    # shs_rest

    def get_shs_rest(self) -> torch.Tensor:
        return self.gaussians[self._shs_rest_name]

    @property
    def shs_rest(self) -> torch.Tensor:
        return self.gaussians[self._shs_rest_name]

    @shs_rest.setter
    def shs_rest(self, v):
        self.gaussians[self._shs_rest_name] = v

    # shs

    def get_shs(self) -> torch.Tensor:
        """
        Return: [n, N_SHs, 3]
        """
        return torch.cat((self.shs_dc, self.shs_rest), dim=1)

    # max_sh_degree

    @abstractmethod
    def get_max_sh_degree(self) -> int:
        raise NotImplementedError()

    @property
    def max_sh_degree(self) -> int:
        return self.get_max_sh_degree()

    # active_sh_degree

    @abstractmethod
    def get_active_sh_degree(self) -> int:
        raise NotImplementedError()

    @property
    def active_sh_degree(self) -> int:
        return self.get_active_sh_degree()

    @abstractmethod
    def set_active_sh_degree(self, v):
        raise NotImplementedError()

    @active_sh_degree.setter
    def active_sh_degree(self, v):
        self.set_active_sh_degree(v)


class HasNewGetters(
    HasMeanGetter,
    HasScaleGetter,
    HasRotationGetter,
    HasOpacityGetter,
    HasSHs,
    ABC,
):
    pass


class HasVanillaGetters(ABC):
    @property
    @abstractmethod
    def get_scaling(self):
        pass

    @property
    @abstractmethod
    def get_rotation(self):
        pass

    @property
    @abstractmethod
    def get_xyz(self):
        pass

    @property
    @abstractmethod
    def get_features(self):
        pass

    @property
    @abstractmethod
    def get_opacity(self):
        pass

    @abstractmethod
    def get_covariance(self, scaling_modifier: float = 1.0):
        pass


class FreezableParameterDict(nn.ParameterDict):

    def __init__(
        self, parameters: Any = None, new_requires_grad: Optional[bool] = None
    ) -> None:
        self.new_requires_grad = new_requires_grad
        super().__init__(parameters)

    def __setitem__(self, key: str, value: Any) -> None:
        # get existing parameter's `requires_grad` state
        current_value = self.get(key, None)
        if current_value is None:
            # if key not exists, use `self.new_requires_grad`
            requires_grad = self.new_requires_grad
            # if `self.new_requires_grad` is None, get from `value`
            if requires_grad is None:
                requires_grad = value.requires_grad
        else:
            requires_grad = current_value.requires_grad

        super().__setitem__(key, value)

        # update `requires_grad` state in-place
        self[key].requires_grad_(requires_grad)


SHS_REST_DIM_TO_DEGREE = {
    0: 0,
    3: 1,
    8: 2,
    15: 3,
}


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def RGB2SH(rgb):
    return (rgb - 0.5) / C0
