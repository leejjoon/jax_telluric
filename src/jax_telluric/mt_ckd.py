"""JAX implementation of the MT_CKD water-vapor continuum."""

# Copyright © Atmospheric and Environmental Research, Inc., 2022.
# All rights reserved. This implementation is derived from the MT_CKD module
# distributed with LBLRTM. AER grants use, copying, modification, and
# redistribution for scientific and research purposes with this notice and
# appropriate acknowledgment. It may not be incorporated into proprietary or
# commercial software without AER's express written consent. It is provided
# without express or implied warranties. General reference: Mlawer et al.
# (2012), doi:10.1098/rsta.2011.0295. Questions: aer_contnm@aer.com.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import jax.numpy as jnp
import numpy as np

from .types import AtmosphereProfile


_RADIATION_CONSTANT_K_CM = 1.4387752


def _cubic_stencils(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the four-point interpolation indices and LBLRTM XINT weights."""

    spacing = np.diff(source)
    if not np.allclose(spacing, spacing[0], rtol=1.0e-12, atol=0.0):
        raise ValueError("MT_CKD coefficient wavenumbers must be evenly spaced")
    position = (target - source[0]) / spacing[0]
    # XINT adds ONEPL=1.001 before truncating its one-based source index.
    lower = np.floor(position + 0.001).astype(int)
    if np.any(lower < 1) or np.any(lower + 2 >= source.size):
        raise ValueError(
            "model wavenumbers need one MT_CKD coefficient point of padding on each side"
        )
    fraction = position - lower
    c = (3.0 - 2.0 * fraction) * fraction**2
    b = 0.5 * fraction * (1.0 - fraction)
    b1 = b * (1.0 - fraction)
    b2 = b * fraction
    weights = np.stack((-b1, 1.0 - c + b2, c + b1, -b2), axis=1)
    offsets = np.arange(-1, 3)
    return lower[:, None] + offsets[None, :], weights


@dataclass(frozen=True)
class MTCKDWaterContinuum:
    """Runtime MT_CKD H2O self and foreign continuum for one spectral grid.

    The input arrays are the reference quantities distributed in AER's
    ``absco-ref_wv-mt-ckd.nc`` file.  ``from_netcdf`` is the usual constructor.
    Optical depth is evaluated from the model pressure, temperature, H2O VMR,
    and molecular column, so the backend can be reused across profiles.
    """

    wavenumber_cm1: np.ndarray
    coefficient_wavenumber_cm1: np.ndarray
    self_absco_ref: np.ndarray
    foreign_absco_ref: np.ndarray
    self_temperature_exponent: np.ndarray
    reference_pressure_hpa: float = 1013.0
    reference_temperature_k: float = 296.0
    version: str = "unknown"
    _indices: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        target = np.asarray(self.wavenumber_cm1, dtype=float)
        source = np.asarray(self.coefficient_wavenumber_cm1, dtype=float)
        self_ref = np.asarray(self.self_absco_ref, dtype=float)
        foreign_ref = np.asarray(self.foreign_absco_ref, dtype=float)
        exponent = np.asarray(self.self_temperature_exponent, dtype=float)
        if target.ndim != 1 or target.size < 2 or np.any(np.diff(target) <= 0.0):
            raise ValueError("model wavenumbers must be a strictly increasing array")
        if source.ndim != 1 or source.size < 4 or np.any(np.diff(source) <= 0.0):
            raise ValueError("MT_CKD coefficient wavenumbers must be strictly increasing")
        if any(array.shape != source.shape for array in (self_ref, foreign_ref, exponent)):
            raise ValueError("MT_CKD coefficient arrays must match the coefficient grid")
        arrays = (target, source, self_ref, foreign_ref, exponent)
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("MT_CKD arrays must be finite")
        if np.any(self_ref < 0.0) or np.any(foreign_ref < 0.0):
            raise ValueError("MT_CKD reference coefficients must be nonnegative")
        if self.reference_pressure_hpa <= 0.0 or self.reference_temperature_k <= 0.0:
            raise ValueError("MT_CKD reference pressure and temperature must be positive")
        indices, weights = _cubic_stencils(source, target)
        object.__setattr__(self, "wavenumber_cm1", target)
        object.__setattr__(self, "coefficient_wavenumber_cm1", source)
        object.__setattr__(self, "self_absco_ref", self_ref)
        object.__setattr__(self, "foreign_absco_ref", foreign_ref)
        object.__setattr__(self, "self_temperature_exponent", exponent)
        object.__setattr__(self, "_indices", indices)
        object.__setattr__(self, "_weights", weights)

    @classmethod
    def from_netcdf(
        cls, path: str | Path, wavenumber_cm1: np.ndarray
    ) -> "MTCKDWaterContinuum":
        """Load the official MT_CKD coefficient file distributed with LBLRTM."""

        from scipy.io import netcdf_file

        with netcdf_file(path, "r", mmap=False) as dataset:
            def values(name: str) -> np.ndarray:
                return np.asarray(dataset.variables[name].data.copy(), dtype=float)

            title = getattr(dataset, "Title", b"unknown")
            if isinstance(title, bytes):
                title = title.decode("utf-8", errors="replace")
            return cls(
                wavenumber_cm1=np.asarray(wavenumber_cm1, dtype=float),
                coefficient_wavenumber_cm1=values("wavenumbers"),
                self_absco_ref=values("self_absco_ref"),
                foreign_absco_ref=values("for_absco_ref"),
                self_temperature_exponent=values("self_texp"),
                reference_pressure_hpa=float(values("ref_press")),
                reference_temperature_k=float(values("ref_temp")),
                version=str(title).strip(),
            )

    def optical_depth(
        self, profile: AtmosphereProfile, scaled_vmr: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Return vertical self plus foreign optical depth by layer."""

        if "H2O" not in scaled_vmr:
            raise ValueError("MT_CKD water continuum requires an H2O VMR profile")
        temperature = jnp.asarray(profile.temperature_k)[:, None, None]
        pressure_edges = jnp.asarray(profile.pressure_edges_bar)
        # Continuum opacity is proportional to absorber column times collider
        # density.  With the layer state held constant and hydrostatic column
        # coordinate dN proportional to dP, the exact layer-mean pressure is
        # the arithmetic mean of its bounding pressures.
        pressure_hpa = (
            0.5 * (pressure_edges[:-1] + pressure_edges[1:])[:, None, None] * 1000.0
        )
        water_vmr = jnp.asarray(scaled_vmr["H2O"])[:, None, None]
        indices = jnp.asarray(self._indices)
        weights = jnp.asarray(self._weights)[None, :, :]

        coefficient_nu = jnp.asarray(self.coefficient_wavenumber_cm1)[indices][None, :, :]
        self_ref = jnp.asarray(self.self_absco_ref)[indices][None, :, :]
        foreign_ref = jnp.asarray(self.foreign_absco_ref)[indices][None, :, :]
        exponent = jnp.asarray(self.self_temperature_exponent)[indices][None, :, :]

        ratio = coefficient_nu * _RADIATION_CONSTANT_K_CM / temperature
        radiation = jnp.where(
            ratio <= 0.01,
            0.5 * ratio * coefficient_nu,
            coefficient_nu * jnp.tanh(0.5 * ratio),
        )
        density_ratio = (
            pressure_hpa / self.reference_pressure_hpa
            * self.reference_temperature_k / temperature
        )
        self_coefficient = (
            self_ref
            * (self.reference_temperature_k / temperature) ** exponent
            * density_ratio
            * radiation
        )
        foreign_coefficient = foreign_ref * density_ratio * radiation
        # VMR is constant across the four spectral stencil points. Apply it
        # after interpolation so reverse-mode differentiation does not carry
        # four times as many abundance-dependent values.
        self_cross_section = jnp.sum(self_coefficient * weights, axis=-1)
        foreign_cross_section = jnp.sum(foreign_coefficient * weights, axis=-1)
        absorption_cross_section = (
            self_cross_section * water_vmr[:, 0, :]
            + foreign_cross_section * (1.0 - water_vmr[:, 0, :])
        )
        water_column = (
            jnp.asarray(profile.air_column_cm2) * jnp.asarray(scaled_vmr["H2O"])
        )
        return absorption_cross_section * water_column[:, None]

    def bind(self, profile: AtmosphereProfile) -> "_BoundMTCKDWaterContinuum":
        """Precompute the fixed pressure-temperature part for one model profile."""

        if "H2O" not in profile.vmr:
            raise ValueError("MT_CKD water continuum requires an H2O VMR profile")
        temperature = np.asarray(profile.temperature_k)[:, None, None]
        edges = np.asarray(profile.pressure_edges_bar)
        pressure_hpa = 0.5 * (edges[:-1] + edges[1:])[:, None, None] * 1000.0
        coefficient_nu = self.coefficient_wavenumber_cm1[self._indices][None, :, :]
        self_ref = self.self_absco_ref[self._indices][None, :, :]
        foreign_ref = self.foreign_absco_ref[self._indices][None, :, :]
        exponent = self.self_temperature_exponent[self._indices][None, :, :]
        ratio = coefficient_nu * _RADIATION_CONSTANT_K_CM / temperature
        radiation = np.where(
            ratio <= 0.01,
            0.5 * ratio * coefficient_nu,
            coefficient_nu * np.tanh(0.5 * ratio),
        )
        density_ratio = (
            pressure_hpa / self.reference_pressure_hpa
            * self.reference_temperature_k / temperature
        )
        weights = self._weights[None, :, :]
        self_cross_section = np.sum(
            self_ref
            * (self.reference_temperature_k / temperature) ** exponent
            * density_ratio
            * radiation
            * weights,
            axis=-1,
        )
        foreign_cross_section = np.sum(
            foreign_ref * density_ratio * radiation * weights, axis=-1
        )
        return _BoundMTCKDWaterContinuum(
            np.asarray(profile.pressure_edges_bar),
            np.asarray(profile.temperature_k),
            self_cross_section,
            foreign_cross_section,
        )


@dataclass(frozen=True)
class _BoundMTCKDWaterContinuum:
    """MT_CKD spectral factors prepared for a fixed pressure-temperature profile."""

    pressure_edges_bar: np.ndarray
    temperature_k: np.ndarray
    self_cross_section: np.ndarray
    foreign_cross_section: np.ndarray

    def optical_depth(
        self, profile: AtmosphereProfile, scaled_vmr: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray:
        checks = (
            (self.pressure_edges_bar, np.asarray(profile.pressure_edges_bar)),
            (self.temperature_k, np.asarray(profile.temperature_k)),
        )
        if any(
            left.shape != right.shape or not np.array_equal(left, right)
            for left, right in checks
        ):
            raise ValueError("prepared MT_CKD continuum does not match the model profile")
        water_vmr = jnp.asarray(scaled_vmr["H2O"])[:, None]
        cross_section = (
            jnp.asarray(self.self_cross_section) * water_vmr
            + jnp.asarray(self.foreign_cross_section) * (1.0 - water_vmr)
        )
        water_column = jnp.asarray(profile.air_column_cm2)[:, None] * water_vmr
        return cross_section * water_column
