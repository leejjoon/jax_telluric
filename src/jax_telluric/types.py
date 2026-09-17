"""Validated public data types used by the forward model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


class TelluricParameters(NamedTuple):
    """Differentiable atmospheric and per-order nuisance parameters.

    ``log_column_scales`` maps molecule names to natural-log abundance scale
    factors. Continuum coefficients describe log throughput in increasing
    Chebyshev order, which guarantees a positive multiplicative continuum.

    ``velocity_kms`` shifts the whole prediction and is the wavelength-solution
    zero point. ``stellar_velocity_kms`` additionally shifts only the source
    spectrum, and therefore has an effect only for an order that carries
    ``source_flux_model_grid``.
    """

    log_column_scales: Mapping[str, ArrayLike]
    velocity_kms: ArrayLike
    wavelength_stretch: ArrayLike
    lsf_sigma_kms: ArrayLike
    continuum_coeffs: ArrayLike
    log_jitter: ArrayLike
    stellar_velocity_kms: ArrayLike = 0.0


@dataclass(frozen=True)
class AtmosphereProfile:
    """Layered atmosphere ordered from top to bottom.

    Pressure edges have ``n_layer + 1`` entries and must increase toward the
    ground. Temperatures, altitudes, mean molecular weights, and every VMR
    profile have ``n_layer`` entries. VMR means volume mixing ratio.
    """

    pressure_edges_bar: ArrayLike
    temperature_k: ArrayLike
    altitude_km: ArrayLike
    vmr: Mapping[str, ArrayLike]
    mean_molecular_weight_g_mol: ArrayLike = 28.9647
    gravity_m_s2: ArrayLike = 9.80665

    def __post_init__(self) -> None:
        edges = np.asarray(self.pressure_edges_bar, dtype=float)
        temperature = np.asarray(self.temperature_k, dtype=float)
        altitude = np.asarray(self.altitude_km, dtype=float)
        if edges.ndim != 1 or temperature.ndim != 1 or altitude.ndim != 1:
            raise ValueError("atmospheric profile arrays must be one-dimensional")
        if len(edges) != len(temperature) + 1 or len(altitude) != len(temperature):
            raise ValueError("pressure edges must have one more value than layer arrays")
        if np.any(~np.isfinite(edges)) or np.any(edges <= 0.0):
            raise ValueError("pressure edges must be finite and positive")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("pressure edges must increase from top to bottom")
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("temperatures must be finite and positive")
        if np.any(np.diff(altitude) >= 0.0):
            raise ValueError("altitude must decrease from top to bottom")

        nlayer = len(temperature)
        normalized_vmr: dict[str, np.ndarray] = {}
        for species, values in self.vmr.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != (nlayer,) or np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
                raise ValueError(f"invalid VMR profile for {species}")
            normalized_vmr[species.upper()] = arr

        mmw = np.broadcast_to(np.asarray(self.mean_molecular_weight_g_mol, dtype=float), (nlayer,))
        gravity = np.broadcast_to(np.asarray(self.gravity_m_s2, dtype=float), (nlayer,))
        if np.any(mmw <= 0.0) or np.any(gravity <= 0.0):
            raise ValueError("mean molecular weight and gravity must be positive")

        object.__setattr__(self, "pressure_edges_bar", edges)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "altitude_km", altitude)
        object.__setattr__(self, "vmr", normalized_vmr)
        object.__setattr__(self, "mean_molecular_weight_g_mol", mmw)
        object.__setattr__(self, "gravity_m_s2", gravity)

    @property
    def pressure_layer_bar(self) -> np.ndarray:
        """Geometric-mean pressure at each layer center."""

        edges = np.asarray(self.pressure_edges_bar)
        return np.sqrt(edges[:-1] * edges[1:])

    @property
    def air_column_cm2(self) -> np.ndarray:
        """Hydrostatic total molecular column in each layer."""

        avogadro = 6.02214076e23
        delta_pressure_pa = np.diff(np.asarray(self.pressure_edges_bar)) * 1.0e5
        molar_mass_kg_mol = np.asarray(self.mean_molecular_weight_g_mol) * 1.0e-3
        return delta_pressure_pa / np.asarray(self.gravity_m_s2) / molar_mass_kg_mol * avogadro / 1.0e4


@dataclass(frozen=True)
class SpectralOrder:
    """One extracted order on a strictly increasing vacuum-wavelength grid.

    A source spectrum may be supplied either on the pixel grid, as
    ``source_flux``, or on the forward model's high-resolution wavenumber grid,
    as ``source_flux_model_grid``. The pixel-grid form cannot represent
    structure narrower than a pixel and is convolved again by the fitted line
    spread function, so a synthetic stellar spectrum belongs on the model grid.
    The two are mutually exclusive. The model-grid array is checked against the
    grid itself by :class:`~jax_telluric.model.TelluricModel`, which is the only
    place both are known.
    """

    wavelength_vacuum_nm: ArrayLike
    flux: ArrayLike
    uncertainty: ArrayLike
    mask: ArrayLike | None = None
    source_flux: ArrayLike | None = None
    zenith_angle_deg: float = 0.0
    source_flux_model_grid: ArrayLike | None = None

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_vacuum_nm, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        uncertainty = np.asarray(self.uncertainty, dtype=float)
        if wavelength.ndim != 1 or len(wavelength) < 3:
            raise ValueError("an order needs at least three wavelength pixels")
        if flux.shape != wavelength.shape or uncertainty.shape != wavelength.shape:
            raise ValueError("wavelength, flux, and uncertainty shapes must match")
        if np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("vacuum wavelength must be finite and strictly increasing")
        if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty <= 0.0):
            raise ValueError("uncertainties must be finite and positive")
        if self.source_flux is not None and self.source_flux_model_grid is not None:
            raise ValueError("supply the source on the pixel grid or the model grid, not both")
        mask = np.ones_like(wavelength, dtype=bool) if self.mask is None else np.asarray(self.mask, dtype=bool)
        source = np.ones_like(wavelength) if self.source_flux is None else np.asarray(self.source_flux, dtype=float)
        if mask.shape != wavelength.shape or source.shape != wavelength.shape:
            raise ValueError("mask and source flux shapes must match wavelength")
        if self.source_flux_model_grid is not None:
            model_source = np.asarray(self.source_flux_model_grid, dtype=float)
            if model_source.ndim != 1 or model_source.size < 8:
                raise ValueError("model-grid source flux must be one-dimensional with at least eight samples")
            if np.any(~np.isfinite(model_source)) or np.any(model_source <= 0.0):
                raise ValueError("model-grid source flux must be finite and positive")
            object.__setattr__(self, "source_flux_model_grid", model_source)
        if not 0.0 <= self.zenith_angle_deg < 90.0:
            raise ValueError("zenith angle must be in [0, 90) degrees")
        object.__setattr__(self, "wavelength_vacuum_nm", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "source_flux", source)
