"""Differentiable transmission and per-order observation model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import jax.numpy as jnp
import numpy as np

from .types import AtmosphereProfile, SpectralOrder, TelluricParameters

_C_KMS = 299792.458


def igrins_wavenumber_grid(
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    resolving_power: float = 45_000.0,
    samples_per_resolution: float = 4.0,
    margin_cm1: float = 25.0,
) -> np.ndarray:
    """Create an ascending constant-velocity grid padded beyond one order."""

    if not 0.0 < wavelength_min_nm < wavelength_max_nm:
        raise ValueError("wavelength limits must be positive and increasing")
    if resolving_power <= 0.0 or samples_per_resolution < 2.0 or margin_cm1 < 0.0:
        raise ValueError("invalid resolution, sampling, or margin")
    nu_min = 1.0e7 / wavelength_max_nm - margin_cm1
    nu_max = 1.0e7 / wavelength_min_nm + margin_cm1
    if nu_min <= 0.0:
        raise ValueError("margin extends below zero wavenumber")
    dlog = 1.0 / (resolving_power * samples_per_resolution)
    npoints = int(np.ceil(np.log(nu_max / nu_min) / dlog)) + 1
    return np.geomspace(nu_min, nu_max, npoints)


class OpacityBackend(Protocol):
    """Internal seam for prepared opacity implementations."""

    species: tuple[str, ...]

    def cross_sections(
        self,
        temperature_k: jnp.ndarray,
        pressure_bar: jnp.ndarray,
        partial_pressure_bar: Mapping[str, jnp.ndarray],
    ) -> Mapping[str, jnp.ndarray]:
        """Return cross sections in cm2/molecule with shape (layer, wavenumber)."""


class ContinuumBackend(Protocol):
    """Internal seam for continuum optical depth."""

    def optical_depth(
        self, profile: AtmosphereProfile, scaled_vmr: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Return vertical optical depth with shape (layer, wavenumber)."""


class CorrectionBackend(Protocol):
    """Internal seam for an opt-in fixed-profile optical-depth correction."""

    species: tuple[str, ...]

    def validate(self, profile: AtmosphereProfile, wavenumber_cm1: np.ndarray) -> None: ...

    def optical_depth(
        self, profile: AtmosphereProfile, scaled_vmr: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray: ...


@dataclass(frozen=True)
class ArrayOpacityBackend:
    """Fixed cross sections for fixtures and precomputed-opacity workflows."""

    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        normalized = {name.upper(): np.asarray(value, dtype=float) for name, value in self.values.items()}
        shapes = {value.shape for value in normalized.values()}
        if not normalized or len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError("cross sections must share one (layer, wavenumber) shape")
        if any(np.any(value < 0.0) or np.any(~np.isfinite(value)) for value in normalized.values()):
            raise ValueError("cross sections must be finite and nonnegative")
        object.__setattr__(self, "values", normalized)

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.values)

    def cross_sections(self, temperature_k, pressure_bar, partial_pressure_bar):
        del temperature_k, pressure_bar, partial_pressure_bar
        return {name: jnp.asarray(value) for name, value in self.values.items()}


@dataclass(frozen=True)
class ReferenceWaterContinuum:
    """LBLRTM-derived MT_CKD H2O continuum terms at a reference profile.

    Foreign continuum scales linearly with H2O abundance. Self continuum
    scales quadratically. This keeps the fit differentiable while retaining
    the exact LBLRTM spectral shape for the fixed pressure-temperature profile.
    """

    reference_vmr: np.ndarray
    self_optical_depth: np.ndarray
    foreign_optical_depth: np.ndarray

    def __post_init__(self) -> None:
        vmr = np.asarray(self.reference_vmr, dtype=float)
        self_tau = np.asarray(self.self_optical_depth, dtype=float)
        foreign_tau = np.asarray(self.foreign_optical_depth, dtype=float)
        if self_tau.shape != foreign_tau.shape or self_tau.shape[0] != vmr.size:
            raise ValueError("continuum arrays must share shape (layer, wavenumber)")
        if np.any(vmr <= 0.0) or np.any(self_tau < 0.0) or np.any(foreign_tau < 0.0):
            raise ValueError("reference VMR must be positive and optical depths nonnegative")
        object.__setattr__(self, "reference_vmr", vmr)
        object.__setattr__(self, "self_optical_depth", self_tau)
        object.__setattr__(self, "foreign_optical_depth", foreign_tau)

    def optical_depth(self, profile, scaled_vmr):
        del profile
        scale = scaled_vmr["H2O"] / jnp.asarray(self.reference_vmr)
        return (
            jnp.asarray(self.self_optical_depth) * scale[:, None] ** 2
            + jnp.asarray(self.foreign_optical_depth) * scale[:, None]
        )


class TelluricModel:
    """Prepared high-resolution atmosphere and instrument forward model."""

    def __init__(
        self,
        profile: AtmosphereProfile,
        wavenumber_cm1: np.ndarray,
        opacity: OpacityBackend,
        continuum: ContinuumBackend | None = None,
        max_lsf_sigma_kms: float = 20.0,
        accuracy_mode: str = "fast",
        correction: CorrectionBackend | None = None,
    ) -> None:
        nu = np.asarray(wavenumber_cm1, dtype=float)
        if nu.ndim != 1 or len(nu) < 8 or np.any(np.diff(nu) <= 0.0):
            raise ValueError("wavenumber grid must be strictly increasing with at least eight points")
        dlog = np.diff(np.log(nu))
        if not np.allclose(dlog, dlog[0], rtol=1e-5, atol=0.0):
            raise ValueError("wavenumber grid must be evenly spaced in log wavenumber")
        if set(opacity.species) - set(profile.vmr):
            raise ValueError("the atmosphere has no VMR profile for an opacity species")
        if accuracy_mode not in ("fast", "lblrtm_corrected"):
            raise ValueError("accuracy_mode must be 'fast' or 'lblrtm_corrected'")
        if accuracy_mode == "fast" and correction is not None:
            raise ValueError("a correction requires accuracy_mode='lblrtm_corrected'")
        if accuracy_mode == "lblrtm_corrected" and correction is None:
            raise ValueError("accuracy_mode='lblrtm_corrected' requires a correction template")
        if accuracy_mode == "lblrtm_corrected" and continuum is not None:
            raise ValueError("LBLRTM corrections already include the H2O continuum")
        if correction is not None:
            correction.validate(profile, nu)
            validate_opacity = getattr(correction, "validate_opacity", None)
            if validate_opacity is not None:
                validate_opacity(opacity)
        self.profile = profile
        self.wavenumber_cm1 = jnp.asarray(nu)
        self.opacity = opacity
        correction_species = () if correction is None else correction.species
        self.species = tuple(dict.fromkeys((*opacity.species, *correction_species)))
        self.continuum = continuum
        self.accuracy_mode = accuracy_mode
        self.correction = correction
        self.velocity_step_kms = float(dlog[0] * _C_KMS)
        self.kernel_half_width = int(np.ceil(5.0 * max_lsf_sigma_kms / self.velocity_step_kms))

    @classmethod
    def prepare(cls, profile, wavenumber_cm1, opacity, **kwargs) -> "TelluricModel":
        return cls(profile, wavenumber_cm1, opacity, **kwargs)

    def transmission(self, parameters: TelluricParameters, zenith_angle_deg: float = 0.0) -> jnp.ndarray:
        pressure = jnp.asarray(self.profile.pressure_layer_bar)
        vmr_scaled = {
            species: jnp.asarray(vmr)
            * jnp.exp(jnp.asarray(parameters.log_column_scales.get(species, 0.0)))
            for species, vmr in self.profile.vmr.items()
        }
        partial_pressure = {
            species: pressure * vmr_scaled[species] for species in self.opacity.species
        }
        xs = self.opacity.cross_sections(jnp.asarray(self.profile.temperature_k), pressure, partial_pressure)
        air_column = jnp.asarray(self.profile.air_column_cm2)
        tau = jnp.zeros((air_column.size, self.wavenumber_cm1.size))
        if self.continuum is not None:
            tau = tau + self.continuum.optical_depth(self.profile, vmr_scaled)
        if self.correction is not None:
            tau = tau + self.correction.optical_depth(self.profile, vmr_scaled)
        for species in self.opacity.species:
            tau = tau + xs[species] * (air_column * vmr_scaled[species])[:, None]
        mu = jnp.cos(jnp.deg2rad(zenith_angle_deg))
        return jnp.exp(-jnp.sum(tau, axis=0) / mu)

    def _convolve_lsf(self, spectrum: jnp.ndarray, sigma_kms: jnp.ndarray) -> jnp.ndarray:
        offsets = jnp.arange(-self.kernel_half_width, self.kernel_half_width + 1)
        sigma_pixels = jnp.maximum(sigma_kms / self.velocity_step_kms, 1.0e-6)
        kernel = jnp.exp(-0.5 * (offsets / sigma_pixels) ** 2)
        kernel = kernel / jnp.sum(kernel)
        padded = jnp.pad(spectrum, (self.kernel_half_width, self.kernel_half_width), mode="edge")
        return jnp.convolve(padded, kernel, mode="valid")

    def predict(self, order: SpectralOrder, parameters: TelluricParameters) -> jnp.ndarray:
        wavelength = jnp.asarray(order.wavelength_vacuum_nm)
        x = jnp.linspace(-1.0, 1.0, wavelength.size)
        shifted_wavelength = wavelength * (1.0 + jnp.asarray(parameters.velocity_kms) / _C_KMS)
        shifted_wavelength = shifted_wavelength * (1.0 + jnp.asarray(parameters.wavelength_stretch) * x)

        wavelength_hi = 1.0e7 / self.wavenumber_cm1
        source_hi = jnp.interp(wavelength_hi, wavelength, jnp.asarray(order.source_flux))
        raw_hi = self.transmission(parameters, order.zenith_angle_deg) * source_hi
        convolved_hi = self._convolve_lsf(raw_hi, jnp.asarray(parameters.lsf_sigma_kms))

        pixel_edges = jnp.concatenate(
            [
                shifted_wavelength[:1] - 0.5 * (shifted_wavelength[1:2] - shifted_wavelength[:1]),
                0.5 * (shifted_wavelength[:-1] + shifted_wavelength[1:]),
                shifted_wavelength[-1:] + 0.5 * (shifted_wavelength[-1:] - shifted_wavelength[-2:-1]),
            ]
        )
        left = jnp.interp(pixel_edges[:-1], wavelength_hi[::-1], convolved_hi[::-1])
        center = jnp.interp(shifted_wavelength, wavelength_hi[::-1], convolved_hi[::-1])
        right = jnp.interp(pixel_edges[1:], wavelength_hi[::-1], convolved_hi[::-1])
        pixel_average = (left + 4.0 * center + right) / 6.0
        coefficients = jnp.asarray(parameters.continuum_coeffs)
        t0 = jnp.ones_like(x)
        continuum_log = coefficients[0] * t0
        if coefficients.size > 1:
            t1 = x
            continuum_log = continuum_log + coefficients[1] * t1
            for index in range(2, coefficients.size):
                t2 = 2.0 * x * t1 - t0
                continuum_log = continuum_log + coefficients[index] * t2
                t0, t1 = t1, t2
        continuum = jnp.exp(continuum_log)
        return continuum * pixel_average
