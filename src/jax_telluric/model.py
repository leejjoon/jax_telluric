"""Differentiable transmission and per-order observation model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from .types import AtmosphereProfile, SpectralOrder, TelluricParameters

_C_KMS = 299792.458
# Full width at half maximum of sinc(x) = sin(x)/x, in units of 1 / (2 L).
_BOXCAR_FWHM_CONSTANT = 1.20671


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


def trim_wavenumber_grid(
    wavenumber_cm1: np.ndarray, nu_min: float, nu_max: float, margin_cm1: float
) -> np.ndarray:
    """Keep the samples within ``margin_cm1`` of a window, preserving phase.

    Two margins are easily confused. A *line* margin decides which lines can
    reach the window and belongs to line selection; on this atlas 25 cm-1 of it
    leaves 70% of the grid outside any data, and up to 90% on a narrow page.
    A *grid* margin only has to cover what the forward model reaches back for:
    the LSF kernel, the Doppler shifts, and the instrument profile's edge
    padding. A few cm-1 covers all three.

    Trimming rather than regenerating keeps the spacing *and* the phase of the
    original grid, so the trimmed model samples the same wavenumbers and the
    only difference is what was cut. Measured against an untrimmed 25 cm-1
    grid at a 2 cm-1 margin: maximum pixel-flux difference 2.5e-4, confined to
    the outermost pixels, against 5.5e-3 of photon noise.
    """

    grid = np.asarray(wavenumber_cm1, dtype=float)
    if grid.ndim != 1 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("the wavenumber grid must be one-dimensional and increasing")
    if not 0.0 < nu_min < nu_max or margin_cm1 < 0.0:
        raise ValueError("invalid window or margin")
    if grid[0] > nu_min or grid[-1] < nu_max:
        raise ValueError("the grid does not cover the requested window")
    # Take the bracketing sample on each side, so the result always spans the
    # window plus the margin rather than stopping just inside it.
    lower = max(0, int(np.searchsorted(grid, nu_min - margin_cm1, side="right")) - 1)
    upper = min(grid.size, int(np.searchsorted(grid, nu_max + margin_cm1, side="left")) + 1)
    trimmed = grid[lower:upper]
    if trimmed.size < 8:
        raise ValueError("trimming leaves too few samples")
    return trimmed


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

    def mt_ckd_optical_depth(
        self, profile: AtmosphereProfile, scaled_vmr: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray: ...


class InstrumentProfile(Protocol):
    """Internal seam for a non-Gaussian instrument line shape."""

    def convolve(
        self, spectrum: jnp.ndarray, parameters: TelluricParameters, velocity_step_kms: float
    ) -> jnp.ndarray:
        """Return the spectrum convolved with the instrument profile."""


def _catmull_rom(values: jnp.ndarray, position: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a uniformly spaced array at fractional sample positions.

    Linear interpolation would make the objective's derivative a staircase in
    velocity: it changes slope every time the shift crosses a sample. This
    spline is C1, so the gradient a fitter sees stays continuous.
    """

    count = values.shape[0]
    lower = jnp.floor(position)
    fraction = position - lower
    index = lower.astype(jnp.int32)

    def at(offset: int) -> jnp.ndarray:
        return values[jnp.clip(index + offset, 0, count - 1)]

    p0, p1, p2, p3 = at(-1), at(0), at(1), at(2)
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * fraction
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * fraction**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * fraction**3
    )


def _gaussian_convolve(
    spectrum: jnp.ndarray, sigma_kms: jnp.ndarray, velocity_step_kms: float, half_width: int
) -> jnp.ndarray:
    offsets = jnp.arange(-half_width, half_width + 1)
    sigma_pixels = jnp.maximum(jnp.asarray(sigma_kms) / velocity_step_kms, 1.0e-6)
    kernel = jnp.exp(-0.5 * (offsets / sigma_pixels) ** 2)
    kernel = kernel / jnp.sum(kernel)
    padded = jnp.pad(spectrum, (half_width, half_width), mode="edge")
    return jnp.convolve(padded, kernel, mode="valid")


@dataclass(frozen=True)
class BoxcarFTSInstrumentProfile:
    """Unapodized FTS sinc profile, with a Gaussian for residual broadening.

    A Fourier transform spectrometer truncates its interferogram at the maximum
    optical path difference, so its line shape is ``sinc(2 pi L dnu)`` and its
    transfer function is a boxcar in optical path difference. That is applied
    here by multiplying the transform of the spectrum, because the sinc decays
    only as 1/x and any truncated kernel would be wrong at every half width.

    The sinc is the measured, known part of the profile. ``lsf_sigma_kms``
    remains free and absorbs what is left -- coadd smearing, macroturbulence
    mismatch, and the residual apodization the measurement does not resolve.

    The model grid is uniform in log wavenumber, so this applies a profile of
    constant *velocity* width, while a real FTS profile has constant width in
    wavenumber. Across a window of fractional width w the resulting width error
    is w; it is 0.4% across a 20 cm-1 window at 5000 cm-1.
    """

    mopd_cm: float
    wavenumber_center_cm1: float
    max_residual_sigma_kms: float = 5.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mopd_cm) or self.mopd_cm <= 0.0:
            raise ValueError("maximum optical path difference must be finite and positive")
        if not np.isfinite(self.wavenumber_center_cm1) or self.wavenumber_center_cm1 <= 0.0:
            raise ValueError("the profile's center wavenumber must be finite and positive")
        if not np.isfinite(self.max_residual_sigma_kms) or self.max_residual_sigma_kms < 0.0:
            raise ValueError("the residual broadening bound must be finite and nonnegative")

    @property
    def first_zero_kms(self) -> float:
        """Velocity offset of the first zero of the sinc."""

        return _C_KMS / (2.0 * self.mopd_cm * self.wavenumber_center_cm1)

    @property
    def fwhm_cm1(self) -> float:
        return _BOXCAR_FWHM_CONSTANT / (2.0 * self.mopd_cm)

    @property
    def resolving_power(self) -> float:
        return self.wavenumber_center_cm1 / self.fwhm_cm1

    def convolve(self, spectrum, parameters, velocity_step_kms):
        count = spectrum.shape[0]
        # Edge padding keeps the cyclic transform from folding the far end of
        # the order into the near end through the sinc's slowly decaying wings.
        pad = count // 2
        padded = jnp.pad(spectrum, (pad, pad), mode="edge")
        frequency = jnp.fft.rfftfreq(padded.shape[0], d=velocity_step_kms)
        transfer = (frequency <= 1.0 / (2.0 * self.first_zero_kms)).astype(padded.dtype)
        convolved = jnp.fft.irfft(jnp.fft.rfft(padded) * transfer, n=padded.shape[0])[pad : pad + count]
        if self.max_residual_sigma_kms <= 0.0:
            return convolved
        half_width = int(np.ceil(5.0 * self.max_residual_sigma_kms / velocity_step_kms))
        return _gaussian_convolve(convolved, parameters.lsf_sigma_kms, velocity_step_kms, half_width)


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
class LinearizedOpacityBackend:
    """Precomputed cross sections, first order in the self-broadening pressure.

    A fitted column scale reaches ``xsvector`` only through the self-broadening
    partial pressure, and only weakly: ``gamma_hitran`` is linear in it and the
    Voigt profile is smooth. Expanding about a reference partial pressure
    therefore captures almost all of that dependence with two fixed arrays,
    which removes the line-by-line kernel -- over 90% of a forward call -- from
    every optimizer iteration while leaving the column scales differentiable.

    Measured on the Kitt Peak profile at 5005-5025 cm-1 against the exact
    calculator, over water columns from 0.50x to 2.72x the reference: maximum
    transmission error 6.7e-4 and rms 2.9e-5, against 5.5e-3 of photon noise.
    Simply holding the cross sections fixed is 70x worse (1.5e-2 maximum).
    Accuracy degrades away from the reference, so build this from a converged
    fit when the column is known to be far from the profile's own value.
    """

    values: Mapping[str, np.ndarray]
    derivative: Mapping[str, np.ndarray]
    reference_partial_pressure_bar: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        values = {name.upper(): np.asarray(v, dtype=float) for name, v in self.values.items()}
        derivative = {name.upper(): np.asarray(v, dtype=float) for name, v in self.derivative.items()}
        reference = {
            name.upper(): np.asarray(v, dtype=float)
            for name, v in self.reference_partial_pressure_bar.items()
        }
        if set(values) != set(derivative) or set(values) != set(reference):
            raise ValueError("values, derivative, and reference must cover the same species")
        shapes = {v.shape for v in values.values()} | {v.shape for v in derivative.values()}
        if not values or len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError("cross sections must share one (layer, wavenumber) shape")
        layers = next(iter(shapes))[0]
        if any(v.shape != (layers,) for v in reference.values()):
            raise ValueError("reference partial pressures must have one value per layer")
        if any(np.any(v < 0.0) or np.any(~np.isfinite(v)) for v in values.values()):
            raise ValueError("cross sections must be finite and nonnegative")
        if any(np.any(~np.isfinite(v)) for v in derivative.values()):
            raise ValueError("cross-section derivatives must be finite")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "derivative", derivative)
        object.__setattr__(self, "reference_partial_pressure_bar", reference)

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.values)

    def cross_sections(self, temperature_k, pressure_bar, partial_pressure_bar):
        del temperature_k, pressure_bar
        result = {}
        for species, value in self.values.items():
            delta = (
                jnp.asarray(partial_pressure_bar[species])
                - jnp.asarray(self.reference_partial_pressure_bar[species])
            )
            # A cross section cannot be negative. The expansion stays far from
            # this over any column a fit explores; the floor only stops an
            # extrapolated value from turning into a negative optical depth.
            result[species] = jnp.maximum(
                jnp.asarray(value) + jnp.asarray(self.derivative[species]) * delta[:, None], 0.0
            )
        return result


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
        pixel_integration: str = "simpson",
        instrument: InstrumentProfile | None = None,
    ) -> None:
        nu = np.asarray(wavenumber_cm1, dtype=float)
        if nu.ndim != 1 or len(nu) < 8 or np.any(np.diff(nu) <= 0.0):
            raise ValueError("wavenumber grid must be strictly increasing with at least eight points")
        dlog = np.diff(np.log(nu))
        if not np.allclose(dlog, dlog[0], rtol=1e-5, atol=0.0):
            raise ValueError("wavenumber grid must be evenly spaced in log wavenumber")
        if set(opacity.species) - set(profile.vmr):
            raise ValueError("the atmosphere has no VMR profile for an opacity species")
        if accuracy_mode not in ("fast", "mt_ckd", "lblrtm_corrected"):
            raise ValueError("accuracy_mode must be 'fast', 'mt_ckd', or 'lblrtm_corrected'")
        if pixel_integration not in ("simpson", "point"):
            raise ValueError("pixel_integration must be 'simpson' or 'point'")
        if accuracy_mode == "fast" and correction is not None:
            raise ValueError("a correction requires accuracy_mode='mt_ckd' or 'lblrtm_corrected'")
        if accuracy_mode == "lblrtm_corrected" and correction is None:
            raise ValueError("accuracy_mode='lblrtm_corrected' requires a correction template")
        if accuracy_mode == "lblrtm_corrected" and continuum is not None:
            raise ValueError("LBLRTM-corrected mode already includes the H2O continuum")
        if accuracy_mode == "mt_ckd" and (continuum is None) == (correction is None):
            raise ValueError(
                "accuracy_mode='mt_ckd' requires exactly one continuum backend or correction template"
            )
        if correction is not None:
            if accuracy_mode == "mt_ckd":
                validate_mt_ckd = getattr(correction, "validate_mt_ckd", None)
                if validate_mt_ckd is None or not hasattr(correction, "mt_ckd_optical_depth"):
                    raise TypeError("mt_ckd mode requires an MT_CKD correction template")
                validate_mt_ckd(profile, nu)
            else:
                correction.validate(profile, nu)
                validate_opacity = getattr(correction, "validate_opacity", None)
                if validate_opacity is not None:
                    validate_opacity(opacity)
        self.profile = profile
        self.wavenumber_cm1 = jnp.asarray(nu)
        self.opacity = opacity
        correction_species = correction.species if accuracy_mode == "lblrtm_corrected" else ()
        self.species = tuple(dict.fromkeys((*opacity.species, *correction_species)))
        bind_continuum = getattr(continuum, "bind", None)
        self.continuum = bind_continuum(profile) if bind_continuum is not None else continuum
        self.accuracy_mode = accuracy_mode
        self.correction = correction
        self.pixel_integration = pixel_integration
        self.instrument = instrument
        self.velocity_step_kms = float(dlog[0] * _C_KMS)
        self.kernel_half_width = int(np.ceil(5.0 * max_lsf_sigma_kms / self.velocity_step_kms))

    @classmethod
    def prepare(cls, profile, wavenumber_cm1, opacity, **kwargs) -> "TelluricModel":
        return cls(profile, wavenumber_cm1, opacity, **kwargs)

    def precompute_opacity(
        self,
        parameters: TelluricParameters | None = None,
        self_broadening: str = "linear",
        relative_step: float = 0.25,
    ) -> "TelluricModel":
        """Return a copy whose cross sections are precomputed for this profile.

        The line-by-line kernel is over 90% of a forward call, yet a fitted
        parameter reaches it only through the self-broadening partial pressure.
        Evaluating it once here and carrying the result as fixed arrays makes an
        optimizer iteration roughly thirty times cheaper, with the column scales
        still free through the linear ``tau = sigma * column`` factor and the
        continuum. Everything else in the model is unchanged, so the returned
        copy is used exactly like the original.

        ``self_broadening="linear"`` expands to first order about the reference
        partial pressure, which holds over a wide range of columns and needs no
        iteration. ``"frozen"`` drops that term entirely; it is cheaper to build
        but only accurate near the reference, so a fit using it must be repeated
        from its own result until the column stops moving.

        ``parameters`` sets the reference column scales; the profile's own
        values are used when it is omitted. The derivative is a central
        difference of ``relative_step`` times the reference partial pressure,
        because a JVP through the calculator's out-of-range ``lax.cond`` is
        batched into a select that evaluates the dense fallback branch.
        """

        if self_broadening not in ("linear", "frozen"):
            raise ValueError("self_broadening must be 'linear' or 'frozen'")
        if not 0.0 < relative_step < 1.0:
            raise ValueError("relative_step must lie between zero and one")
        pressure = jnp.asarray(self.profile.pressure_layer_bar)
        scales = {} if parameters is None else parameters.log_column_scales
        vmr_scaled = {
            species: jnp.asarray(vmr) * jnp.exp(jnp.asarray(scales.get(species, 0.0)))
            for species, vmr in self.profile.vmr.items()
        }
        reference = {
            species: pressure * vmr_scaled[species] for species in self.opacity.species
        }
        temperature = jnp.asarray(self.profile.temperature_k)
        # One compilation, reused for all three evaluations here and for every
        # later call on this model. Called eagerly the kernel dispatches
        # operation by operation and costs more than the whole fit it is meant
        # to accelerate; compiled afresh each time, it costs seconds.
        cached = getattr(self, "_cross_section_evaluator", None)
        if cached is None or cached[0] is not self.opacity:
            backend = self.opacity
            evaluate = jax.jit(
                lambda partial: backend.cross_sections(temperature, pressure, partial)
            )
            self._cross_section_evaluator = (backend, evaluate)
        else:
            evaluate = cached[1]
        values = evaluate(reference)
        if self_broadening == "frozen":
            backend = ArrayOpacityBackend(
                {species: np.asarray(value) for species, value in values.items()}
            )
        else:
            step = {
                species: relative_step * np.asarray(value)
                for species, value in reference.items()
            }
            upper = evaluate({s: reference[s] + step[s] for s in reference})
            lower = evaluate({s: reference[s] - step[s] for s in reference})
            derivative = {}
            for species, width in step.items():
                # A species with no self pressure has no self-broadening term
                # to expand, and dividing by its zero step would be undefined.
                safe = np.where(width > 0.0, width, 1.0)
                slope = (np.asarray(upper[species]) - np.asarray(lower[species])) / (2.0 * safe)[:, None]
                derivative[species] = np.where((width > 0.0)[:, None], slope, 0.0)
            backend = LinearizedOpacityBackend(
                {species: np.asarray(value) for species, value in values.items()},
                derivative,
                {species: np.asarray(value) for species, value in reference.items()},
            )
        precomputed = copy.copy(self)
        precomputed.opacity = backend
        # The copy's own evaluator would otherwise be the original's, compiled
        # against a backend it no longer holds.
        precomputed._cross_section_evaluator = None
        return precomputed

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
            if self.accuracy_mode == "mt_ckd":
                tau = tau + self.correction.mt_ckd_optical_depth(self.profile, vmr_scaled)
            else:
                tau = tau + self.correction.optical_depth(self.profile, vmr_scaled)
        for species in self.opacity.species:
            tau = tau + xs[species] * (air_column * vmr_scaled[species])[:, None]
        mu = jnp.cos(jnp.deg2rad(zenith_angle_deg))
        return jnp.exp(-jnp.sum(tau, axis=0) / mu)

    def _convolve_lsf(self, spectrum: jnp.ndarray, sigma_kms: jnp.ndarray) -> jnp.ndarray:
        return _gaussian_convolve(spectrum, sigma_kms, self.velocity_step_kms, self.kernel_half_width)

    def _shift_log_uniform(self, values: jnp.ndarray, velocity_kms: jnp.ndarray) -> jnp.ndarray:
        """Doppler-shift a spectrum sampled uniformly in log wavenumber.

        On this grid a shift is a constant translation in samples, so no
        wavelength-space interpolation is needed. The 25 cm-1 padding is far
        wider than any plausible stellar velocity, so clamping at the ends
        never reaches the fitted window.
        """

        shift = jnp.log1p(jnp.asarray(velocity_kms) / _C_KMS) * (_C_KMS / self.velocity_step_kms)
        count = values.shape[0]
        position = jnp.clip(jnp.arange(count) + shift, 0.0, count - 1.0)
        return _catmull_rom(values, position)

    def predict(self, order: SpectralOrder, parameters: TelluricParameters) -> jnp.ndarray:
        wavelength = jnp.asarray(order.wavelength_vacuum_nm)
        x = jnp.linspace(-1.0, 1.0, wavelength.size)
        shifted_wavelength = wavelength * (1.0 + jnp.asarray(parameters.velocity_kms) / _C_KMS)
        shifted_wavelength = shifted_wavelength * (1.0 + jnp.asarray(parameters.wavelength_stretch) * x)

        wavelength_hi = 1.0e7 / self.wavenumber_cm1
        if order.source_flux_model_grid is None:
            source_hi = jnp.interp(wavelength_hi, wavelength, jnp.asarray(order.source_flux))
        else:
            model_source = jnp.asarray(order.source_flux_model_grid)
            if model_source.shape != (self.wavenumber_cm1.size,):
                raise ValueError("model-grid source flux must match the model wavenumber grid")
            source_hi = self._shift_log_uniform(model_source, parameters.stellar_velocity_kms)
        raw_hi = self.transmission(parameters, order.zenith_angle_deg) * source_hi
        if self.instrument is None:
            convolved_hi = self._convolve_lsf(raw_hi, jnp.asarray(parameters.lsf_sigma_kms))
        else:
            convolved_hi = self.instrument.convolve(raw_hi, parameters, self.velocity_step_kms)

        pixel_edges = jnp.concatenate(
            [
                shifted_wavelength[:1] - 0.5 * (shifted_wavelength[1:2] - shifted_wavelength[:1]),
                0.5 * (shifted_wavelength[:-1] + shifted_wavelength[1:]),
                shifted_wavelength[-1:] + 0.5 * (shifted_wavelength[-1:] - shifted_wavelength[-2:-1]),
            ]
        )
        center = jnp.interp(shifted_wavelength, wavelength_hi[::-1], convolved_hi[::-1])
        if self.pixel_integration == "point":
            # A Fourier transform spectrometer delivers point samples of a
            # band-limited function; averaging over a notional pixel would add
            # a low-pass filter the instrument never applied.
            pixel_average = center
        else:
            left = jnp.interp(pixel_edges[:-1], wavelength_hi[::-1], convolved_hi[::-1])
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
