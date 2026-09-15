"""Readers and provenance helpers for LBLRTM validation products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, correlation_lags


@dataclass(frozen=True)
class LBLRTMSpectrum:
    wavenumber_cm1: np.ndarray
    transmission: np.ndarray


@dataclass(frozen=True)
class ValidationMetrics:
    median_absolute_error: float
    percentile_99_absolute_error: float
    maximum_absolute_error: float
    line_shift_resolution_elements: float
    compared_samples: int


def read_tape12_single_precision(path: str | Path) -> LBLRTMSpectrum:
    """Read transmission panels from a single-precision LBLRTM TAPE12.

    LBLRTM's mixed-type unformatted file is compiler-layout dependent. This
    reader intentionally supports the GNU single-precision build used by the
    reproducible reference toolchain and rejects malformed/truncated records.
    """

    content = Path(path).read_bytes()
    offset = 1068
    panel_header = struct.Struct("=ddfl")
    record_markers_and_header = struct.calcsize("=4f")
    wavenumber_panels: list[np.ndarray] = []
    transmission_panels: list[np.ndarray] = []

    while offset + panel_header.size <= len(content):
        panel_start, _panel_end, spacing, npoints = panel_header.unpack_from(content, offset)
        if npoints <= 0:
            break
        if not np.isfinite(panel_start) or not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("invalid LBLRTM panel header")
        offset += panel_header.size + record_markers_and_header
        array_size = struct.calcsize(f"={npoints}f")
        if offset + 2 * array_size > len(content):
            raise ValueError("truncated LBLRTM panel")
        offset += array_size  # radiance block
        transmission = np.asarray(struct.unpack_from(f"={npoints}f", content, offset))
        offset += array_size + 8  # payload plus Fortran record markers
        wavenumber_panels.append(panel_start + spacing * np.arange(npoints))
        transmission_panels.append(transmission)

    if not wavenumber_panels:
        raise ValueError("no transmission panels found in TAPE12")
    return LBLRTMSpectrum(
        wavenumber_cm1=np.concatenate(wavenumber_panels),
        transmission=np.concatenate(transmission_panels),
    )


def degrade_to_resolving_power(
    spectrum: LBLRTMSpectrum,
    resolving_power: float = 45_000.0,
    samples_per_resolution: float = 3.0,
) -> LBLRTMSpectrum:
    """Apply a Gaussian LSF and sample a spectrum on a constant-velocity grid."""

    nu = np.asarray(spectrum.wavenumber_cm1, dtype=float)
    transmission = np.clip(np.asarray(spectrum.transmission, dtype=float), 0.0, 1.0)
    if resolving_power <= 0.0 or samples_per_resolution < 2.0:
        raise ValueError("resolving power and sampling must be positive")
    log_grid = np.linspace(np.log(nu[0]), np.log(nu[-1]), len(nu))
    uniform = np.interp(np.exp(log_grid), nu, transmission)
    sigma_log = 1.0 / (resolving_power * 2.0 * np.sqrt(2.0 * np.log(2.0)))
    broadened = gaussian_filter1d(uniform, sigma_log / (log_grid[1] - log_grid[0]), mode="nearest")
    count = int(np.floor(np.log(nu[-1] / nu[0]) * resolving_power * samples_per_resolution)) + 1
    output_nu = np.geomspace(nu[0], nu[-1], count)
    return LBLRTMSpectrum(output_nu, np.interp(np.log(output_nu), log_grid, broadened))


def compare_transmission(
    reference: LBLRTMSpectrum,
    model: LBLRTMSpectrum,
    resolving_power: float = 45_000.0,
    minimum_reference_transmission: float = 0.05,
) -> ValidationMetrics:
    """Compute the validation errors and global line displacement."""

    ref_nu = np.asarray(reference.wavenumber_cm1)
    ref_flux = np.asarray(reference.transmission)
    model_flux = np.interp(ref_nu, model.wavenumber_cm1, model.transmission)
    keep = np.isfinite(ref_flux) & np.isfinite(model_flux) & (ref_flux > minimum_reference_transmission)
    if np.count_nonzero(keep) < 3:
        raise ValueError("too few valid reference samples")
    errors = np.abs(ref_flux[keep] - model_flux[keep])

    # Cross-correlate absorption depths. The sub-sample parabolic refinement
    # makes the result useful on the normal three-samples-per-resolution grid.
    correlation_keep = np.isfinite(ref_flux) & np.isfinite(model_flux)
    ref_depth = 1.0 - ref_flux[correlation_keep]
    model_depth = 1.0 - model_flux[correlation_keep]
    ref_depth -= np.mean(ref_depth)
    model_depth -= np.mean(model_depth)
    correlation = correlate(model_depth, ref_depth, mode="full", method="fft")
    lags = correlation_lags(len(model_depth), len(ref_depth), mode="full")
    dlog = float(np.median(np.diff(np.log(ref_nu[correlation_keep]))))
    allowed = np.abs(lags * dlog * resolving_power) <= 2.0
    allowed_indices = np.flatnonzero(allowed)
    peak = int(allowed_indices[np.argmax(correlation[allowed])])
    lag = float(lags[peak])
    if 0 < peak < len(correlation) - 1:
        left, center, right = correlation[peak - 1 : peak + 2]
        denominator = left - 2.0 * center + right
        if denominator != 0.0:
            lag += 0.5 * (left - right) / denominator
    shift_resolution_elements = lag * dlog * resolving_power
    return ValidationMetrics(
        median_absolute_error=float(np.median(errors)),
        percentile_99_absolute_error=float(np.percentile(errors, 99.0)),
        maximum_absolute_error=float(np.max(errors)),
        line_shift_resolution_elements=float(shift_resolution_elements),
        compared_samples=int(len(errors)),
    )
