#!/usr/bin/env python
"""Synthesize the fixed Arcturus stellar spectrum used as the telluric fit's source.

Payne Zero needs Python >= 3.11 and PyTorch, while this package is pinned to
3.10 through ``exojax==2.5.0``, so this script does not run in the project
environment. It runs in Payne Zero's own environment and writes an npz that
:class:`jax_telluric.StellarSpectrum` reads back:

    cd /home/jjlee/work/payne-zero
    export PAYNE_ZERO_DATA_ROOT=$PWD/source_data_files
    export PAYNE_ZERO_SYNTHESIS_CACHE_DIR=$PWD/.cache/payne-zero/synthesis
    export NUMBA_CACHE_DIR=$PWD/.cache/payne-zero/numba-atmosphere
    .venv/bin/python /home/jjlee/work/lblrtm/scripts/generate_payne_zero_arcturus.py

The spectrum is a *fixed* input: the stellar labels are held at literature
values and are not fitted. Only the stellar velocity, and optionally the
broadening applied later by ``jax_telluric.prepare_stellar_source``, remain
free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# Arcturus (K1.5 III). All inside Payne Zero's stated five-label support of
# 4,000-10,500 K, 0.7-5.3 in logg, -2.5-0.5 in [M/H], -0.1-0.5 in [alpha/M],
# and 0.5-4.0 km/s in microturbulence.
ARCTURUS_LABELS = {
    "effective_temperature": 4286.0,
    "log_surface_gravity": 1.66,
    "metallicity": -0.52,
    "alpha_enhancement": 0.30,
    "microturbulence_km_s": 1.7,
}
ARCTURUS_LABEL_SOURCE = (
    "adopted in differentiable_stellar_spectroscopy dss/calib/star.py, following "
    "the Payne Zero paper's anchor values"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults cover the padded model grid for the 5005-5025 cm-1 window
    # (4980-5050 cm-1), with a little margin so resampling never extrapolates.
    parser.add_argument("--wavelength-start-nm", type=float, default=1979.0)
    parser.add_argument("--wavelength-end-nm", type=float, default=2009.0)
    # The model grid runs at 0.749 km/s, so the source must be at least that
    # fine or resample_stellar_source refuses it as a resolution lie.
    parser.add_argument("--r-grid", type=int, default=600_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/home/jjlee/work/lblrtm/data/stellar/arcturus_payne_zero_2000nm.npz"),
    )
    args = parser.parse_args()

    from payne_zero_synthesis import synthesize_from_labels

    spectrum = synthesize_from_labels(
        **ARCTURUS_LABELS,
        wavelength_start_nm=args.wavelength_start_nm,
        wavelength_end_nm=args.wavelength_end_nm,
        r_grid=args.r_grid,
        device=args.device,
    )

    wavelength_nm = np.asarray(spectrum.wavelength_nm, dtype=float)
    normalized = np.asarray(spectrum.normalized_flux, dtype=float)
    if np.any(~np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise SystemExit("synthesis returned non-positive or non-finite normalized flux")

    # jax_telluric works in ascending vacuum wavenumber.
    wavenumber_cm1 = 1.0e7 / wavelength_nm
    order = np.argsort(wavenumber_cm1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wavenumber_cm1=wavenumber_cm1[order],
        flux=normalized[order],
        wavelength_vacuum_nm=wavelength_nm,
        flux_total=np.asarray(spectrum.flux_total, dtype=float),
        flux_continuum=np.asarray(spectrum.flux_continuum, dtype=float),
    )

    metadata = {
        "star": "Arcturus",
        "labels": ARCTURUS_LABELS,
        "label_source": ARCTURUS_LABEL_SOURCE,
        "wavelength_nm": [args.wavelength_start_nm, args.wavelength_end_nm],
        "r_grid": args.r_grid,
        "points": int(wavelength_nm.size),
        "velocity_step_kms": float(
            np.median(np.diff(np.log(wavenumber_cm1[order]))) * 299792.458
        ),
        "normalized_flux_range": [float(normalized.min()), float(normalized.max())],
        "initializer_family": getattr(spectrum, "initializer_family", None),
        # False means the spectrum rests on the learned initializer atmosphere
        # rather than a converged solve. Recorded because it bounds how far the
        # stellar model can be trusted, and it is not visible in the arrays.
        "atmosphere_converged": bool(getattr(spectrum, "atmosphere_converged", False)),
        "atmosphere_closure_required": bool(
            getattr(spectrum, "atmosphere_closure_required", False)
        ),
        "seconds": float(getattr(spectrum, "seconds", float("nan"))),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
