#!/usr/bin/env python
"""Divide a fitted telluric transmission out of an observed page.

Writes the corrected spectrum alongside everything needed to judge it: the
transmission that was divided out, a quality flag, the stellar model the fit
used, and the atlas authors' own corrected column for comparison.

Two cautions are built into the output rather than left to the reader.

The correction is a division by a *convolved* transmission, which is the usual
approximation and is not exact: the instrument profile acts on the product of
star and atmosphere, not on each separately. The error grows with line depth.

Where the atmosphere is opaque there is no signal to recover and the division
amplifies noise without bound. Those pixels are flagged rather than dropped, so
that nothing downstream mistakes a ratio of two small numbers for a measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def chebyshev_continuum(coefficients: np.ndarray, count: int) -> np.ndarray:
    """Reproduce TelluricModel.predict's exp(Chebyshev) continuum."""
    x = np.linspace(-1.0, 1.0, count)
    total = np.zeros(count)
    t0, t1 = np.ones(count), x
    for index, value in enumerate(np.asarray(coefficients)):
        if index == 0:
            total = total + value * t0
        elif index == 1:
            total = total + value * t1
        else:
            t2 = 2.0 * x * t1 - t0
            total = total + value * t2
            t0, t1 = t1, t2
    return np.exp(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--diagnostic", type=Path,
                        default=root / "benchmarks/results/arcturus_ab5000_star.npz")
    parser.add_argument("--report", type=Path,
                        default=root / "docs/arcturus_ab5000_fit_star.json")
    parser.add_argument("--min-transmission", type=float, default=0.15,
                        help="below this the correction is flagged as unreliable")
    parser.add_argument("--output", type=Path,
                        default=root / "data/corrected/arcturus_ab5000_summer_corrected.npz")
    args = parser.parse_args()

    values = np.load(args.diagnostic)
    meta = json.loads(args.report.read_text())
    order = np.argsort(values["wavenumber_cm1"])
    take = lambda key: values[key][order]

    wavenumber = take("wavenumber_cm1")
    flux, mask = take("flux"), take("mask")
    transmission = take("transmission_pixels")
    continuum = chebyshev_continuum(
        np.asarray(meta["parameters"]["continuum_coeffs"]), flux.size
    )[order]

    safe = np.maximum(transmission, 1.0e-6)
    # The naive correction, kept for comparison. It divides by a *convolved*
    # transmission, and Conv(T x S) / Conv(T) is not S, so it leaves a
    # derivative-shaped artefact beside every sharp line that 1/T then
    # amplifies. Measured on this page, 86% of that excursion is reproduced by
    # applying the same division to the model alone.
    naive = flux / safe
    normalized_naive = naive / continuum

    model_flux = take("model_flux")
    if "stellar_only_pixels" in values:
        # Correct by the model ratio: transfer the data's fractional departure
        # from the model onto the clean stellar spectrum. Nothing is divided by
        # a convolved transmission, and where the fit is exact this returns the
        # star exactly.
        star_only = take("stellar_only_pixels")
        corrected = (flux / np.maximum(model_flux, 1.0e-6)) * star_only
        method = "model ratio: (observed / model) * star"
    else:
        corrected = naive
        star_only = np.full_like(flux, np.nan)
        method = "naive: observed / transmission (no star-only prediction available)"
    normalized = corrected / continuum

    reliable = mask & (transmission >= args.min_transmission)
    # Propagating the pixel noise through the division is the honest error bar:
    # it diverges exactly where the transmission does not.
    uncertainty = take("uncertainty") / safe / continuum

    np.savez_compressed(
        args.output,
        wavenumber_cm1=wavenumber,
        # NaN at masked pixels, so plotting cannot resurrect the placeholder.
        observed_plot=np.where(mask, flux, np.nan),
        corrected_plot=np.where(reliable, normalized, np.nan),
        wavelength_vacuum_nm=1.0e7 / wavenumber,
        observed=flux,
        transmission=transmission,
        continuum=continuum,
        corrected=corrected,
        corrected_normalized=normalized,
        corrected_naive=naive,
        corrected_naive_normalized=normalized_naive,
        stellar_only=star_only / continuum,
        corrected_uncertainty=uncertainty,
        reliable=reliable,
        stellar_model_pixel_grid=model_flux / np.maximum(transmission * continuum, 1e-6),
        atlas_ratioed=take("atlas_ratioed"),
        atlas_telluric=take("atlas_telluric"),
    )

    good = reliable
    summary = {
        "source": str(args.diagnostic),
        "page": meta["page"]["name"], "epoch": meta["page"]["epoch"],
        "wavenumber_cm1": [float(wavenumber[0]), float(wavenumber[-1])],
        "pixels": int(flux.size),
        "reliable_pixels": int(good.sum()),
        "flagged_pixels": int((~good).sum()),
        "method": method,
        "min_transmission": args.min_transmission,
        "median_transmission": float(np.median(transmission)),
        "median_corrected_uncertainty": float(np.median(uncertainty[good])),
        "worst_corrected_uncertainty": float(np.max(uncertainty[good])),
        "caution": (
            "Flagged pixels have no recoverable signal. 'corrected_naive' is the plain division "
            "by the transmission and carries a derivative-shaped artefact at every sharp line; "
            "prefer 'corrected'."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
