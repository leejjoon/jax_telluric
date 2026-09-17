#!/usr/bin/env python
"""Measure the FTS instrument line shape of the Arcturus IR atlas from its data.

An FTS spectrum is the Fourier transform of an interferogram truncated at the
maximum optical path difference, so the transform of a page *is* that
interferogram: its envelope is flat out to the MOPD and then cuts. Both the MOPD
and the apodization are therefore measurable, and neither has to be assumed.

The atlas documents neither. This script recovers them per page and per epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


ATLAS_ROOT = Path(
    "/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/arcturus/ir"
)
# Columns of the 89-character records; see the atlas table.doc.
_EPOCH_OBSERVED_COLUMN = {"summer": 1, "winter": 4}
# Full width at half maximum of sinc(x) = sin(x)/x, in units of 1/(2 L).
_BOXCAR_FWHM_CONSTANT = 1.20671


def _running_median(values: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    return np.asarray(
        [np.median(padded[index : index + width]) for index in range(len(values))]
    )


def interferogram_envelope(
    wavenumber_cm1: np.ndarray, flux: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return optical path difference in cm and the interferogram envelope."""

    spacing = float(np.median(np.diff(wavenumber_cm1)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("atlas wavenumbers must be increasing and evenly spaced")
    # The mean is a delta at zero path difference and the window suppresses the
    # leakage that would otherwise fill the region beyond the cut.
    windowed = (flux - np.mean(flux)) * np.hanning(len(flux))
    envelope = np.abs(np.fft.rfft(windowed))
    return np.fft.rfftfreq(len(flux), d=spacing), envelope


def measure_mopd(
    path_difference_cm: np.ndarray, envelope: np.ndarray, smoothing: int = 5
) -> dict:
    """Locate the truncation of the interferogram and describe its sharpness.

    The envelope does not fall to zero beyond the cut: the atlas stores five
    significant digits, and that quantization leaves a floor near 1e-5 of the
    peak. So the cut is found as the steepest sustained drop, not as the last
    sample above a noise threshold -- a threshold detector locks onto the floor
    and overestimates the MOPD by several centimetres.
    """

    peak = float(np.max(envelope))
    if peak <= 0.0:
        raise ValueError("degenerate interferogram envelope")
    logarithmic = np.log10(np.maximum(_running_median(envelope, smoothing), 1e-30) / peak)
    nyquist_cm = float(path_difference_cm[-1])
    spacing_cm = float(path_difference_cm[1])

    span = max(int(round(0.3 / spacing_cm)), 3)
    searchable = np.flatnonzero(
        (path_difference_cm > 2.0) & (path_difference_cm < nyquist_cm - 1.0)
    )
    searchable = searchable[searchable + span < len(logarithmic)]
    if len(searchable) < 3:
        raise ValueError("too few samples to locate a truncation")
    drops = logarithmic[searchable] - logarithmic[searchable + span]
    start = int(searchable[int(np.argmax(drops))])
    drop_decades = float(drops[int(np.argmax(drops))])

    in_band = float(
        np.median(logarithmic[(path_difference_cm > 2.0) & (path_difference_cm < path_difference_cm[start])])
    )
    beyond = (path_difference_cm > path_difference_cm[start] + 0.5) & (
        path_difference_cm < 0.95 * nyquist_cm
    )
    tail = float(np.median(logarithmic[beyond])) if np.any(beyond) else in_band - 3.0
    threshold = 0.5 * (in_band + tail)

    within = np.flatnonzero(
        (logarithmic > threshold) & (path_difference_cm < path_difference_cm[start] + 0.5)
    )
    mopd_cm = float(path_difference_cm[within[-1]]) if len(within) else float("nan")

    # An unapodized cut falls within a couple of samples. Norton-Beer tapers
    # over roughly L/3, so the transition width separates the two.
    taper = np.flatnonzero(
        (logarithmic < in_band - 0.5)
        & (logarithmic > tail + 0.5)
        & (path_difference_cm > path_difference_cm[start] - 0.5)
        & (path_difference_cm < path_difference_cm[start] + 1.0)
    )
    transition_cm = (
        float(path_difference_cm[taper[-1]] - path_difference_cm[taper[0]])
        if len(taper) > 1
        else spacing_cm
    )

    return {
        "mopd_cm": mopd_cm,
        "drop_decades": drop_decades,
        "in_band_log10": in_band,
        "tail_log10": tail,
        "transition_cm": transition_cm,
        "path_difference_resolution_cm": spacing_cm,
        "nyquist_path_difference_cm": nyquist_cm,
    }


def measure_page(path: Path, epoch: str) -> dict:
    values = np.loadtxt(path)
    wavenumber = values[:, 0]
    flux = values[:, _EPOCH_OBSERVED_COLUMN[epoch]]
    path_difference_cm, envelope = interferogram_envelope(wavenumber, flux)
    result = measure_mopd(path_difference_cm, envelope)

    center = float(0.5 * (wavenumber[0] + wavenumber[-1]))
    mopd = result["mopd_cm"]
    result.update(
        {
            "page": path.name,
            "epoch": epoch,
            "wavenumber_min_cm1": float(wavenumber[0]),
            "wavenumber_max_cm1": float(wavenumber[-1]),
            "samples": int(len(wavenumber)),
            "spacing_cm1": float(np.median(np.diff(wavenumber))),
        }
    )
    if np.isfinite(mopd) and mopd > 0.0:
        fwhm = _BOXCAR_FWHM_CONSTANT / (2.0 * mopd)
        result["ils_fwhm_cm1"] = float(fwhm)
        result["resolving_power_at_center"] = float(center / fwhm)
        result["samples_per_fwhm"] = float(fwhm / result["spacing_cm1"])
        # Beyond this the MOPD is longer than the sampling can express and the
        # measurement only recovers the decimation, not the interferogram.
        result["mopd_measurable"] = bool(mopd < 0.95 * result["nyquist_path_difference_cm"])
    else:
        result["mopd_measurable"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=ATLAS_ROOT)
    parser.add_argument("--pages", nargs="*", default=None, help="page file names")
    parser.add_argument("--output", type=Path, default=Path("docs/arcturus_ils.json"))
    args = parser.parse_args()

    if args.pages:
        pages = [args.atlas_root / name for name in args.pages]
    else:
        pattern = re.compile(r"^ab\d+[._]?$")
        pages = sorted(p for p in args.atlas_root.iterdir() if pattern.match(p.name))
    if not pages:
        raise SystemExit(f"no atlas pages found under {args.atlas_root}")

    measurements = []
    for page in pages:
        for epoch in ("summer", "winter"):
            try:
                measurements.append(measure_page(page, epoch))
            except (ValueError, IndexError) as exc:
                measurements.append({"page": page.name, "epoch": epoch, "error": str(exc)})

    usable = [
        m for m in measurements if m.get("mopd_measurable") and np.isfinite(m.get("mopd_cm", np.nan))
    ]
    summary = {}
    for epoch in ("summer", "winter"):
        values = [m["mopd_cm"] for m in usable if m["epoch"] == epoch]
        if values:
            summary[epoch] = {
                "pages": len(values),
                "median_mopd_cm": float(np.median(values)),
                "mad_mopd_cm": float(np.median(np.abs(np.asarray(values) - np.median(values)))),
                "median_transition_cm": float(
                    np.median([m["transition_cm"] for m in usable if m["epoch"] == epoch])
                ),
            }

    report = {
        "atlas_root": str(args.atlas_root),
        "method": (
            "Hann-windowed real FFT of the observed column; MOPD located at the "
            "steepest sustained drop of the smoothed log envelope."
        ),
        "boxcar_fwhm_constant": _BOXCAR_FWHM_CONSTANT,
        "summary": summary,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for epoch, values in summary.items():
        print(
            f"{epoch}: MOPD = {values['median_mopd_cm']:.2f} +- {values['mad_mopd_cm']:.2f} cm "
            f"over {values['pages']} pages, transition {values['median_transition_cm']:.2f} cm"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
