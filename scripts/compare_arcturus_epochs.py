#!/usr/bin/env python
"""Locate the residual systematic by comparing the two atlas epochs (L5).

The summer and winter spectra of one page see the same star through different
atmospheres, separated by about 40 km/s of relative Doppler shift. That makes
the frame in which their residuals agree diagnostic:

    a systematic shared at the same TELLURIC rest wavenumber is atmospheric;
    a systematic shared at the same STELLAR rest wavenumber is the stellar model.

Each epoch is shifted into each candidate rest frame by its own fitted
velocity, so neither frame is handicapped by being left unaligned. The opposite
sign of each shift is also evaluated: an alignment that only works with the
correct sign is a real feature of the data, while one that works either way is
an artefact of the resampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_C_KMS = 299792.458


def _load(diagnostic: Path, report: Path):
    values = np.load(diagnostic)
    meta = json.loads(report.read_text())
    # The diagnostic arrays follow ascending wavelength, so wavenumber descends.
    order = np.argsort(values["wavenumber_cm1"])
    arrays = {
        key: (value[order] if isinstance(value, np.ndarray) and value.shape == order.shape else value)
        for key, value in values.items()
    }
    return arrays, meta


def _resample(source_cm1, values, mask, target_cm1, pixel_cm1):
    """Interpolate from good samples only, invalidating gaps wider than a pixel."""
    good_x, good_y = source_cm1[mask], values[mask]
    out = np.interp(target_cm1, good_x, good_y, left=np.nan, right=np.nan)
    nearest = np.abs(target_cm1[:, None] - good_x[None, :]).min(axis=1)
    return out, np.isfinite(out) & (nearest <= pixel_cm1)


def compare_in_frame(wavenumber_cm1, first, second, velocity_first, velocity_second, sign=1.0):
    """Put both epochs in one rest frame and measure how well they agree."""
    pixel = float(np.median(np.diff(wavenumber_cm1)))
    left = wavenumber_cm1 * (1.0 + sign * velocity_first / _C_KMS)
    right = wavenumber_cm1 * (1.0 + sign * velocity_second / _C_KMS)
    common = np.linspace(
        max(left.min(), right.min()), min(left.max(), right.max()), wavenumber_cm1.size
    )
    a, ok_a = _resample(left, first["residual"], first["mask"], common, pixel)
    b, ok_b = _resample(right, second["residual"], second["mask"], common, pixel)
    ok = ok_a & ok_b
    return {
        "pixels": int(ok.sum()),
        "correlation": float(np.corrcoef(a[ok], b[ok])[0, 1]),
        "difference_rms": float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))),
    }, common, a, b, ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--summer-npz", type=Path,
                        default=root / "benchmarks/results/arcturus_ab5000_star.npz")
    parser.add_argument("--summer-report", type=Path,
                        default=root / "docs/arcturus_ab5000_fit_star.json")
    parser.add_argument("--winter-npz", type=Path,
                        default=root / "benchmarks/results/arcturus_ab5000_winter.npz")
    parser.add_argument("--winter-report", type=Path,
                        default=root / "docs/arcturus_ab5000_winter.json")
    parser.add_argument("--output", type=Path, default=root / "docs/arcturus_two_epoch.json")
    args = parser.parse_args()

    summer, summer_meta = _load(args.summer_npz, args.summer_report)
    winter, winter_meta = _load(args.winter_npz, args.winter_report)
    wavenumber = summer["wavenumber_cm1"]
    if not np.allclose(wavenumber, winter["wavenumber_cm1"]):
        raise SystemExit("the two epochs must be fitted on the same pixel grid")

    telluric = (summer_meta["parameters"]["velocity_kms"], winter_meta["parameters"]["velocity_kms"])
    stellar = (summer_meta["parameters"]["stellar_velocity_kms"],
               winter_meta["parameters"]["stellar_velocity_kms"])

    frames = {}
    for name, (first, second) in (("telluric", telluric), ("stellar", stellar)):
        frames[name] = compare_in_frame(wavenumber, summer, winter, first, second)[0]
        frames[name]["velocities_kms"] = [first, second]
        # The sign control: a real alignment only works one way round.
        frames[name]["reversed_sign"] = compare_in_frame(
            wavenumber, summer, winter, first, second, sign=-1.0
        )[0]

    both = summer["mask"] & winter["mask"]
    sigma = 0.5 * (summer_meta["page"]["uncertainty"] + winter_meta["page"]["uncertainty"])
    pixel_kms = _C_KMS * float(np.median(np.diff(wavenumber))) / float(np.mean(wavenumber))

    report = {
        "mean_pixel_sigma": sigma,
        "pixel_kms": pixel_kms,
        "expected_difference_rms_if_noise": float(np.sqrt(2.0) * sigma),
        "unaligned_observer_frame": {
            "pixels": int(both.sum()),
            "correlation": float(np.corrcoef(summer["residual"][both], winter["residual"][both])[0, 1]),
            "difference_rms": float(np.sqrt(np.mean((summer["residual"] - winter["residual"])[both] ** 2))),
        },
        "frames": frames,
        "telluric_velocity_difference_kms": float(telluric[0] - telluric[1]),
        "telluric_velocity_difference_pixels": float(abs(telluric[0] - telluric[1]) / pixel_kms),
        "stellar_velocity_separation_kms": float(stellar[0] - stellar[1]),
        "summer_residual_rms": float(np.sqrt(np.mean(summer["residual"][both] ** 2))),
        "winter_residual_rms": float(np.sqrt(np.mean(winter["residual"][both] ** 2))),
        "note": (
            "A systematic shared at the same telluric rest wavenumber is atmospheric; one shared "
            "at the same stellar rest wavenumber is the stellar model. 'reversed_sign' is a null "
            "control: resampling alone would raise both signs equally."
        ),
    }
    report["dominant_frame"] = (
        "stellar model"
        if frames["stellar"]["correlation"] > frames["telluric"]["correlation"]
        else "atmosphere"
    )
    report["sign_control_passed"] = bool(
        frames["stellar"]["correlation"] > 3.0 * frames["stellar"]["reversed_sign"]["correlation"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"telluric velocities {telluric[0]:+.3f} / {telluric[1]:+.3f} km/s "
          f"= {report['telluric_velocity_difference_pixels']:.2f} pixel apart")
    print(f"{'frame':<12}{'correlation':>14}{'reversed':>12}{'difference rms':>18}")
    for name, values in frames.items():
        print(f"{name:<12}{values['correlation']:>14.4f}"
              f"{values['reversed_sign']['correlation']:>12.4f}{values['difference_rms']:>18.5f}")
    print(f"dominant: {report['dominant_frame']}, sign control "
          f"{'passed' if report['sign_control_passed'] else 'FAILED'}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
