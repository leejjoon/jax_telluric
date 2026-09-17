#!/usr/bin/env python
"""Synthetic checks on the Arcturus telluric fit, before any real data is trusted.

Three questions, each answered by generating data from a known truth and fitting
it back:

L0  Can the fitter recover its own parameters at the real noise level?
L1  Is the internal grid fine enough, or does it bias the retrieved columns?
L2  How much does the Gaussian line spread function cost against the measured
    unapodized FTS sinc?

L1 and L2 generate one reference spectrum on the finest grid with the sinc, then
fit it with each candidate configuration, so the comparison isolates the model
choice rather than re-deriving the truth each time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from jax_telluric import (
    AERLineDatabase,
    BoxcarFTSInstrumentProfile,
    ExoJAXOpacityBackend,
    MTCKDWaterContinuum,
    SpectralOrder,
    StellarSpectrum,
    TelluricModel,
    TelluricParameters,
    fit_order,
    igrins_wavenumber_grid,
    load_atmosphere_csv,
    prepare_stellar_source,
)

MOLECULE_IDS = {"H2O": 1, "CO2": 2}
TRUTH = {"H2O": np.log(1.20), "CO2": np.log(1.15)}
TRUTH_VELOCITY_KMS = 0.30
TRUTH_LSF_KMS = 0.40
TRUTH_CONTINUUM = np.array([-0.02, -0.03, 0.02, -0.03])
NOISE = 0.005
MOPD_CM = 15.167


def build(profile, root, v1, v2, margin, resolving_power, samples, gaussian_ils, chunk):
    grid = igrins_wavenumber_grid(
        1.0e7 / v2, 1.0e7 / v1, resolving_power=resolving_power,
        samples_per_resolution=samples, margin_cm1=margin,
    )
    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    databases = {}
    for species, molecule_id in MOLECULE_IDS.items():
        name = f"{molecule_id:02d}_{species}"
        databases[species] = AERLineDatabase(
            line_root / name / name, species, (float(grid[0]), float(grid[-1])), margin_cm1=0.0
        )
    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=True, mixed_precision=True, pressure_shift=True,
        layer_chunk_size=chunk,
    )
    continuum = MTCKDWaterContinuum.from_netcdf(
        root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", grid
    )
    instrument = None if gaussian_ils else BoxcarFTSInstrumentProfile(
        mopd_cm=MOPD_CM, wavenumber_center_cm1=float(0.5 * (v1 + v2)), max_residual_sigma_kms=4.0
    )
    model = TelluricModel(
        profile, grid, opacity, continuum=continuum, accuracy_mode="mt_ckd",
        max_lsf_sigma_kms=4.0, pixel_integration="point", instrument=instrument,
    )
    return model, grid


def truth_parameters(species):
    return TelluricParameters(
        log_column_scales={name: TRUTH[name] for name in species},
        velocity_kms=TRUTH_VELOCITY_KMS,
        wavelength_stretch=0.0,
        lsf_sigma_kms=TRUTH_LSF_KMS,
        continuum_coeffs=TRUTH_CONTINUUM,
        log_jitter=np.log(1.0e-4),
    )


def fit_bounds(species, pinned_jitter):
    fixed = 0.0
    bounds = {name: (-2.0, 2.0) for name in species}
    bounds.update({
        "velocity_kms": (-5.0, 5.0),
        "wavelength_stretch": (fixed, fixed),
        "lsf_sigma_kms": (0.05, 4.0),
        "log_jitter": (pinned_jitter, pinned_jitter),
    })
    for index in range(len(TRUTH_CONTINUUM)):
        bounds[f"continuum_{index}"] = (-2.0, 2.0) if index == 0 else (-0.5, 0.5)
    return bounds


def run_fit(model, wavelength, flux, uncertainty):
    order = SpectralOrder(wavelength, flux, np.full(wavelength.size, uncertainty))
    start = TelluricParameters(
        log_column_scales={name: 0.0 for name in model.species},
        velocity_kms=0.0, wavelength_stretch=0.0, lsf_sigma_kms=0.5,
        continuum_coeffs=np.zeros(len(TRUTH_CONTINUUM)), log_jitter=np.log(uncertainty),
    )
    return fit_order(model, order, start, fit_bounds(model.species, float(np.log(uncertainty))))


def summarize(result, model, uncertainty):
    covariance = result.covariance
    names = list(model.species)
    errors = {}
    if covariance is not None:
        for index, name in enumerate(names):
            errors[name] = float(np.sqrt(max(covariance[index, index], 0.0)))
    residual = np.asarray(result.residuals)
    return {
        "success": bool(result.success),
        "message": result.message,
        **{f"{name}_scale": float(np.exp(result.parameters.log_column_scales[name])) for name in names},
        **{f"{name}_error": errors.get(name, float("nan")) for name in names},
        "velocity_kms": float(result.parameters.velocity_kms),
        "lsf_sigma_kms": float(result.parameters.lsf_sigma_kms),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "rms_over_noise": float(np.sqrt(np.mean(residual**2)) / uncertainty),
        "reduced_chi2": float(np.mean((residual / uncertainty) ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=float, default=5005.0)
    parser.add_argument("--v2", type=float, default=5025.0)
    parser.add_argument("--margin-cm1", type=float, default=25.0)
    parser.add_argument("--pixels", type=int, default=1001)
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/kitt_peak_1994.csv"))
    parser.add_argument("--layer-chunk-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=Path("docs/arcturus_validation.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    profile = load_atmosphere_csv(root / args.profile)
    wavelength = np.linspace(1.0e7 / args.v2, 1.0e7 / args.v1, args.pixels)
    rng = np.random.default_rng(args.seed)
    started = time.time()

    common = dict(v1=args.v1, v2=args.v2, margin=args.margin_cm1, chunk=args.layer_chunk_size)

    # The reference: generated on the finest grid with the measured sinc.
    reference_model, _ = build(profile, root, resolving_power=200_000.0, samples=4.0,
                               gaussian_ils=False, **common)
    truth = truth_parameters(reference_model.species)
    blank = SpectralOrder(wavelength, np.ones(args.pixels), np.full(args.pixels, NOISE))
    clean = np.asarray(reference_model.predict(blank, truth))
    noisy = clean + rng.normal(0.0, NOISE, clean.size)
    print(f"reference generated on {reference_model.wavenumber_cm1.size} points "
          f"({reference_model.velocity_step_kms:.3f} km/s)")

    report = {
        "window_cm1": [args.v1, args.v2],
        "pixels": args.pixels,
        "noise": NOISE,
        "truth": {
            **{f"{name}_scale": float(np.exp(value)) for name, value in TRUTH.items()},
            "velocity_kms": TRUTH_VELOCITY_KMS,
            "lsf_sigma_kms": TRUTH_LSF_KMS,
        },
        "reference": {
            "resolving_power": 200_000.0, "samples_per_resolution": 4.0,
            "points": int(reference_model.wavenumber_cm1.size),
            "velocity_step_kms": reference_model.velocity_step_kms,
            "instrument": "boxcar FTS sinc", "mopd_cm": MOPD_CM,
        },
        "cases": [],
    }

    configurations = [
        ("L1_45k_x4", 45_000.0, 4.0, False),
        ("L1_100k_x4", 100_000.0, 4.0, False),
        ("L1_200k_x4", 200_000.0, 4.0, False),
        ("L2_100k_x4_gaussian", 100_000.0, 4.0, True),
    ]
    for label, resolving_power, samples, gaussian in configurations:
        model, grid = build(profile, root, resolving_power=resolving_power, samples=samples,
                            gaussian_ils=gaussian, **common)
        case_started = time.time()
        result = run_fit(model, wavelength, noisy, NOISE)
        case = {
            "case": label,
            "resolving_power": resolving_power,
            "samples_per_resolution": samples,
            "points": int(grid.size),
            "velocity_step_kms": model.velocity_step_kms,
            "instrument": "gaussian" if gaussian else "boxcar FTS sinc",
            "seconds": round(time.time() - case_started, 1),
            **summarize(result, model, NOISE),
        }
        report["cases"].append(case)
        print(f"[{label}] {case['points']:5d} pts  H2O {case['H2O_scale']:.4f}  "
              f"CO2 {case['CO2_scale']:.4f}  rms/noise {case['rms_over_noise']:.3f}  "
              f"({case['seconds']} s)")

    by_label = {case["case"]: case for case in report["cases"]}

    # L0: the finest grid with the correct instrument must recover the truth.
    # The criterion is a fractional error, not a pull: L-BFGS-B's inverse
    # Hessian is a byproduct of its line search, not a covariance, and here it
    # overestimates the column errors by more than two orders of magnitude
    # (34-38% against a measured scatter near 0.1%). Those errors are reported
    # for the record and must not be quoted as uncertainties.
    l0 = by_label["L1_200k_x4"]
    l0_checks = {"note": "formal errors from L-BFGS-B hess_inv are unreliable; see the module docstring"}
    for name, value in TRUTH.items():
        l0_checks[f"{name}_fractional_error"] = float(
            (l0[f"{name}_scale"] - np.exp(value)) / np.exp(value)
        )
        l0_checks[f"{name}_formal_error"] = l0[f"{name}_error"]
    l0_checks["velocity_error_kms"] = float(l0["velocity_kms"] - TRUTH_VELOCITY_KMS)
    l0_checks["reduced_chi2"] = l0["reduced_chi2"]
    l0_checks["threshold_fractional_error"] = 0.01
    l0_checks["passed"] = bool(
        all(
            abs(l0_checks[f"{name}_fractional_error"]) <= 0.01 for name in TRUTH
        )
        and abs(l0["reduced_chi2"] - 1.0) <= 0.15
        and abs(l0_checks["velocity_error_kms"]) <= 0.1
    )
    report["L0_injection_recovery"] = l0_checks

    # L1: the two finest grids must agree to a fixed fraction. Comparing against
    # the formal error would be no test at all, for the reason given in L0.
    fine, finest = by_label["L1_100k_x4"], by_label["L1_200k_x4"]
    coarse = by_label["L1_45k_x4"]
    tolerance = 0.005
    l1 = {"threshold_fractional_shift": tolerance}
    for name in TRUTH:
        shift = abs(fine[f"{name}_scale"] - finest[f"{name}_scale"]) / finest[f"{name}_scale"]
        l1[name] = {
            "fractional_shift_100k_vs_200k": float(shift),
            "converged": bool(shift <= tolerance),
            "fractional_shift_45k_vs_200k": float(
                abs(coarse[f"{name}_scale"] - finest[f"{name}_scale"]) / finest[f"{name}_scale"]
            ),
        }
    l1["coarse_45k_rms_over_noise"] = coarse["rms_over_noise"]
    l1["passed"] = bool(all(l1[name]["converged"] for name in TRUTH))
    report["L1_grid_convergence"] = l1

    # L2: a Gaussian standing in for the measured sinc.
    gaussian_case = by_label["L2_100k_x4_gaussian"]
    sinc_case = by_label["L1_100k_x4"]
    report["L2_instrument_profile"] = {
        "gaussian_rms_over_noise": gaussian_case["rms_over_noise"],
        "sinc_rms_over_noise": sinc_case["rms_over_noise"],
        "gaussian_penalty": float(gaussian_case["rms_over_noise"] / sinc_case["rms_over_noise"]),
        **{
            f"{name}_bias_vs_sinc": float(
                (gaussian_case[f"{name}_scale"] - sinc_case[f"{name}_scale"]) / sinc_case[f"{name}_scale"]
            )
            for name in TRUTH
        },
        "threshold_rms_over_noise": 1.2,
        "passed": bool(gaussian_case["rms_over_noise"] <= 1.2),
    }

    report["runtime_seconds"] = round(time.time() - started, 1)
    (root / args.output).parent.mkdir(parents=True, exist_ok=True)
    (root / args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key in ("L0_injection_recovery", "L1_grid_convergence", "L2_instrument_profile"):
        print(f"{key}: passed={report[key]['passed']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
