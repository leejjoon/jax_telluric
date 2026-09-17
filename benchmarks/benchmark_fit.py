#!/usr/bin/env python
"""Measure what a staged telluric fit costs, and what precomputing saves.

The forward-model benchmark in ``benchmark.py`` times one evaluation. What a
fit actually spends is different: XLA compilation happens once per distinct
jitted function and costs seconds, while one evaluation costs milliseconds, so
a four-stage fit that recompiles per stage is dominated by compilation.

This runs the same synthetic page three ways -- recompiling per stage, sharing
one compilation, and additionally precomputing the opacity -- and records both
the timings and the disagreement between the fitted parameters. Needs the AER
line files and the MT_CKD file from ``bootstrap_lblrtm.sh``, not the binary.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_module
import statistics
import time
from pathlib import Path

MOLECULE_IDS = {"H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
STAGES = ("continuum", "velocity", "columns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--v1", type=float, default=5005.0)
    parser.add_argument("--v2", type=float, default=5025.0)
    parser.add_argument("--species", default="H2O,CO2")
    parser.add_argument("--pixels", type=int, default=1001)
    parser.add_argument("--resolving-power", type=float, default=100_000.0)
    parser.add_argument("--samples-per-resolution", type=float, default=4.0)
    parser.add_argument("--profile", type=Path, default=root / "data/profiles/kitt_peak_1994.csv")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=root / "docs/precomputed_opacity_results.json")
    args = parser.parse_args()

    # Must be selected before importing JAX or jax_telluric.
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jax_telluric import (
        AERLineDatabase, ExoJAXOpacityBackend, MTCKDWaterContinuum, OrderObjective,
        SpectralOrder, TelluricModel, TelluricParameters, fit_order,
        igrins_wavenumber_grid, load_atmosphere_csv,
    )

    species = [name.strip().upper() for name in args.species.split(",") if name.strip()]
    grid = igrins_wavenumber_grid(
        1.0e7 / args.v2, 1.0e7 / args.v1, resolving_power=args.resolving_power,
        samples_per_resolution=args.samples_per_resolution, margin_cm1=25.0,
    )
    profile = load_atmosphere_csv(args.profile)
    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    databases, lines = {}, {}
    for name in species:
        stem = f"{MOLECULE_IDS[name]:02d}_{name}"
        databases[name] = AERLineDatabase(
            line_root / stem / stem, name, (float(grid[0]), float(grid[-1])), margin_cm1=0.0
        )
        lines[name] = int(np.asarray(databases[name].nu_lines).size)
    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=True, mixed_precision=True, pressure_shift=True,
    )
    continuum = MTCKDWaterContinuum.from_netcdf(
        root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", grid
    )
    model = TelluricModel(profile, grid, opacity, continuum=continuum, accuracy_mode="mt_ckd",
                          max_lsf_sigma_kms=4.0, pixel_integration="point")

    wavelength = 1.0e7 / np.linspace(args.v1, args.v2, args.pixels)[::-1]
    uncertainty = np.full(args.pixels, 0.0055)
    degree = 3
    truth = TelluricParameters(
        log_column_scales={name: value for name, value in zip(species, (0.35, -0.12, 0.0, 0.0))},
        velocity_kms=0.3, wavelength_stretch=0.0, lsf_sigma_kms=1.2,
        continuum_coeffs=np.array([0.02, -0.01, 0.005, 0.0]), log_jitter=-9.0,
    )
    blank = SpectralOrder(wavelength, np.ones(args.pixels), uncertainty,
                          source_flux=np.ones(args.pixels))
    observed = np.asarray(model.predict(blank, truth)) + np.random.default_rng(7).normal(
        0.0, 0.0055, args.pixels
    )
    order = SpectralOrder(wavelength, observed, uncertainty, source_flux=np.ones(args.pixels))

    start = TelluricParameters(
        log_column_scales={name: 0.0 for name in species}, velocity_kms=0.0,
        wavelength_stretch=0.0, lsf_sigma_kms=1.0,
        continuum_coeffs=np.zeros(degree + 1), log_jitter=-9.0,
    )

    def bounds_for(stage: str, current: TelluricParameters) -> dict:
        pin = lambda value: (float(value), float(value))
        free_columns = stage == "columns"
        free_velocity = stage in ("velocity", "columns")
        bounds = {"wavelength_stretch": (0.0, 0.0), "log_jitter": (-12.0, -4.0)}
        for name in species:
            bounds[name] = (-2.0, 2.0) if free_columns else pin(current.log_column_scales[name])
        bounds["velocity_kms"] = (-10.0, 10.0) if free_velocity else pin(current.velocity_kms)
        bounds["lsf_sigma_kms"] = (0.05, 4.0) if free_velocity else pin(current.lsf_sigma_kms)
        for index in range(degree + 1):
            bounds[f"continuum_{index}"] = (-0.5, 0.5)
        return bounds

    def staged(fit_model, share: bool) -> tuple[float, TelluricParameters, int]:
        started = time.perf_counter()
        shared = OrderObjective(fit_model, order, degree + 1) if share else None
        current, iterations = start, 0
        for stage in STAGES:
            result = fit_order(fit_model, order, current, bounds_for(stage, current), objective=shared)
            current, iterations = result.parameters, iterations + result.iterations
        return time.perf_counter() - started, current, iterations

    def evaluation_cost(fit_model) -> dict:
        objective = OrderObjective(fit_model, order, degree + 1)
        vector = objective.codec.pack(truth)
        started = time.perf_counter()
        objective(vector)
        compile_s = time.perf_counter() - started
        for _ in range(2):
            objective(vector)
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            objective(vector)
            samples.append(time.perf_counter() - started)
        return {"compile_and_first_s": compile_s,
                "value_and_gradient_median_ms": 1000.0 * statistics.median(samples)}

    started = time.perf_counter()
    precomputed = model.precompute_opacity(self_broadening="linear")
    precompute_s = time.perf_counter() - started

    report = {
        "platform": args.platform,
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "python": platform_module.python_version(),
        "case": {
            "wavenumber_cm1": [args.v1, args.v2],
            "species": species, "lines": lines,
            "atmospheric_layers": int(len(profile.temperature_k)),
            "high_resolution_samples": int(grid.size),
            "detector_pixels": args.pixels,
            "accuracy_mode": "mt_ckd", "stages": list(STAGES),
        },
        "evaluation": {"live_opacity": evaluation_cost(model),
                       "precomputed_opacity": evaluation_cost(precomputed)},
        "precompute_build_s": precompute_s,
    }

    runs = {}
    for label, fit_model, share in (("compile_per_stage", model, False),
                                    ("shared_compilation", model, True),
                                    ("precomputed_and_shared", precomputed, True)):
        seconds, parameters, iterations = staged(fit_model, share)
        # Every run is scored through the exact model, never its own.
        flux = np.asarray(model.predict(order, parameters))
        runs[label] = {
            "seconds": seconds, "iterations": iterations,
            "residual_rms": float(np.sqrt(np.mean((observed - flux) ** 2))),
            "parameters": {
                **{name: float(parameters.log_column_scales[name]) for name in species},
                "velocity_kms": float(parameters.velocity_kms),
                "lsf_sigma_kms": float(parameters.lsf_sigma_kms),
                "log_jitter": float(parameters.log_jitter),
            },
            "_flux": flux,
        }
    reference = runs["compile_per_stage"]
    for label, run in runs.items():
        difference = run.pop("_flux") - reference["_flux"] if label != "compile_per_stage" else None
        if difference is not None:
            run["flux_vs_reference"] = {"max": float(np.max(np.abs(difference))),
                                        "rms": float(np.sqrt(np.mean(difference**2)))}
            run["parameter_relative_difference"] = {
                name: abs(value - reference["parameters"][name]) / max(abs(reference["parameters"][name]), 1e-12)
                for name, value in run["parameters"].items()
            }
        run["speedup_over_reference"] = reference["seconds"] / run["seconds"]
    reference.pop("_flux", None)
    total = runs["precomputed_and_shared"]["seconds"] + precompute_s
    report["staged_fit"] = runs
    report["end_to_end_speedup"] = reference["seconds"] / total
    report["truth"] = {name: float(truth.log_column_scales[name]) for name in species}

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "staged_fit"}, indent=2))
    for label, run in runs.items():
        print(f"{label:24s} {run['seconds']:7.2f} s  {run['iterations']:4d} it  "
              f"speedup {run['speedup_over_reference']:5.2f}x")
    print(f"end to end, including the {precompute_s:.2f} s precompute: "
          f"{report['end_to_end_speedup']:.1f}x")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
