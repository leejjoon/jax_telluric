#!/usr/bin/env python
"""Validate the native JAX MT_CKD continuum against LBLRTM 12.17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jax_telluric import (
    AtmosphereProfile,
    LBLRTMRunConfig,
    MTCKDWaterContinuum,
    run_lblrtm,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=float, default=5000.0)
    parser.add_argument("--v2", type=float, default=5020.0)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/native_mt_ckd_validation.json")
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    lblrtm_root = root / "data/lblrtm"
    executable = lblrtm_root / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
    tape3 = lblrtm_root / "run_lnfl_igrins/TAPE3"
    coefficient_file = lblrtm_root / "LBLRTM/data/absco-ref_wv-mt-ckd.nc"
    states = (
        ("cold_low_pressure", (0.05, 0.20), 225.0, 0.002),
        ("temperate_mid_pressure", (0.30, 0.60), 255.0, 0.004),
        ("warm_high_pressure", (0.70, 0.90), 285.0, 0.010),
    )

    results = []
    for name, pressure_edges, temperature, water_vmr in states:
        profile = AtmosphereProfile(
            pressure_edges,
            [temperature],
            [5.0],
            {"H2O": [water_vmr]},
        )
        spectra = {}
        for label, continuum_flag in (("all", 1), ("no_water", 4)):
            spectra[label] = run_lblrtm(
                lblrtm_root / f"native_mt_ckd_validation/{name}_{label}",
                profile,
                LBLRTMRunConfig(
                    args.v1,
                    args.v2,
                    continuum_flag=continuum_flag,
                    description=f"native MT_CKD validation: {name}",
                ),
                executable,
                tape3,
                coefficient_file,
            )
        full = spectra["all"]
        no_water = spectra["no_water"]
        reference_tau = np.log(
            np.clip(no_water.transmission, 1.0e-30, None)
            / np.clip(full.transmission, 1.0e-30, None)
        )
        continuum = MTCKDWaterContinuum.from_netcdf(
            coefficient_file, full.wavenumber_cm1
        )
        native_tau = np.asarray(
            continuum.optical_depth(profile, {"H2O": np.asarray([water_vmr])})
        ).sum(axis=0)
        valid = (
            (full.transmission > 1.0e-5)
            & (no_water.transmission > 1.0e-5)
            & (reference_tau > 1.0e-7)
        )
        absolute_error = np.abs(native_tau[valid] - reference_tau[valid])
        relative_error = absolute_error / reference_tau[valid]
        results.append(
            {
                "name": name,
                "pressure_edges_bar": list(pressure_edges),
                "temperature_k": temperature,
                "h2o_vmr": water_vmr,
                "samples": int(np.count_nonzero(valid)),
                "reference_optical_depth_range": [
                    float(np.min(reference_tau[valid])),
                    float(np.max(reference_tau[valid])),
                ],
                "median_relative_error": float(np.median(relative_error)),
                "percentile_99_relative_error": float(np.percentile(relative_error, 99)),
                "maximum_relative_error": float(np.max(relative_error)),
                "maximum_absolute_error": float(np.max(absolute_error)),
            }
        )

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lblrtm": "12.17",
        "mt_ckd": "4.3",
        "wavenumber_cm1": [args.v1, args.v2],
        "method": "difference of LBLRTM continuum flags 1 and 4",
        "cases": results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
