#!/usr/bin/env python
"""Build an opt-in differentiable LBLRTM correction template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_telluric import (
    AERLineDatabase,
    ExoJAXOpacityBackend,
    LBLRTMOpticalDepthCorrection,
    read_tape12_single_precision,
    TelluricModel,
    TelluricParameters,
    build_lblrtm_correction,
    igrins_wavenumber_grid,
    load_atmosphere_csv,
)


MOLECULE_IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=float, default=5000.0)
    parser.add_argument("--v2", type=float, default=5020.0)
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/example_midlatitude.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/corrections/lblrtm_5000_5020.npz"))
    args = parser.parse_args()
    if not 0.0 < args.v1 < args.v2:
        parser.error("require 0 < v1 < v2")

    root = Path(__file__).resolve().parents[1]
    reference = root / "data/lblrtm"
    profile = load_atmosphere_csv(root / args.profile)
    grid = igrins_wavenumber_grid(1.0e7 / args.v2, 1.0e7 / args.v1)
    line_root = reference / "AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    databases = {}
    for species, molecule_id in MOLECULE_IDS.items():
        name = f"{molecule_id:02d}_{species}"
        try:
            databases[species] = AERLineDatabase(
                line_root / name / name, species, (float(grid[0]), float(grid[-1])), margin_cm1=0.0
            )
        except ValueError as exc:
            if not str(exc).startswith(f"no {species} lines found"):
                raise
    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse", vectorize_layers=True, mixed_precision=False
    )
    correction = build_lblrtm_correction(
        reference / "run_corrections",
        profile,
        grid,
        opacity,
        reference / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl",
        reference / "run_lnfl_igrins/TAPE3",
        reference / "LBLRTM/data/absco-ref_wv-mt-ckd.nc",
    )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    correction.save(output)

    # Round-trip and differentiate once so a generated artifact is known to
    # work through the public corrected-mode interface.
    loaded = LBLRTMOpticalDepthCorrection.load(output)
    model = TelluricModel(
        profile, grid, opacity, accuracy_mode="lblrtm_corrected", correction=loaded
    )
    species = model.species

    def transmission(log_scales):
        parameters = TelluricParameters(
            dict(zip(species, log_scales)), 0.0, 0.0, 3.0, jnp.asarray([0.0]), np.log(1.0e-5)
        )
        return model.transmission(parameters)

    value, jacobian = jax.jit(lambda p: (transmission(p), jax.jacfwd(transmission)(p)))(
        jnp.zeros(len(species))
    )
    if not np.all(np.isfinite(np.asarray(value))) or not np.all(np.isfinite(np.asarray(jacobian))):
        raise RuntimeError("generated correction failed the differentiability smoke test")

    reference_spectrum = read_tape12_single_precision(
        reference / "run_corrections/continuum_all/TAPE12"
    )
    reference_transmission = np.interp(
        grid, reference_spectrum.wavenumber_cm1, reference_spectrum.transmission
    )
    parameters = TelluricParameters(
        {name: 0.0 for name in species}, 0.0, 0.0, 3.0, jnp.asarray([0.0]), np.log(1.0e-5)
    )
    fast_model = TelluricModel(profile, grid, opacity)
    fast_transmission = np.asarray(fast_model.transmission(parameters))
    corrected_transmission = np.asarray(value)
    keep = reference_transmission > 0.05

    def error_summary(candidate):
        error = np.abs(candidate[keep] - reference_transmission[keep])
        return {
            "median_absolute": float(np.median(error)),
            "percentile_99_absolute": float(np.percentile(error, 99.0)),
            "maximum_absolute": float(np.max(error)),
        }

    fast_error = error_summary(fast_transmission)
    corrected_error = error_summary(corrected_transmission)
    if corrected_error["median_absolute"] >= fast_error["median_absolute"]:
        raise RuntimeError("LBLRTM correction did not improve median agreement")

    executable = reference / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl"
    tape3 = reference / "run_lnfl_igrins/TAPE3"
    mt_ckd_data = reference / "LBLRTM/data/absco-ref_wv-mt-ckd.nc"

    def reference_error(run_profile, parameters, name, zenith_angle_deg=0.0):
        from jax_telluric import LBLRTMRunConfig, run_lblrtm

        spectrum = run_lblrtm(
            reference / f"run_corrections/validation_{name}",
            run_profile,
            LBLRTMRunConfig(float(grid[0]), float(grid[-1]), zenith_angle_deg, 1),
            executable,
            tape3,
            mt_ckd_data,
        )
        expected = np.interp(grid, spectrum.wavenumber_cm1, spectrum.transmission)
        actual = np.asarray(model.transmission(parameters, zenith_angle_deg))
        valid = expected > 0.05
        error = np.abs(actual[valid] - expected[valid])
        return {
            "median_absolute": float(np.median(error)),
            "percentile_99_absolute": float(np.percentile(error, 99.0)),
            "maximum_absolute": float(np.max(error)),
        }

    h2o_scale_validation = {}
    for scale in (0.5, 2.0):
        scaled_vmr = dict(profile.vmr)
        scaled_vmr["H2O"] = np.asarray(profile.vmr["H2O"]) * scale
        scaled_profile = type(profile)(
            profile.pressure_edges_bar, profile.temperature_k, profile.altitude_km,
            scaled_vmr, profile.mean_molecular_weight_g_mol, profile.gravity_m_s2,
        )
        scaled_parameters = parameters._replace(
            log_column_scales={name: np.log(scale) if name == "H2O" else 0.0 for name in species}
        )
        h2o_scale_validation[str(scale)] = reference_error(
            scaled_profile, scaled_parameters, f"h2o_{scale:g}"
        )

    airmass_validation = {}
    for airmass in (1.5, 2.5):
        angle = float(np.degrees(np.arccos(1.0 / airmass)))
        airmass_validation[str(airmass)] = reference_error(
            profile, parameters, f"airmass_{airmass:g}", angle
        )
    stress_errors = [
        result["percentile_99_absolute"]
        for result in (*h2o_scale_validation.values(), *airmass_validation.values())
    ]
    if corrected_error["percentile_99_absolute"] >= 1.0e-3 or max(stress_errors) >= 5.0e-3:
        raise RuntimeError("generated correction missed its LBLRTM agreement thresholds")

    metadata = {
        "format": 1,
        "lblrtm": "12.17",
        "mt_ckd": "4.3",
        "aer_line_file": "3.9",
        "profile": str(args.profile),
        "requested_wavenumber_cm1": [args.v1, args.v2],
        "correction_grid_cm1": [float(grid[0]), float(grid[-1]), len(grid)],
        "species": list(species),
        "reference_error_fast": fast_error,
        "reference_error_corrected": corrected_error,
        "h2o_scale_validation": h2o_scale_validation,
        "airmass_validation": airmass_validation,
        "acceptance": {
            "reference_percentile_99_absolute": 1.0e-3,
            "stress_percentile_99_absolute": 5.0e-3
        },
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
