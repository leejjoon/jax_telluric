#!/usr/bin/env python
"""Run the bootstrapped LBLRTM and save an instrument-resolution fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jax_telluric import (
    LBLRTMRunConfig,
    degrade_to_resolving_power,
    load_atmosphere_csv,
    run_lblrtm,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=float, default=5000.0)
    parser.add_argument("--v2", type=float, default=5100.0)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/example_midlatitude.csv"))
    parser.add_argument("--output", type=Path, default=Path("tests/data/lblrtm_k_5000_5100.npz"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    reference = root / "data/lblrtm"
    profile = load_atmosphere_csv(root / args.profile)
    run_arguments = (
        reference / "run_reference",
        profile,
    )
    run_paths = (
        reference / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl",
        reference / "run_lnfl_igrins/TAPE3",
        reference / "LBLRTM/data/absco-ref_wv-mt-ckd.nc",
    )
    # Flags 1--4 isolate H2O self and foreign continuum contributions while
    # retaining all line opacity and the other continuum terms.
    spectra = {}
    for name, continuum_flag in (("all", 1), ("no_self", 2), ("no_foreign", 3), ("no_water", 4)):
        monochromatic = run_lblrtm(
            *run_arguments,
            LBLRTMRunConfig(args.v1, args.v2, args.angle, continuum_flag),
            *run_paths,
        )
        spectra[name] = degrade_to_resolving_power(monochromatic)
    degraded = spectra["all"]
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        wavenumber_cm1=degraded.wavenumber_cm1,
        transmission=degraded.transmission,
        transmission_no_self=spectra["no_self"].transmission,
        transmission_no_foreign=spectra["no_foreign"].transmission,
        transmission_no_water_continuum=spectra["no_water"].transmission,
    )
    metadata = {
        "lblrtm": "12.17",
        "mt_ckd": "4.3",
        "aer_line_file": "3.9",
        "profile": str(args.profile),
        "wavenumber_cm1": [args.v1, args.v2],
        "zenith_angle_deg": args.angle,
        "resolving_power": 45000.0,
        "samples_per_resolution": 3.0,
        "continuum_flags": {"all": 1, "no_self": 2, "no_foreign": 3, "no_water": 4},
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output} with {len(degraded.wavenumber_cm1)} samples")


if __name__ == "__main__":
    main()
