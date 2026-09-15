#!/usr/bin/env python
"""Run the first validation-ladder case: AER CO lines in both engines."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from jax_telluric import (
    AERLineDatabase,
    AtmosphereProfile,
    ExoJAXOpacityBackend,
    LBLRTMRunConfig,
    LBLRTMSpectrum,
    TelluricModel,
    TelluricParameters,
    compare_transmission,
    degrade_to_resolving_power,
    igrins_wavenumber_grid,
    load_atmosphere_csv,
    run_lblrtm,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reference_root = root / "data/lblrtm"
    source = load_atmosphere_csv(root / "data/profiles/example_midlatitude.csv")
    profile = AtmosphereProfile(
        source.pressure_edges_bar,
        source.temperature_k,
        source.altitude_km,
        {"CO": source.vmr["CO"]},
    )
    limits = (4290.0, 4310.0)
    lblrtm = run_lblrtm(
        reference_root / "run_compare_co",
        profile,
        LBLRTMRunConfig(*limits, continuum_flag=0, description="CO-only comparison"),
        reference_root / "LBLRTM/lblrtm_v12.17_linux_gnu_sgl",
        reference_root / "run_lnfl_igrins/TAPE3",
        reference_root / "LBLRTM/data/absco-ref_wv-mt-ckd.nc",
    )
    nu_grid = igrins_wavenumber_grid(
        1.0e7 / limits[1],
        1.0e7 / limits[0],
        resolving_power=120_000.0,
        samples_per_resolution=4.0,
    )
    database = AERLineDatabase(
        reference_root / "AER_Line_File/aer_v_3.9/line_files_By_Molecule/05_CO/05_CO",
        "CO",
        limits,
    )
    backend = ExoJAXOpacityBackend.prepare({"CO": database}, nu_grid, methods="direct")
    model = TelluricModel(profile, nu_grid, backend)
    parameters = TelluricParameters(
        {"CO": 0.0}, 0.0, 0.0, 2.8, np.asarray([0.0]), np.log(1.0e-5)
    )
    exojax = LBLRTMSpectrum(nu_grid, np.asarray(model.transmission(parameters)))
    metrics = compare_transmission(
        degrade_to_resolving_power(lblrtm),
        degrade_to_resolving_power(exojax),
    )
    result = {
        "case": "CO-only, continuum disabled, AER 3.9 lines",
        "wavenumber_cm1": list(limits),
        "lblrtm": "12.17",
        "exojax": "2.5.0",
        "resolving_power": 45_000.0,
        "metrics": asdict(metrics),
        "thresholds": {
            "median_absolute_error": 1.0e-3,
            "percentile_99_absolute_error": 5.0e-3,
            "absolute_line_shift_resolution_elements": 0.1,
        },
    }
    output = root / "tests/data/aer_co_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
