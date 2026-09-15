"""Extended real-line mixed-precision checks; run in a dedicated process.

Example: CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda .venv/bin/python
scripts/validate_mixed_precision.py
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_telluric import (
    AERLineDatabase, AtmosphereProfile, ExoJAXOpacityBackend, SpectralOrder,
    TelluricModel, TelluricParameters, fit_order, igrins_wavenumber_grid,
    load_atmosphere_csv,
)
from exojax.opacity import OpaDirect
from jax_telluric.direct import SparseCoreDirect

ROOT = Path(__file__).resolve().parents[1]
LINES = ROOT / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
OUTPUT = ROOT / "docs/mixed_precision_validation.json"
IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}


def database(species, limits):
    name = f"{IDS[species]:02d}_{species}"
    return AERLineDatabase(LINES / name / name, species, limits)


def metrics(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise AssertionError("nonfinite validation output")
    delta = candidate - reference
    return {"max_absolute": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-300))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="reuse completed checks in the report")
    args = parser.parse_args()
    started = time.time()
    report = {"device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind,
              "jax": jax.__version__, "x64": jax.config.x64_enabled,
              "thresholds": {"opacity_relative_l2": 1e-6, "derivative_relative_l2": 1e-4,
                             "flux_absolute": 1e-6, "fit_shift_sigma": 0.01},
              "opacity_cases": [], "spectral_cases": [], "fits": []}
    if args.resume and OUTPUT.exists():
        report = json.loads(OUTPUT.read_text())
        assert report["device_kind"] == jax.devices()[0].device_kind
        assert report["jax"] == jax.__version__
    report["passed"] = False
    report["line_source"] = "AER 3.9"
    report["profile_sha256"] = hashlib.sha256(
        (ROOT / "data/profiles/example_midlatitude.csv").read_bytes()).hexdigest()

    def save():
        report["elapsed_s"] = time.time() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    # Deterministic Cartesian sweep covers low pressure, humid mixtures,
    # the core-list temperature boundary, and the float64 fallback.
    states = [(t, p, p * f) for t in (180., 240., 300., 399.9, 400., 400.1, 450.)
              for p in (1e-5, 0.01, 0.1, 1.1) for f in (1e-5, 0.001, 0.03)]
    report["thermodynamic_states"] = states
    for species, center in (("H2O", 6250.), ("H2O", 5000.), ("CO2", 6350.),
                            ("CH4", 6000.), ("CO", 4300.), ("N2O", 4500.), ("O2", 6300.)):
        limits = (center, center + 10.)
        if any(case["species"] == species and case["interval_cm1"] == list(limits)
               for case in report["opacity_cases"]):
            continue
        grid = igrins_wavenumber_grid(1e7 / limits[1], 1e7 / limits[0])
        db = database(species, limits)
        original = OpaDirect(db, grid)
        mixed = SparseCoreDirect(db, grid, mixed_precision=True)
        funcs = [jax.jit(lambda p, op=op: (op.xsvector(*p),
                 jax.jacfwd(lambda q: op.xsvector(*q))(p))) for op in (original, mixed)]
        case = {"species": species, "interval_cm1": limits, "lines": len(db.nu_lines),
                "states": len(states), "worst_opacity_relative_l2": 0.,
                "worst_derivative_relative_l2": [0., 0., 0.]}
        for state in states:
            old, new = [func(jnp.array(state)) for func in funcs]
            value_error = metrics(old[0], new[0])["relative_l2"]
            derivative_errors = [metrics(old[1][:, k], new[1][:, k])["relative_l2"] for k in range(3)]
            case["worst_opacity_relative_l2"] = max(case["worst_opacity_relative_l2"], value_error)
            case["worst_derivative_relative_l2"] = np.maximum(
                case["worst_derivative_relative_l2"], derivative_errors).tolist()
            assert value_error < 1e-6, (species, state, value_error)
            assert max(derivative_errors) < 1e-4, (species, state, derivative_errors)
        report["opacity_cases"].append(case)
        print(json.dumps(case), flush=True)
        save()
        del funcs, original, mixed, db
        jax.clear_caches()

    profile = load_atmosphere_csv(ROOT / "data/profiles/example_midlatitude.csv")
    for center in (6250., 5000., 4300.):
        limits = (center, center + 20.)
        if sum(case["interval_cm1"] == list(limits) for case in report["spectral_cases"]) == 15:
            continue
        grid = igrins_wavenumber_grid(1e7 / limits[1], 1e7 / limits[0])
        dbs = {}
        absent = []
        for species in IDS:
            try:
                dbs[species] = database(species, limits)
            except ValueError as exc:
                if str(exc).startswith(f"no {species} lines found"):
                    absent.append(species)
                else:
                    raise
        models = [TelluricModel(profile, grid, ExoJAXOpacityBackend.prepare(
            dbs, grid, methods="direct_sparse", vectorize_layers=True, mixed_precision=mixed))
            for mixed in (False, True)]
        wavelength = np.linspace(1e7 / limits[1], 1e7 / limits[0], 512)
        species = tuple(IDS)

        def unpack(p):
            return TelluricParameters(dict(zip(species, p[:6])), p[6], p[7] * 1e-5,
                                     p[8], p[9:11], np.log(1e-5))

        for airmass in (1., 1.5, 2.5):
            order = SpectralOrder(wavelength, np.ones(512), np.full(512, 0.001),
                                  zenith_angle_deg=np.degrees(np.arccos(1 / airmass)))
            funcs = [jax.jit(lambda p, model=model: (model.predict(order, unpack(p)),
                     jax.jacfwd(lambda q: model.predict(order, unpack(q)))(p))) for model in models]
            for water_scale in (0.05, 0.5, 1., 2., 5.):
                if any(case["interval_cm1"] == list(limits) and case["airmass"] == airmass
                       and case["H2O_scale"] == water_scale for case in report["spectral_cases"]):
                    continue
                p = jnp.array([np.log(water_scale), 0.1, -0.1, 0.2, -0.2, 0.,
                               0.3, 0.2, 2.8, 0.01, -0.005])
                old, new = [func(p) for func in funcs]
                err = metrics(old[0], new[0])
                deriv = [metrics(old[1][:, k], new[1][:, k])["relative_l2"] for k in range(11)]
                case = {"interval_cm1": limits, "airmass": airmass, "H2O_scale": water_scale,
                        "included_species": list(dbs), "species_without_lines_in_window": absent,
                        "flux": err, "jacobian_relative_l2_by_parameter": deriv}
                assert err["max_absolute"] < 1e-6, case
                assert max(deriv) < 1e-4, case
                report["spectral_cases"].append(case)
            print(f"spectra {limits} airmass={airmass} passed", flush=True)
            save()
        del funcs, models, dbs
        jax.clear_caches()

    # Paired recovery through the actual public fitter. Fix stretch, slope,
    # and jitter to isolate identifiable H2O, velocity, width, and throughput.
    limits = (5000., 5020.)
    grid = igrins_wavenumber_grid(1e7 / limits[1], 1e7 / limits[0])
    dbs = {"H2O": database("H2O", limits)}
    models = [TelluricModel(profile, grid, ExoJAXOpacityBackend.prepare(
        dbs, grid, methods="direct_sparse", vectorize_layers=True, mixed_precision=mixed))
        for mixed in (False, True)]
    wavelength = np.linspace(1e7 / limits[1], 1e7 / limits[0], 512)
    blank = SpectralOrder(wavelength, np.ones(512), np.full(512, .001), zenith_angle_deg=35.)
    truth = TelluricParameters({"H2O": 0.2}, 0.4, 0., 3.0, np.array([.01]), np.log(1e-5))
    initial = TelluricParameters({"H2O": 0.}, 0., 0., 2.8, np.array([0.]), np.log(1e-5))
    noiseless = np.asarray(jax.jit(lambda p: models[0].predict(blank, p))(truth))
    bounds = {"H2O": (-2., 2.), "velocity_kms": (-3., 3.), "wavelength_stretch": (0., 0.),
              "lsf_sigma_kms": (1., 5.), "continuum_0": (-.1, .1),
              "log_jitter": (float(truth.log_jitter), float(truth.log_jitter))}

    def compact(p):
        return np.array([p.log_column_scales["H2O"], p.velocity_kms, p.lsf_sigma_kms, p.continuum_coeffs[0]])

    def expanded(p):
        return TelluricParameters({"H2O": p[0]}, p[1], 0., p[2], p[3:4], truth.log_jitter)

    jacobian = np.asarray(jax.jacfwd(lambda p: models[0].predict(blank, expanded(p)))(jnp.array(compact(truth))))
    sigma = np.sqrt(np.diag(np.linalg.inv(jacobian.T @ jacobian / .001**2)))
    # Fits are cheap relative to the opacity sweeps and exercise optimizer
    # behavior, so always rerun them when resuming a validation.
    report["fits"] = []
    for seed in (17, 42, 123):
        flux = noiseless + np.random.default_rng(seed).normal(0., .001, 512)
        order = SpectralOrder(wavelength, flux, np.full(512, .001), zenith_angle_deg=35.)
        fits = [fit_order(model, order, initial, bounds) for model in models]
        values = [compact(f.parameters) for f in fits]
        shift_sigma = abs(values[1] - values[0]) / sigma
        case = {"seed": seed, "parameter_names": ["log_H2O", "velocity_kms", "lsf_sigma_kms", "log_continuum"],
                "truth": compact(truth).tolist(), "float64": values[0].tolist(), "mixed": values[1].tolist(),
                "local_fisher_sigma": sigma.tolist(), "precision_shift_sigma": shift_sigma.tolist(),
                "recovery_sigma": ((values[1] - compact(truth)) / sigma).tolist(),
                "success": [f.success for f in fits], "messages": [f.message for f in fits]}
        report["fits"].append(case)
        save()
        print(json.dumps(case), flush=True)
    report["fit_failures"] = [case["seed"] for case in report["fits"]
                              if not all(case["success"])
                              or max(case["precision_shift_sigma"]) >= .01
                              or max(abs(x) for x in case["recovery_sigma"]) >= 4.]
    report["passed"] = not report["fit_failures"]
    save()
    assert report["passed"], f"fit validation failed for seeds {report['fit_failures']}"
    print(f"PASS: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
