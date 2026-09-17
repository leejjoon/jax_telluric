#!/usr/bin/env python
"""Fit telluric absorption and a continuum to one Arcturus atlas page.

Uses ``accuracy_mode="mt_ckd"`` with the native differentiable continuum, so
LBLRTM is not involved at fit time. The cost is a line-physics bias against
LBLRTM of median 0.0039 / p99 0.058 in this window, against a per-pixel noise
near 0.005; it is reported alongside every fitted column scale rather than
being treated as random error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np

from jax_telluric import (
    AERLineDatabase,
    ArrayOpacityBackend,
    BoxcarFTSInstrumentProfile,
    ExoJAXOpacityBackend,
    MTCKDWaterContinuum,
    StellarSpectrum,
    TelluricModel,
    TelluricParameters,
    arcturus_spectral_order,
    epoch_velocity_kms,
    OrderObjective,
    trim_wavenumber_grid,
    fit_order,
    igrins_wavenumber_grid,
    load_atmosphere_csv,
    prepare_stellar_source,
    read_arcturus_page,
)

MOLECULE_IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
DEFAULT_PAGE = (
    "/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/arcturus/ir/ab5000_"
)
# Stages are cumulative: each frees more parameters starting from the last fit.
STAGES = {
    "continuum": ("continuum_*", "log_jitter"),
    "velocity": ("continuum_*", "log_jitter", "velocity_kms", "lsf_sigma_kms"),
    "columns": ("continuum_*", "log_jitter", "velocity_kms", "lsf_sigma_kms", "species_*"),
    "stellar": (
        "continuum_*", "log_jitter", "velocity_kms", "lsf_sigma_kms", "species_*",
        "stellar_velocity_kms",
    ),
}


def measured_mopd_cm(page_name: str, epoch: str, report: Path) -> float:
    """Look up the interferogram truncation measured by measure_atlas_ils.py."""

    values = json.loads(report.read_text())
    for entry in values["measurements"]:
        if entry.get("page") == page_name and entry.get("epoch") == epoch:
            if not entry.get("mopd_measurable"):
                raise ValueError(f"{page_name} {epoch} has no usable MOPD measurement")
            return float(entry["mopd_cm"])
    raise ValueError(f"no MOPD measurement for {page_name} {epoch} in {report}")


def build_bounds(species, continuum_degree, free, initial, include_stellar, pinned=()):
    """Bounds that pin everything except the named free parameters."""

    def window(name, lower, upper, pinned_at):
        if name in pinned:
            return (pinned_at, pinned_at)
        if name in free or (name.startswith("continuum_") and "continuum_*" in free) or (
            name in species and "species_*" in free
        ):
            return (lower, upper)
        return (pinned_at, pinned_at)

    bounds = {}
    for name in species:
        bounds[name] = window(name, -2.0, 2.0, float(initial.log_column_scales[name]))
    bounds["velocity_kms"] = window("velocity_kms", -5.0, 5.0, float(initial.velocity_kms))
    # An FTS wavenumber scale is exactly linear, so the only physical freedom is
    # a multiplicative factor, which is already the velocity. A free stretch
    # would only absorb model error.
    bounds["wavelength_stretch"] = (0.0, 0.0)
    bounds["lsf_sigma_kms"] = window("lsf_sigma_kms", 0.05, 4.0, float(initial.lsf_sigma_kms))
    for index in range(continuum_degree + 1):
        current = float(np.asarray(initial.continuum_coeffs)[index])
        limit = 2.0 if index == 0 else 0.5
        bounds[f"continuum_{index}"] = window(f"continuum_{index}", -limit, limit, current)
    bounds["log_jitter"] = window("log_jitter", np.log(1e-4), np.log(0.1), float(initial.log_jitter))
    if include_stellar:
        bounds["stellar_velocity_kms"] = window(
            "stellar_velocity_kms", -45.0, 45.0, float(initial.stellar_velocity_kms)
        )
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, default=Path(DEFAULT_PAGE))
    parser.add_argument("--epoch", choices=("summer", "winter"), default="summer")
    parser.add_argument("--column", choices=("observed", "telluric"), default="observed",
                        help="'telluric' fits the atlas's own transmission column (L3)")
    parser.add_argument("--v1", type=float, default=5005.0)
    parser.add_argument("--v2", type=float, default=5025.0)
    parser.add_argument("--margin-cm1", type=float, default=25.0,
                        help="how far outside the window a line may still contribute")
    parser.add_argument("--grid-margin-cm1", type=float, default=5.0,
                        help="how far outside the window the model grid extends")
    parser.add_argument("--resolving-power", type=float, default=100_000.0)
    parser.add_argument("--samples-per-resolution", type=float, default=4.0)
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/kitt_peak_1994.csv"))
    # N2O and CH4 have mean optical depth below 2e-4 across this window, so
    # freeing them just lets the optimizer run them to a bound.
    parser.add_argument("--species", default="H2O,CO2",
                        help="comma-separated molecules to give opacity, or 'all'")
    parser.add_argument("--stellar", default="flat", help="'flat' or a path to an npz")
    parser.add_argument("--vsini-kms", type=float, default=2.0)
    parser.add_argument("--macroturbulence-kms", type=float, default=2.15)
    parser.add_argument("--continuum-degree", type=int, default=3)
    parser.add_argument("--stages", default="continuum,velocity,columns")
    parser.add_argument("--pin", action="append", default=[], metavar="NAME=VALUE",
                        help="hold a parameter at a value, e.g. --pin CO2=0.2249 (log scale)")
    parser.add_argument("--ils-report", type=Path, default=Path("docs/arcturus_ils.json"))
    parser.add_argument("--gaussian-ils", action="store_true",
                        help="use the Gaussian line spread function instead of the measured sinc")
    parser.add_argument("--simpson", action="store_true",
                        help="integrate over pixels instead of point sampling")
    # vmap over layers costs memory (the wing matrix is dense in lines x grid)
    # but a Python layer loop unrolls into one copy of that per layer, and
    # reverse mode doubles it: measured compile 24 s vectorized against ~400 s.
    parser.add_argument("--vectorize-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precompute-opacity", action=argparse.BooleanOptionalAction, default=True,
                        help="evaluate the line-by-line kernel once instead of per iteration")
    parser.add_argument("--self-broadening", choices=("linear", "frozen"), default="linear",
                        help="how a precomputed kernel carries its self-pressure dependence")
    parser.add_argument("--layer-chunk-size", type=int, default=0,
                        help="layers vectorized at once; lower it if the device runs out of memory")
    parser.add_argument("--report", type=Path, default=Path("docs/arcturus_ab5000_fit.json"))
    parser.add_argument("--diagnostic-npz", type=Path,
                        default=Path("benchmarks/results/arcturus_ab5000.npz"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    started = time.time()

    page = read_arcturus_page(args.page, args.epoch).select(args.v1, args.v2)
    profile = load_atmosphere_csv(root / args.profile)
    grid = igrins_wavenumber_grid(
        1.0e7 / args.v2, 1.0e7 / args.v1,
        resolving_power=args.resolving_power,
        samples_per_resolution=args.samples_per_resolution,
        margin_cm1=args.margin_cm1,
    )
    # Lines are selected over the full reach of their wings; the grid only has
    # to cover the window plus what the LSF, the Doppler shifts and the
    # instrument profile reach back for. Sharing one margin put most of the
    # grid where there is no data.
    grid = trim_wavenumber_grid(grid, args.v1, args.v2, args.grid_margin_cm1)

    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    wanted = (
        set(MOLECULE_IDS)
        if args.species == "all"
        else {name.strip().upper() for name in args.species.split(",")}
    )
    unknown = wanted - set(MOLECULE_IDS)
    if unknown:
        raise SystemExit(f"unknown species: {', '.join(sorted(unknown))}")
    databases, skipped = {}, []
    for species, molecule_id in sorted(MOLECULE_IDS.items()):
        if species not in wanted:
            continue
        name = f"{molecule_id:02d}_{species}"
        try:
            databases[species] = AERLineDatabase(
                line_root / name / name, species, (args.v1, args.v2), margin_cm1=args.margin_cm1
            )
        except ValueError as exc:
            if not str(exc).startswith(f"no {species} lines found"):
                raise
            skipped.append(species)
    lines_per_species = {name: int(db.nu_lines.size) for name, db in databases.items()}

    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=args.vectorize_layers,
        mixed_precision=args.mixed_precision,
        pressure_shift=True,
        layer_chunk_size=args.layer_chunk_size or None,
    )
    continuum = MTCKDWaterContinuum.from_netcdf(
        root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", grid
    )
    mopd_cm = None if args.gaussian_ils else measured_mopd_cm(
        args.page.name, args.epoch, root / args.ils_report
    )
    instrument = None if args.gaussian_ils else BoxcarFTSInstrumentProfile(
        mopd_cm=mopd_cm,
        wavenumber_center_cm1=float(0.5 * (args.v1 + args.v2)),
        max_residual_sigma_kms=4.0,
    )
    model = TelluricModel(
        profile, grid, opacity,
        continuum=continuum,
        accuracy_mode="mt_ckd",
        max_lsf_sigma_kms=4.0,
        pixel_integration="simpson" if args.simpson else "point",
        instrument=instrument,
    )

    if args.stellar == "flat":
        stellar = StellarSpectrum.flat(grid)
        source = prepare_stellar_source(stellar, model)
    else:
        stellar = StellarSpectrum.from_npz(args.stellar)
        source = prepare_stellar_source(
            stellar, model,
            vsini_kms=args.vsini_kms, macroturbulence_kms=args.macroturbulence_kms,
        )
    order = arcturus_spectral_order(page, column=args.column, source_flux_model_grid=source)

    # Evaluate the line-by-line kernel once and carry it as fixed arrays,
    # expanded to first order in the self-broadening pressure -- the only route
    # a fitted parameter takes into it, and a weak one. The exact model stays
    # available and every product below is re-evaluated through it.
    fit_model = (model.precompute_opacity(self_broadening=args.self_broadening)
                 if args.precompute_opacity else model)

    parameters = TelluricParameters(
        log_column_scales={name: 0.0 for name in model.species},
        velocity_kms=0.0,
        wavelength_stretch=0.0,
        lsf_sigma_kms=0.5,
        continuum_coeffs=np.zeros(args.continuum_degree + 1),
        log_jitter=float(np.log(np.median(order.uncertainty))),
        # The atlas readme's Doppler factor is the starting point; the fit is
        # what decides, and the sign convention is checked by L0.
        stellar_velocity_kms=epoch_velocity_kms(args.epoch) if args.stellar != "flat" else 0.0,
    )
    parameters = parameters._replace(
        continuum_coeffs=np.concatenate(
            [[float(np.log(np.median(order.flux[order.mask])))], np.zeros(args.continuum_degree)]
        )
    )

    pinned = {}
    for entry in args.pin:
        name, _, value = entry.partition("=")
        name = name.strip()
        try:
            pinned[name] = float(value)
        except ValueError:
            raise SystemExit(f"--pin expects NAME=VALUE, got {entry!r}")
    unknown = set(pinned) - set(model.species) - {
        "velocity_kms", "stellar_velocity_kms", "lsf_sigma_kms", "log_jitter",
        *(f"continuum_{i}" for i in range(args.continuum_degree + 1)),
    }
    if unknown:
        raise SystemExit(f"cannot pin unknown parameter(s): {', '.join(sorted(unknown))}")
    if pinned:
        scales = dict(parameters.log_column_scales)
        scales.update({k: v for k, v in pinned.items() if k in model.species})
        replacements = {
            k: v for k, v in pinned.items()
            if k in ("velocity_kms", "stellar_velocity_kms", "lsf_sigma_kms", "log_jitter")
        }
        parameters = parameters._replace(log_column_scales=scales, **replacements)
        print(f"pinned: {', '.join(f'{k}={v:g}' for k, v in sorted(pinned.items()))}")

    stage_reports = []
    # One compilation for every stage: a fresh jax.jit closure per call is a
    # fresh cache entry, and this graph costs seconds to compile against
    # milliseconds to run.
    shared = OrderObjective(fit_model, order, args.continuum_degree + 1)
    for stage in args.stages.split(","):
        stage = stage.strip()
        if stage not in STAGES:
            raise SystemExit(f"unknown stage {stage}; choose from {', '.join(STAGES)}")
        bounds = build_bounds(
            model.species, args.continuum_degree, STAGES[stage], parameters,
            include_stellar=order.source_flux_model_grid is not None, pinned=pinned,
        )
        stage_started = time.time()
        result = fit_order(fit_model, order, parameters, bounds, objective=shared)
        parameters = result.parameters
        stage_reports.append({
            "stage": stage,
            "free": [name for name, (lo, hi) in bounds.items() if lo < hi],
            "objective": result.objective,
            "success": bool(result.success),
            "message": result.message,
            "iterations": result.iterations,
            "seconds": round(time.time() - stage_started, 2),
        })
        print(f"[{stage}] objective={result.objective:.6g} success={result.success} "
              f"iterations={result.iterations} ({stage_reports[-1]['seconds']} s)")

    # A parameter resting on a bound is not a measurement; say so explicitly.
    final_bounds = build_bounds(
        model.species, args.continuum_degree, STAGES[args.stages.split(",")[-1].strip()],
        parameters, include_stellar=order.source_flux_model_grid is not None, pinned=pinned,
    )
    packed = {
        **{name: float(v) for name, v in result.parameters.log_column_scales.items()},
        "velocity_kms": float(result.parameters.velocity_kms),
        "stellar_velocity_kms": float(result.parameters.stellar_velocity_kms),
        "wavelength_stretch": float(result.parameters.wavelength_stretch),
        "lsf_sigma_kms": float(result.parameters.lsf_sigma_kms),
        "log_jitter": float(result.parameters.log_jitter),
        **{f"continuum_{i}": float(v) for i, v in enumerate(np.asarray(result.parameters.continuum_coeffs))},
    }
    at_bound = sorted(
        name for name, value in packed.items()
        if name in final_bounds and final_bounds[name][0] < final_bounds[name][1]
        and min(abs(value - final_bounds[name][0]), abs(value - final_bounds[name][1])) < 1e-6
    )
    if at_bound:
        print(f"WARNING: parameters resting on a bound: {', '.join(at_bound)}")

    # The same forward model with the atmosphere removed: the star through the
    # same instrument, sampled the same way. Correcting by the model ratio needs
    # it, and that is the only way to avoid dividing by a convolved
    # transmission, which does not recover the star (see docs/arcturus_fit.md).
    star_only_model = TelluricModel(
        profile, grid,
        ArrayOpacityBackend({name: np.zeros((len(profile.temperature_k), grid.size))
                             for name in model.species}),
        accuracy_mode="fast", max_lsf_sigma_kms=4.0,
        pixel_integration=model.pixel_integration, instrument=instrument,
    )
    stellar_only_pixels = np.asarray(star_only_model.predict(order, result.parameters))

    # Re-evaluate without the precomputation's approximation. Refreezing at the
    # fitted parameters is exact there -- the expansion's offset from its own
    # reference is zero -- and reuses the compilation, so this costs
    # milliseconds rather than the half minute an eager call would.
    exact_model = model.precompute_opacity(result.parameters)
    exact_flux = np.asarray(exact_model.predict(order, result.parameters))
    exact_transmission = np.asarray(
        exact_model.transmission(result.parameters, order.zenith_angle_deg))
    residual = np.asarray(order.flux) - exact_flux
    mask = np.asarray(order.mask)
    scaled = residual[mask] / np.asarray(order.uncertainty)[mask]
    transmission_pixels = np.interp(
        np.asarray(order.wavelength_vacuum_nm),
        (1.0e7 / np.asarray(model.wavenumber_cm1))[::-1],
        exact_transmission[::-1],
    )
    atlas_telluric = page.telluric[::-1]
    comparable = mask & (atlas_telluric > 0.2) & (atlas_telluric <= 1.0)

    report = {
        "page": {
            "path": str(args.page), "name": args.page.name, "epoch": args.epoch,
            "sha256": page.sha256, "column": args.column,
            "wavenumber_cm1": [float(page.wavenumber_vacuum_cm1[0]), float(page.wavenumber_vacuum_cm1[-1])],
            "pixels": int(mask.size), "masked": int((~mask).sum()),
            "spacing_cm1": page.spacing_cm1,
            "uncertainty": float(order.uncertainty[0]),
        },
        "grid": {
            "resolving_power": args.resolving_power,
            "samples_per_resolution": args.samples_per_resolution,
            "margin_cm1": args.margin_cm1,
            "grid_margin_cm1": args.grid_margin_cm1,
            "points": int(grid.size),
            "velocity_step_kms": model.velocity_step_kms,
        },
        "physics": {
            "accuracy_mode": model.accuracy_mode,
            "continuum": "native MT_CKD 4.3",
            "pressure_shift": True,
            "mixed_precision": args.mixed_precision,
            "precomputed_opacity": (args.self_broadening if args.precompute_opacity else None),
            "pixel_integration": model.pixel_integration,
            "instrument": "gaussian" if instrument is None else "boxcar FTS sinc",
            "mopd_cm": mopd_cm,
            "instrument_resolving_power": None if instrument is None else instrument.resolving_power,
            "profile": str(args.profile),
            "species": list(model.species),
            "lines_per_species": lines_per_species,
            "species_without_lines": skipped,
            "zenith_angle_deg": order.zenith_angle_deg,
            "lblrtm_bias_note": (
                "mt_ckd mode disagrees with LBLRTM by median 0.0039 / p99 0.058 in this window "
                "(data/corrections/lblrtm_5000_5020.json); the fitted column scales absorb part "
                "of that and are biased accordingly."
            ),
        },
        "stellar": {
            "source": args.stellar,
            "vsini_kms": args.vsini_kms if args.stellar != "flat" else 0.0,
            "macroturbulence_kms": args.macroturbulence_kms if args.stellar != "flat" else 0.0,
            "readme_velocity_kms": epoch_velocity_kms(args.epoch),
        },
        "pinned": pinned,
        "stages": stage_reports,
        "at_bound": at_bound,
        "parameters": {
            **{name: float(value) for name, value in result.parameters.log_column_scales.items()},
            "velocity_kms": float(result.parameters.velocity_kms),
            "stellar_velocity_kms": float(result.parameters.stellar_velocity_kms),
            "lsf_sigma_kms": float(result.parameters.lsf_sigma_kms),
            "continuum_coeffs": [float(v) for v in np.asarray(result.parameters.continuum_coeffs)],
            "log_jitter": float(result.parameters.log_jitter),
        },
        "residuals": {
            "reduced_chi2": float(np.sum(scaled**2) / scaled.size),
            "rms": float(np.sqrt(np.mean(residual[mask] ** 2))),
            "percentile_99_absolute": float(np.percentile(np.abs(residual[mask]), 99.0)),
            "jitter_over_uncertainty": float(
                np.exp(result.parameters.log_jitter) / np.median(order.uncertainty)
            ),
        },
        "atlas_telluric_comparison": {
            "restricted_to": "atlas telluric in (0.2, 1.0]",
            "pixels": int(comparable.sum()),
            "median_absolute": float(np.median(np.abs(transmission_pixels[comparable] - atlas_telluric[comparable]))),
            "percentile_99_absolute": float(np.percentile(np.abs(transmission_pixels[comparable] - atlas_telluric[comparable]), 99.0)),
            "note": "Hinkle's telluric column is a scaled transmission from a different observation; a consistency floor, not truth.",
        },
        "runtime_seconds": round(time.time() - started, 1),
        "platform": str(jax.devices()[0]),
        "jax": jax.__version__,
    }

    (root / args.report).parent.mkdir(parents=True, exist_ok=True)
    (root / args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / args.diagnostic_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / args.diagnostic_npz,
        wavelength_vacuum_nm=np.asarray(order.wavelength_vacuum_nm),
        wavenumber_cm1=page.wavenumber_vacuum_cm1[::-1],
        flux=np.asarray(order.flux), uncertainty=np.asarray(order.uncertainty), mask=mask,
        # SpectralOrder.flux carries a placeholder of 1.0 at masked pixels,
        # because it must be finite and positive everywhere. That placeholder
        # sits in the core of every saturated line and will look like a feature
        # if it is plotted, so keep the untouched column alongside it.
        observed_raw=getattr(page, args.column)[::-1],
        model_flux=exact_flux, residual=residual,
        transmission_pixels=transmission_pixels,
        atlas_telluric=atlas_telluric, atlas_ratioed=page.ratioed[::-1],
        grid_wavenumber_cm1=np.asarray(model.wavenumber_cm1),
        grid_transmission=exact_transmission,
        stellar_source=np.asarray(source),
        stellar_only_pixels=stellar_only_pixels,
    )
    print(json.dumps(report["residuals"], indent=2))
    print(f"wrote {args.report} and {args.diagnostic_npz}")


if __name__ == "__main__":
    main()
