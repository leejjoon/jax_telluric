#!/usr/bin/env python
"""Fit and correct many Arcturus atlas pages, resumably.

Differences from the single-page driver, all forced by running across the whole
atlas rather than one hand-picked window:

*Species are chosen per page, not fixed.* A list that suits 2 micron is wrong at
2.3 (CO) or 3.3 (CH4), and freeing a species with no absorption in the window
just sends it to a bound. Every species with lines is kept in the model so its
opacity is right; only those whose optical depth is actually measurable are
freed.

*The grid is sized per page.* Pages run from 7 to 57 cm-1 wide and the line
density varies by orders of magnitude, so the layer chunk is chosen from the
line count to stay inside device memory.

*Failures are recorded, not fatal.* One page that will not converge must not
stop the run, and a resumed run must not repeat completed work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

MOLECULE_IDS = {"H2O": 1, "CO2": 2, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
STAGES = ("continuum", "velocity", "columns", "stellar")


def page_windows(ils_report: Path, epoch: str) -> list[dict]:
    """Each page's own wavenumber range, trimmed off its neighbours' overlaps."""
    values = json.loads(ils_report.read_text())
    rows = [m for m in values["measurements"] if m.get("epoch") == epoch and "error" not in m]
    rows.sort(key=lambda m: m["wavenumber_min_cm1"])
    out = []
    for index, row in enumerate(rows):
        # A page's nominal range starts at its own name and ends where the next
        # page starts; that is what avoids the ~2 cm-1 overlap where adjacent
        # pages differ by a per-page scalar.
        lower = row["wavenumber_min_cm1"] + 2.5
        upper = row["wavenumber_max_cm1"] - 2.5
        if index + 1 < len(rows):
            upper = min(upper, rows[index + 1]["wavenumber_min_cm1"] + 2.5)
        if upper - lower < 4.0:
            continue
        out.append({"page": row["page"], "v1": float(lower), "v2": float(upper),
                    "mopd_cm": float(row["mopd_cm"]), "samples": int(row["samples"])})
    return out


def enable_compilation_cache(directory: Path) -> None:
    """Persist compiled executables so a repeated shape never recompiles.

    Shapes vary per page, so within one pass this mostly does not hit. It pays
    on a resumed run, on a re-run after a code change that leaves the graph
    alone, and across workers sharing one directory, where the same shape would
    otherwise be compiled once per worker.
    """

    import jax

    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def run_one(window, epoch, args, root):
    """Fit one page-epoch and export its corrected spectrum."""
    import jax.numpy as jnp
    from jax_telluric import (
        AERLineDatabase, ArrayOpacityBackend, BoxcarFTSInstrumentProfile,
        select_significant_lines, trim_wavenumber_grid,
        ExoJAXOpacityBackend, MTCKDWaterContinuum, StellarSpectrum, TelluricModel,
        OrderObjective, TelluricParameters, arcturus_spectral_order, epoch_velocity_kms,
        fit_order,
        igrins_wavenumber_grid, load_atmosphere_csv, prepare_stellar_source,
        read_arcturus_page,
    )

    v1, v2 = window["v1"], window["v2"]
    page = read_arcturus_page(args.atlas_root / window["page"], epoch).select(v1, v2)
    timing, _mark = {}, time.time()

    def mark(name):
        nonlocal _mark
        timing[name] = round(time.time() - _mark, 2)
        _mark = time.time()

    profile = load_atmosphere_csv(root / args.profile)
    # Two different margins. Lines are selected over the full reach of their
    # wings; the grid only has to cover the window plus what the LSF, the
    # Doppler shifts and the instrument profile reach back for. Sharing one
    # margin between them put 70% of the grid where there is no data.
    full_grid = igrins_wavenumber_grid(1.0e7 / v2, 1.0e7 / v1,
                                       resolving_power=args.resolving_power,
                                       samples_per_resolution=args.samples_per_resolution,
                                       margin_cm1=args.margin_cm1)
    grid = trim_wavenumber_grid(full_grid, v1, v2, args.grid_margin_cm1)
    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    databases, absent = {}, []
    for species, molecule_id in sorted(MOLECULE_IDS.items()):
        name = f"{molecule_id:02d}_{species}"
        try:
            databases[species] = AERLineDatabase(
                line_root / name / name, species, (v1, v2), margin_cm1=args.margin_cm1)
        except ValueError as exc:
            if not str(exc).startswith(f"no {species} lines found"):
                raise
            absent.append(species)
    mark("line_files")
    if not databases:
        raise RuntimeError("no molecular lines in this window")
    lines = {k: int(v.nu_lines.size) for k, v in databases.items()}
    if args.line_budget > 0.0:
        # Discard the weakest lines whose bounds sum to this much optical
        # depth. The bound is a worst case that assumes they all peak together;
        # measured, a budget of 1e-3 moves the transmission by 4e-6.
        databases = {
            species: select_significant_lines(
                database, profile, species, args.line_budget,
                maximum_column_scale=float(np.exp(2.0)),
            )
            for species, database in databases.items()
        }
        lines_kept = {k: int(np.asarray(v.nu_lines).size) for k, v in databases.items()}
    else:
        lines_kept = lines

    # No layer chunking. XLA fuses the wing sum into one reduction, so peak
    # memory is 1.9 GB on the atlas's widest page whether layers are split or
    # not, while splitting them multiplies compile time fivefold (40.8 s at
    # chunk 2 against 8.0 s unsplit). The option remains for a grid fine enough
    # to break that fusion.
    chunk = args.layer_chunk_size or None
    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=True, mixed_precision=True, pressure_shift=True,
        layer_chunk_size=chunk)
    mark("opacity_prepare")
    continuum = MTCKDWaterContinuum.from_netcdf(
        root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", grid)
    instrument = BoxcarFTSInstrumentProfile(
        mopd_cm=window["mopd_cm"], wavenumber_center_cm1=float(0.5 * (v1 + v2)),
        max_residual_sigma_kms=4.0)
    model = TelluricModel(profile, grid, opacity, continuum=continuum, accuracy_mode="mt_ckd",
                          max_lsf_sigma_kms=4.0, pixel_integration="point", instrument=instrument)

    mark("continuum")
    source = prepare_stellar_source(StellarSpectrum.from_npz(args.stellar), model,
                                    vsini_kms=2.0, macroturbulence_kms=2.15)
    mark("stellar_source")
    order = arcturus_spectral_order(page, source_flux_model_grid=source)

    # Evaluate the line-by-line kernel once and carry it as fixed arrays,
    # expanded to first order in the self-broadening pressure. That is the only
    # route a fitted parameter takes into the kernel, and a weak one, so the
    # column scales stay free while an optimizer iteration gets about thirty
    # times cheaper. The exact model is kept for the products written below.
    fit_model = (model.precompute_opacity(self_broadening=args.self_broadening)
                 if args.precompute_opacity else model)

    mark("precompute")
    # Decide which species can actually be measured here. Everything stays in
    # the model; only what the data constrains is freed.
    pressure = jnp.asarray(profile.pressure_layer_bar)
    partial = {s: pressure * jnp.asarray(profile.vmr[s]) for s in model.species}
    xs = fit_model.opacity.cross_sections(jnp.asarray(profile.temperature_k), pressure, partial)
    column = np.asarray(profile.air_column_cm2)
    tau = {s: float(np.max(np.sum(np.asarray(xs[s]) * (column * np.asarray(profile.vmr[s]))[:, None], axis=0)))
           for s in model.species}
    free_species = sorted(s for s, t in tau.items() if t >= args.min_optical_depth)
    # A page with no measurable absorption is a legitimate result, not an error:
    # the correction is a near-identity and the fit still yields the continuum,
    # the velocities and a corrected spectrum. Record it and carry on.
    negligible_telluric = not free_species

    degree = args.continuum_degree
    parameters = TelluricParameters(
        log_column_scales={s: 0.0 for s in model.species},
        velocity_kms=0.0, wavelength_stretch=0.0, lsf_sigma_kms=0.5,
        continuum_coeffs=np.concatenate([[float(np.log(np.median(np.asarray(order.flux)[order.mask])))],
                                         np.zeros(degree)]),
        log_jitter=float(np.log(np.median(order.uncertainty))),
        stellar_velocity_kms=epoch_velocity_kms(epoch))

    def bounds_for(stage):
        free = {"continuum": {"continuum", "log_jitter"},
                "velocity": {"continuum", "log_jitter", "velocity_kms", "lsf_sigma_kms"},
                "columns": {"continuum", "log_jitter", "velocity_kms", "lsf_sigma_kms", "species"},
                "stellar": {"continuum", "log_jitter", "velocity_kms", "lsf_sigma_kms", "species",
                            "stellar_velocity_kms"}}[stage]
        b = {}
        for s in model.species:
            b[s] = (-2.0, 2.0) if ("species" in free and s in free_species) else (0.0, 0.0)
        b["velocity_kms"] = (-5.0, 5.0) if "velocity_kms" in free else (float(parameters.velocity_kms),) * 2
        b["wavelength_stretch"] = (0.0, 0.0)
        b["lsf_sigma_kms"] = (0.05, 4.0) if "lsf_sigma_kms" in free else (float(parameters.lsf_sigma_kms),) * 2
        for i in range(degree + 1):
            v = float(np.asarray(parameters.continuum_coeffs)[i])
            b[f"continuum_{i}"] = ((-2.0, 2.0) if i == 0 else (-0.5, 0.5)) if "continuum" in free else (v, v)
        b["log_jitter"] = ((np.log(1e-4), np.log(0.1)) if "log_jitter" in free
                           else (float(parameters.log_jitter),) * 2)
        b["stellar_velocity_kms"] = ((-45.0, 45.0) if "stellar_velocity_kms" in free
                                     else (float(parameters.stellar_velocity_kms),) * 2)
        return b

    mark("species_scan")
    stages = []
    # One compilation for every stage. A fresh jax.jit closure per call is a
    # fresh cache entry, and compiling this graph costs seconds while running
    # it costs milliseconds.
    shared = OrderObjective(fit_model, order, degree + 1)
    # Force the compilation here so the stage timings measure fitting, not the
    # one-time XLA cost that would otherwise land entirely on the first stage.
    shared(shared.codec.pack(parameters))
    mark("compile_objective")
    for stage in STAGES:
        started = time.time()
        result = fit_order(fit_model, order, parameters, bounds_for(stage), objective=shared)
        parameters = result.parameters
        stages.append({"stage": stage, "success": bool(result.success),
                       "objective": result.objective, "iterations": result.iterations,
                       "seconds": round(time.time() - started, 1)})

    mark("stages")
    star_only = TelluricModel(
        profile, grid,
        ArrayOpacityBackend({s: np.zeros((len(profile.temperature_k), grid.size)) for s in model.species}),
        accuracy_mode="fast", max_lsf_sigma_kms=4.0,
        pixel_integration="point", instrument=instrument)
    stellar_only_pixels = np.asarray(star_only.predict(order, result.parameters))

    mark("star_only")
    order_idx = np.argsort(1.0e7 / np.asarray(order.wavelength_vacuum_nm))
    nu = (1.0e7 / np.asarray(order.wavelength_vacuum_nm))[order_idx]
    mask = np.asarray(order.mask)[order_idx]
    obs = np.asarray(order.flux)[order_idx]
    # Everything saved is re-evaluated without the precomputation's
    # approximation. Refreezing at the fitted parameters is exact there -- the
    # expansion's offset from its own reference is zero -- and reuses the
    # compilation, so this costs milliseconds rather than the half minute an
    # eager call to the uncompiled kernel would.
    exact_model = model.precompute_opacity(parameters)
    exact_flux = np.asarray(exact_model.predict(order, parameters))
    exact_transmission = np.asarray(exact_model.transmission(parameters, order.zenith_angle_deg))
    model_flux = exact_flux[order_idx]
    star = stellar_only_pixels[order_idx]
    transmission = np.interp(nu, np.asarray(model.wavenumber_cm1), exact_transmission)
    corrected = (obs / np.maximum(model_flux, 1e-6)) * star
    reliable = mask & (transmission >= args.min_transmission)
    residual = (np.asarray(order.flux) - exact_flux)[order_idx]
    sigma = float(order.uncertainty[0])

    mark("products")
    np.savez_compressed(
        args.output_dir / f"{window['page']}_{epoch}.npz",
        wavenumber_cm1=nu, observed=np.where(mask, obs, np.nan), model_flux=model_flux,
        transmission=transmission, corrected=corrected, stellar_only=star,
        residual=residual, mask=mask, reliable=reliable,
        atlas_telluric=page.telluric[::-1], atlas_ratioed=page.ratioed[::-1])

    return {
        "page": window["page"], "epoch": epoch, "wavenumber_cm1": [v1, v2],
        "pixels": int(mask.size), "reliable": int(reliable.sum()),
        "grid_points": int(grid.size), "layer_chunk": chunk, "timing": timing,
        "grid_margin_cm1": args.grid_margin_cm1, "line_margin_cm1": args.margin_cm1,
        "precomputed_opacity": (args.self_broadening if args.precompute_opacity else None),
        "mopd_cm": window["mopd_cm"], "lines": lines, "lines_kept": lines_kept,
        "line_budget": args.line_budget, "absent": absent,
        "max_optical_depth": {k: round(v, 4) for k, v in sorted(tau.items())},
        "free_species": free_species,
        "negligible_telluric": negligible_telluric,
        "median_transmission": float(np.median(transmission[mask])),
        "parameters": {**{s: float(parameters.log_column_scales[s]) for s in model.species},
                       "velocity_kms": float(parameters.velocity_kms),
                       "stellar_velocity_kms": float(parameters.stellar_velocity_kms),
                       "lsf_sigma_kms": float(parameters.lsf_sigma_kms),
                       "log_jitter": float(parameters.log_jitter)},
        "pixel_sigma": sigma,
        "residual_rms": float(np.sqrt(np.mean(residual[mask] ** 2))),
        "residual_rms_over_noise": float(np.sqrt(np.mean(residual[mask] ** 2)) / sigma),
        "reduced_chi2": float(np.mean((residual[mask] / sigma) ** 2)),
        "all_stages_converged": all(s["success"] for s in stages),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--atlas-root", type=Path,
                        default=Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/arcturus/ir"))
    parser.add_argument("--ils-report", type=Path, default=root / "docs/arcturus_ils.json")
    parser.add_argument("--epochs", default="summer,winter")
    parser.add_argument("--pages", default=None, help="comma-separated page names; default every page")
    parser.add_argument("--stride", type=int, default=1, help="take every Nth page (for a subset)")
    parser.add_argument("--profile", type=Path, default=Path("data/profiles/kitt_peak_1994.csv"))
    parser.add_argument("--stellar", type=Path, default=root / "data/stellar/arcturus_payne_zero_full.npz")
    parser.add_argument("--resolving-power", type=float, default=100_000.0)
    parser.add_argument("--samples-per-resolution", type=float, default=4.0)
    parser.add_argument("--margin-cm1", type=float, default=25.0,
                        help="how far outside the window a line may still contribute")
    parser.add_argument("--grid-margin-cm1", type=float, default=5.0,
                        help="how far outside the window the model grid extends")
    parser.add_argument("--continuum-degree", type=int, default=3)
    parser.add_argument("--min-optical-depth", type=float, default=0.02,
                        help="free a species only if its peak vertical optical depth reaches this")
    parser.add_argument("--min-transmission", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=root / "data/corrected/atlas")
    parser.add_argument("--summary", type=Path, default=root / "docs/arcturus_atlas_summary.json")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precompute-opacity", action=argparse.BooleanOptionalAction, default=True,
                        help="evaluate the line-by-line kernel once per page instead of per iteration")
    parser.add_argument("--self-broadening", choices=("linear", "frozen"), default="linear",
                        help="how a precomputed kernel carries its self-pressure dependence")
    parser.add_argument("--layer-chunk-size", type=int, default=0,
                        help="split the layer vmap into chunks; 0 means do not split")
    parser.add_argument("--line-budget", type=float, default=0.0,
                        help="optical depth of the weakest lines to discard; 0 keeps every line")
    # A string, not a Path: Path("") is Path("."), which is truthy and would
    # scatter cache entries through the repository root.
    parser.add_argument("--compilation-cache", default=str(root / ".jax-cache"),
                        help="directory for persisted XLA executables; empty string disables it")
    args = parser.parse_args()
    if args.compilation_cache.strip():
        enable_compilation_cache(Path(args.compilation_cache))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    done = {}
    if args.resume and args.summary.exists():
        for row in json.loads(args.summary.read_text())["results"]:
            done[(row["page"], row["epoch"])] = row

    jobs = []
    for epoch in args.epochs.split(","):
        windows = page_windows(args.ils_report, epoch.strip())
        if args.pages:
            wanted = {p.strip() for p in args.pages.split(",")}
            windows = [w for w in windows if w["page"] in wanted]
        else:
            windows = windows[:: args.stride]
        jobs.extend((w, epoch.strip()) for w in windows)

    results = list(done.values())
    started = time.time()
    for index, (window, epoch) in enumerate(jobs, 1):
        key = (window["page"], epoch)
        if key in done:
            continue
        label = f"{window['page']} {epoch}"
        try:
            row = run_one(window, epoch, args, root)
            status = (f"rms/noise {row['residual_rms_over_noise']:5.2f}  "
                      f"free {'+'.join(row['free_species'])}  "
                      f"{row['grid_points']} pts")
        except Exception as exc:  # a bad page must not stop the run
            row = {"page": window["page"], "epoch": epoch, "error": str(exc),
                   "traceback": traceback.format_exc()[-1500:]}
            status = f"FAILED: {exc}"
        results.append(row)
        done[key] = row
        elapsed = time.time() - started
        print(f"[{index}/{len(jobs)}] {label:<20} {status}   ({elapsed/60:.1f} min elapsed)", flush=True)
        args.summary.write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "atlas_root": str(args.atlas_root), "stellar": str(args.stellar),
             "profile": str(args.profile),
             "settings": {"resolving_power": args.resolving_power,
                          "samples_per_resolution": args.samples_per_resolution,
                          "margin_cm1": args.margin_cm1,
                          "min_optical_depth": args.min_optical_depth,
                          "min_transmission": args.min_transmission},
             "results": sorted(results, key=lambda r: (r["page"], r["epoch"]))},
            indent=2) + "\n", encoding="utf-8")

    ok = [r for r in results if "error" not in r]
    print(f"\n{len(ok)} of {len(results)} succeeded; wrote {args.summary}")
    if ok:
        rms = np.array([r["residual_rms_over_noise"] for r in ok])
        print(f"residual rms / noise: median {np.median(rms):.2f}, "
              f"worst {rms.max():.2f}, best {rms.min():.2f}")


if __name__ == "__main__":
    main()
