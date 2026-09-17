#!/usr/bin/env python
"""Retrieve the H2O column in sub-windows of a page, to localise a disagreement.

L7 compares the water column from two adjacent pages and finds them 5.8% apart.
That can mean two different things, and only a finer scan separates them:

  * a page-level offset -- something about the page as a whole, such as its
    multiplicative scalar or its continuum, biasing the column; or
  * line-to-line scatter -- different water lines simply giving different
    answers, in which case the page-to-page number is not special and the real
    quantity is the spread.

Each sub-window is fitted with everything except the water column and the
continuum held at the page's own full-window solution, so the only thing that
moves is the water.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    arcturus_spectral_order,
    fit_order,
    igrins_wavenumber_grid,
    load_atmosphere_csv,
    prepare_stellar_source,
    read_arcturus_page,
)

MOLECULE_IDS = {"H2O": 1, "CO2": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    atlas = Path("/home/jjlee/work/differentiable_stellar_spectroscopy/data/atlases/arcturus/ir")
    parser.add_argument("--page", type=Path, default=atlas / "ab5000_")
    parser.add_argument("--report", type=Path, default=root / "docs/arcturus_ab5000_fit_star.json",
                        help="the page's full-window fit, used to hold everything else fixed")
    parser.add_argument("--epoch", choices=("summer", "winter"), default="summer")
    parser.add_argument("--v1", type=float, default=5005.0)
    parser.add_argument("--v2", type=float, default=5025.0)
    parser.add_argument("--sub-width-cm1", type=float, default=5.0)
    parser.add_argument("--margin-cm1", type=float, default=25.0)
    parser.add_argument("--resolving-power", type=float, default=100_000.0)
    parser.add_argument("--samples-per-resolution", type=float, default=4.0)
    parser.add_argument("--profile", type=Path, default=root / "data/profiles/kitt_peak_1994.csv")
    parser.add_argument("--stellar", type=Path,
                        default=root / "data/stellar/arcturus_payne_zero_1960_2130nm.npz")
    parser.add_argument("--layer-chunk-size", type=int, default=3)
    parser.add_argument("--output", type=Path, default=root / "docs/arcturus_subwindows_ab5000.json")
    args = parser.parse_args()

    full = json.loads(args.report.read_text())
    held = full["parameters"]
    page = read_arcturus_page(args.page, args.epoch).select(args.v1, args.v2)
    profile = load_atmosphere_csv(args.profile)
    grid = igrins_wavenumber_grid(
        1.0e7 / args.v2, 1.0e7 / args.v1, resolving_power=args.resolving_power,
        samples_per_resolution=args.samples_per_resolution, margin_cm1=args.margin_cm1,
    )
    line_root = root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule"
    databases = {}
    for species, molecule_id in sorted(MOLECULE_IDS.items()):
        name = f"{molecule_id:02d}_{species}"
        databases[species] = AERLineDatabase(
            line_root / name / name, species, (float(grid[0]), float(grid[-1])), margin_cm1=0.0
        )
    opacity = ExoJAXOpacityBackend.prepare(
        databases, grid, methods="direct_sparse",
        temperature_range_k=(float(np.min(profile.temperature_k)), float(np.max(profile.temperature_k))),
        maximum_pressure_bar=float(np.max(profile.pressure_layer_bar)),
        vectorize_layers=True, mixed_precision=True, pressure_shift=True,
        layer_chunk_size=args.layer_chunk_size,
    )
    continuum = MTCKDWaterContinuum.from_netcdf(
        root / "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", grid
    )
    instrument = BoxcarFTSInstrumentProfile(
        mopd_cm=full["physics"]["mopd_cm"],
        wavenumber_center_cm1=float(0.5 * (args.v1 + args.v2)),
        max_residual_sigma_kms=4.0,
    )
    model = TelluricModel(
        profile, grid, opacity, continuum=continuum, accuracy_mode="mt_ckd",
        max_lsf_sigma_kms=4.0, pixel_integration="point", instrument=instrument,
    )
    source = prepare_stellar_source(
        StellarSpectrum.from_npz(args.stellar), model, vsini_kms=2.0, macroturbulence_kms=2.15
    )
    order = arcturus_spectral_order(page, source_flux_model_grid=source)
    wavenumber = 1.0e7 / np.asarray(order.wavelength_vacuum_nm)

    degree = len(held["continuum_coeffs"]) - 1
    start = TelluricParameters(
        log_column_scales={name: held[name] for name in model.species},
        velocity_kms=held["velocity_kms"], wavelength_stretch=0.0,
        lsf_sigma_kms=held["lsf_sigma_kms"],
        continuum_coeffs=np.asarray(held["continuum_coeffs"]),
        log_jitter=held["log_jitter"],
        stellar_velocity_kms=held["stellar_velocity_kms"],
    )
    fixed = 0.0
    bounds = {
        "H2O": (-2.0, 2.0),
        "CO2": (held["CO2"], held["CO2"]),
        "velocity_kms": (held["velocity_kms"], held["velocity_kms"]),
        "stellar_velocity_kms": (held["stellar_velocity_kms"], held["stellar_velocity_kms"]),
        "wavelength_stretch": (fixed, fixed),
        "lsf_sigma_kms": (held["lsf_sigma_kms"], held["lsf_sigma_kms"]),
        "log_jitter": (held["log_jitter"], held["log_jitter"]),
    }
    # Only the constant continuum term stays free, so it can absorb a
    # sub-window's own level without being able to reshape the band.
    for index in range(degree + 1):
        value = float(held["continuum_coeffs"][index])
        bounds[f"continuum_{index}"] = (-2.0, 2.0) if index == 0 else (value, value)

    edges = np.arange(args.v1, args.v2 + 1e-9, args.sub_width_cm1)
    results = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        window = (wavenumber >= lower) & (wavenumber < upper)
        sub_mask = np.asarray(order.mask) & window
        if sub_mask.sum() < 40:
            continue
        sub = SpectralOrder(
            order.wavelength_vacuum_nm, order.flux, order.uncertainty, mask=sub_mask,
            zenith_angle_deg=order.zenith_angle_deg,
            source_flux_model_grid=order.source_flux_model_grid,
        )
        result = fit_order(model, sub, start, bounds)
        depth = 1.0 - np.asarray(result.transmission)
        model_nu = np.asarray(model.wavenumber_cm1)
        inside = (model_nu >= lower) & (model_nu < upper)
        entry = {
            "wavenumber_cm1": [float(lower), float(upper)],
            "center_cm1": float(0.5 * (lower + upper)),
            "pixels": int(sub_mask.sum()),
            "h2o_scale": float(np.exp(result.parameters.log_column_scales["H2O"])),
            "success": bool(result.success),
            "residual_rms": float(np.sqrt(np.mean(np.asarray(result.residuals)[sub_mask] ** 2))),
            "mean_telluric_depth": float(np.mean(depth[inside])),
            "max_telluric_depth": float(np.max(depth[inside])),
            "saturated_fraction": float(np.mean(depth[inside] > 0.95)),
        }
        results.append(entry)
        print(f"  {lower:7.1f}-{upper:<7.1f} H2O {entry['h2o_scale']:.4f}  "
              f"depth {entry['mean_telluric_depth']:.3f}  rms {entry['residual_rms']:.4f}  "
              f"({entry['pixels']} px)")

    scales = np.asarray([r["h2o_scale"] for r in results])
    report = {
        "page": args.page.name, "epoch": args.epoch,
        "window_cm1": [args.v1, args.v2],
        "sub_width_cm1": args.sub_width_cm1,
        "full_window_h2o_scale": float(np.exp(held["H2O"])),
        "subwindow_mean": float(scales.mean()),
        "subwindow_std": float(scales.std(ddof=1)) if scales.size > 1 else 0.0,
        "subwindow_peak_to_peak": float(scales.max() - scales.min()),
        "subwindow_fractional_spread": float((scales.max() - scales.min()) / scales.mean()),
        "subwindows": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"full-window H2O {report['full_window_h2o_scale']:.4f}; sub-windows "
          f"{report['subwindow_mean']:.4f} +- {report['subwindow_std']:.4f}, "
          f"spread {100 * report['subwindow_fractional_spread']:.1f}%")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
