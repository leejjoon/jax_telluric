#!/usr/bin/env python
"""Render a browsable review of the batch telluric corrections.

Produces one diagnostic panel per page-epoch plus an index, so that a run
covering hundreds of pages can be triaged by eye rather than by reading JSON.
Images are written as separate files rather than embedded, because a few
hundred of them will not fit inside one HTML document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def panel(record: dict, npz: Path, out: Path) -> dict:
    d = np.load(npz)
    nu, mask, ok = d["wavenumber_cm1"], d["mask"], d["reliable"]
    obs, corrected, star = d["observed"], d["corrected"], d["stellar_only"]
    transmission, residual = d["transmission"], d["residual"]
    sigma = record["pixel_sigma"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 2.1, 1.1], "hspace": 0.08})

    def flag(ax):
        bad = ~ok
        if bad.any():
            e = np.flatnonzero(np.diff(np.concatenate([[0], bad.view(np.int8), [0]])))
            for a, b in zip(e[::2], e[1::2]):
                ax.axvspan(nu[a], nu[min(b, len(nu) - 1)], color="#fecaca", alpha=0.55, zorder=0)

    ax = axes[0]; flag(ax)
    ax.plot(nu, obs, color="0.3", lw=0.7, label="observed")
    ax.plot(nu, np.where(mask, d["model_flux"], np.nan), color="#c2410c", lw=0.9, label="model")
    ax.set_ylabel("flux", fontsize=8); ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9, ncol=2)
    ax.tick_params(labelsize=7)

    ax = axes[1]; flag(ax)
    ax.plot(nu, np.where(ok, corrected, np.nan), color="#16a34a", lw=0.9, label="corrected")
    ax.plot(nu, np.where(mask, star, np.nan), color="0.45", lw=0.8, ls="--", alpha=0.8, label="stellar model")
    ax.plot(nu, transmission, color="#1d4ed8", lw=0.6, alpha=0.5, label="transmission")
    ax.set_ylabel("flux", fontsize=8); ax.set_ylim(-0.05, 1.3)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9, ncol=3)
    ax.tick_params(labelsize=7)

    ax = axes[2]; flag(ax)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.plot(nu, np.where(mask, residual / sigma, np.nan), color="#7c3aed", lw=0.6)
    ax.set_ylabel(r"resid/$\sigma$", fontsize=8)
    ax.set_xlabel("vacuum wavenumber (cm$^{-1}$)", fontsize=8)
    ax.set_xlim(nu.min(), nu.max()); ax.tick_params(labelsize=7)

    axes[0].set_title(
        f"{record['page']} {record['epoch']}   {nu.min():.1f}-{nu.max():.1f} cm$^{{-1}}$   "
        f"free: {'+'.join(record['free_species'])}   "
        f"rms/noise {record['residual_rms_over_noise']:.2f}   "
        f"median T {record['median_transmission']:.3f}", fontsize=9)
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"width": 11, "height": 6.4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--summary", type=Path, default=root / "docs/arcturus_atlas_summary.json")
    parser.add_argument("--npz-dir", type=Path, default=root / "data/corrected/atlas")
    parser.add_argument("--out-dir", type=Path, default=root / "data/corrected/review")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "panels").mkdir(exist_ok=True)

    summary = json.loads(args.summary.read_text())
    rows = []
    for record in summary["results"]:
        if "error" in record:
            rows.append({**{k: record[k] for k in ("page", "epoch")}, "error": record["error"]})
            continue
        npz = args.npz_dir / f"{record['page']}_{record['epoch']}.npz"
        if not npz.exists():
            continue
        name = f"{record['page']}_{record['epoch']}.png"
        panel(record, npz, args.out_dir / "panels" / name)
        rows.append({
            "page": record["page"], "epoch": record["epoch"],
            "v1": record["wavenumber_cm1"][0], "v2": record["wavenumber_cm1"][1],
            "wavelength_um": round(2.0e4 / (record["wavenumber_cm1"][0] + record["wavenumber_cm1"][1]), 3),
            "free_species": record["free_species"],
            "median_transmission": round(record["median_transmission"], 4),
            "rms_over_noise": round(record["residual_rms_over_noise"], 3),
            "reduced_chi2": round(record["reduced_chi2"], 1),
            "reliable_fraction": round(record["reliable"] / record["pixels"], 4),
            "converged": record["all_stages_converged"],
            "stellar_velocity_kms": round(record["parameters"]["stellar_velocity_kms"], 2),
            "velocity_kms": round(record["parameters"]["velocity_kms"], 3),
            "h2o": round(float(np.exp(record["parameters"].get("H2O", 0.0))), 4),
            "grid_points": record["grid_points"],
            "image": f"panels/{name}",
        })
    index = {"settings": summary.get("settings", {}), "generated": summary.get("generated"),
             "rows": rows}
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    good = [r for r in rows if "error" not in r]
    print(f"rendered {len(good)} panels, {len(rows) - len(good)} failures -> {args.out_dir}")
    if good:
        v = np.array([r["rms_over_noise"] for r in good])
        print(f"rms/noise: median {np.median(v):.2f}, range {v.min():.2f}-{v.max():.2f}")


if __name__ == "__main__":
    main()
