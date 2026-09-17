#!/usr/bin/env python
"""Build a hydrostatic layer profile CSV for an observing site and epoch.

The packaged ``example_midlatitude.csv`` is illustrative and carries present-day
trace-gas abundances. Retrievals that lean on a well-mixed species to break the
airmass degeneracy need the abundances of the epoch the data were taken in: CO2
rose from 357 ppm in 1993-94 to about 420 ppm, and that 17.6% error would map
one for one into the retrieved airmass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


_BOLTZMANN_ERG_K = 1.380649e-16
_AVOGADRO = 6.02214076e23
_EARTH_RADIUS_KM = 6371.0
_DRY_MOLAR_MASS_G_MOL = 28.9647
_WATER_MOLAR_MASS_G_MOL = 18.01528
_SEA_LEVEL_GRAVITY = 9.80665

# Dry-air volume mixing ratios. The 1993-94 column is the epoch of the Hinkle,
# Wallace & Livingston Arcturus atlas; see docs/arcturus_fit.md.
EPOCH_DRY_VMR = {
    "1994": {"CO2": 357.0e-6, "CH4": 1.72e-6, "N2O": 0.310e-6, "CO": 0.10e-6, "O2": 0.2095},
    "2020": {"CO2": 414.0e-6, "CH4": 1.87e-6, "N2O": 0.333e-6, "CO": 0.10e-6, "O2": 0.2095},
}

# Height above the site, in km. Fine near the ground where the water sits, and
# with four layers above 0.25 bar where the well-mixed line cores form.
DEFAULT_EDGES_KM = (0.0, 0.3, 0.7, 1.2, 1.8, 2.6, 3.6, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0)


def _gravity(altitude_km: np.ndarray) -> np.ndarray:
    return _SEA_LEVEL_GRAVITY * (_EARTH_RADIUS_KM / (_EARTH_RADIUS_KM + altitude_km)) ** 2


def build_profile(
    surface_pressure_hpa: float,
    surface_temperature_k: float,
    site_altitude_km: float,
    lapse_rate_k_km: float,
    tropopause_temperature_k: float,
    water_scale_height_km: float,
    stratospheric_water_vmr: float,
    precipitable_water_mm: float,
    dry_vmr: dict,
    edges_km: tuple = DEFAULT_EDGES_KM,
) -> dict:
    """Return per-layer arrays, ordered top to bottom as the model expects."""

    edges = np.asarray(edges_km, dtype=float)
    if edges.ndim != 1 or np.any(np.diff(edges) <= 0.0) or edges[0] != 0.0:
        raise ValueError("layer edges must start at the site and increase")

    # Integrate hydrostatic equilibrium on a fine sub-grid so the layer edges
    # are not sensitive to how coarsely the layers themselves are chosen.
    fine = np.linspace(0.0, edges[-1], 40001)
    fine_temperature = np.maximum(surface_temperature_k - lapse_rate_k_km * fine, tropopause_temperature_k)
    # CGS throughout: g/mol times cm/s2 over the gas constant in erg/(mol K).
    gravity_cm_s2 = _gravity(site_altitude_km + fine) * 100.0
    scale = _DRY_MOLAR_MASS_G_MOL * gravity_cm_s2 / (_AVOGADRO * _BOLTZMANN_ERG_K)
    integrand = scale / fine_temperature
    log_pressure = np.log(surface_pressure_hpa) - np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) * 1.0e5 * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    edge_pressure_hpa = np.exp(np.interp(edges, fine, log_pressure))

    center = 0.5 * (edges[:-1] + edges[1:])
    temperature = np.maximum(surface_temperature_k - lapse_rate_k_km * center, tropopause_temperature_k)
    thickness_cm = np.diff(edges) * 1.0e5
    center_pressure_hpa = np.exp(np.interp(center, fine, log_pressure))
    number_density = center_pressure_hpa * 1000.0 / (_BOLTZMANN_ERG_K * temperature)

    # An exponential water profile on a stratospheric floor, renormalized so the
    # vertical column equals the requested precipitable water.
    shape = np.exp(-center / water_scale_height_km)
    floor_column = stratospheric_water_vmr * number_density * thickness_cm
    target_column = precipitable_water_mm * 0.1 / _WATER_MOLAR_MASS_G_MOL * _AVOGADRO
    scaled = shape * thickness_cm
    remaining = target_column - float(np.sum(floor_column))
    if remaining <= 0.0:
        raise ValueError("requested precipitable water is below the stratospheric floor")
    water_column = scaled * remaining / float(np.sum(scaled)) + floor_column
    water_vmr = water_column / (number_density * thickness_cm)
    if np.any(water_vmr >= 1.0):
        raise ValueError("water vapour mixing ratio reached unity")

    # The public profile stores fractions of moist air, so the well-mixed
    # species are their dry-air values diluted by the local water.
    vmr = {"H2O": water_vmr}
    for species, dry in dry_vmr.items():
        vmr[species] = dry * (1.0 - water_vmr)

    order = np.argsort(center)[::-1]
    return {
        "pressure_top_bar": edge_pressure_hpa[1:][order] / 1000.0,
        "pressure_bottom_bar": edge_pressure_hpa[:-1][order] / 1000.0,
        "temperature_k": temperature[order],
        "altitude_km": (site_altitude_km + center)[order],
        "mean_molecular_weight_g_mol": np.full(len(order), _DRY_MOLAR_MASS_G_MOL),
        "gravity_m_s2": _gravity(site_altitude_km + center)[order],
        **{name: values[order] for name, values in vmr.items()},
        "_precipitable_water_mm": precipitable_water_mm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-pressure-hpa", type=float, default=786.0)
    parser.add_argument("--surface-temperature-k", type=float, default=283.0)
    parser.add_argument("--site-altitude-km", type=float, default=2.096)
    parser.add_argument("--lapse-rate-k-km", type=float, default=6.5)
    parser.add_argument("--tropopause-temperature-k", type=float, default=216.65)
    parser.add_argument("--water-scale-height-km", type=float, default=2.0)
    parser.add_argument("--stratospheric-water-vmr", type=float, default=5.0e-6)
    parser.add_argument("--precipitable-water-mm", type=float, default=5.0)
    parser.add_argument("--epoch", choices=sorted(EPOCH_DRY_VMR), default="1994")
    parser.add_argument("--site", default="Kitt Peak")
    parser.add_argument("--output", type=Path, default=Path("data/profiles/kitt_peak_1994.csv"))
    args = parser.parse_args()

    profile = build_profile(
        args.surface_pressure_hpa,
        args.surface_temperature_k,
        args.site_altitude_km,
        args.lapse_rate_k_km,
        args.tropopause_temperature_k,
        args.water_scale_height_km,
        args.stratospheric_water_vmr,
        args.precipitable_water_mm,
        EPOCH_DRY_VMR[args.epoch],
    )
    profile.pop("_precipitable_water_mm")
    columns = list(profile)
    rows = len(profile["temperature_k"])

    lines = [
        f"# {args.site}, {args.epoch} trace-gas abundances, generated by scripts/make_site_profile.py.",
        f"# surface {args.surface_pressure_hpa:.1f} hPa / {args.surface_temperature_k:.1f} K at "
        f"{args.site_altitude_km:.3f} km, lapse {args.lapse_rate_k_km:.1f} K/km.",
        f"# water: exponential scale height {args.water_scale_height_km:.1f} km on a "
        f"{args.stratospheric_water_vmr:.1e} stratospheric floor, {args.precipitable_water_mm:.2f} mm vertical.",
        "# VMR columns are moist-air volume mixing ratios, dimensionless.",
        ",".join(columns),
    ]
    for index in range(rows):
        values = []
        for name in columns:
            value = profile[name][index]
            values.append(f"{value:.6f}" if name in ("temperature_k", "altitude_km", "gravity_m_s2",
                                                     "mean_molecular_weight_g_mol") else f"{value:.8e}")
        lines.append(",".join(values))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {rows} layers")


if __name__ == "__main__":
    main()
