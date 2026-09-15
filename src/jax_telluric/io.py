"""File adapters for reproducible atmospheric inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .types import AtmosphereProfile

_REQUIRED_COLUMNS = {
    "pressure_top_bar",
    "pressure_bottom_bar",
    "temperature_k",
    "altitude_km",
}
_NON_SPECIES_COLUMNS = _REQUIRED_COLUMNS | {"mean_molecular_weight_g_mol", "gravity_m_s2"}


def load_atmosphere_csv(path: str | Path) -> AtmosphereProfile:
    """Load a top-to-bottom layer profile from a CSV file.

    Any column outside the required/physical columns is interpreted as a
    molecule name containing a volume mixing ratio.
    """

    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(row for row in stream if not row.lstrip().startswith("#"))
        columns = set(reader.fieldnames or ())
        missing = _REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"atmosphere CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("atmosphere CSV has no layers")

    top = np.asarray([float(row["pressure_top_bar"]) for row in rows])
    bottom = np.asarray([float(row["pressure_bottom_bar"]) for row in rows])
    if not np.allclose(bottom[:-1], top[1:], rtol=1.0e-10, atol=0.0):
        raise ValueError("adjacent pressure edges are not contiguous")
    edges = np.concatenate([top[:1], bottom])
    species = sorted(columns - _NON_SPECIES_COLUMNS)
    if not species:
        raise ValueError("atmosphere CSV must include at least one molecule VMR column")

    def optional_column(name: str, default: float) -> np.ndarray:
        if name not in columns:
            return np.full(len(rows), default)
        return np.asarray([float(row[name]) for row in rows])

    return AtmosphereProfile(
        pressure_edges_bar=edges,
        temperature_k=[float(row["temperature_k"]) for row in rows],
        altitude_km=[float(row["altitude_km"]) for row in rows],
        vmr={name.upper(): [float(row[name]) for row in rows] for name in species},
        mean_molecular_weight_g_mol=optional_column("mean_molecular_weight_g_mol", 28.9647),
        gravity_m_s2=optional_column("gravity_m_s2", 9.80665),
    )


def load_mipas_profile(
    path: str | Path,
    observatory_altitude_km: float = 0.0,
    species: tuple[str, ...] = ("H2O", "CO2", "CH4", "O2", "CO", "N2O"),
) -> AtmosphereProfile:
    """Convert a TelFit/RFM MIPAS level profile into model layers.

    The source profile uses altitude in km, pressure in hPa, temperature in K,
    and molecular abundances in ppmv. Levels below the observatory are removed.
    Adjacent retained levels define one hydrostatic model layer.
    """

    variables: dict[str, list[float]] = {}
    current: str | None = None
    with Path(path).open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            if line.startswith("*"):
                name = line[1:].split()[0].upper()
                if name == "END":
                    break
                current = name
                variables[current] = []
                continue
            if current is not None:
                variables[current].extend(float(value) for value in line.split())

    required = {"HGT", "PRE", "TEM", *(name.upper() for name in species)}
    missing = required - set(variables)
    if missing:
        raise ValueError(f"MIPAS profile is missing variables: {', '.join(sorted(missing))}")
    lengths = {len(variables[name]) for name in required}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("MIPAS variables must have the same number of levels")

    altitude = np.asarray(variables["HGT"])
    keep = altitude >= observatory_altitude_km
    if np.count_nonzero(keep) < 2:
        raise ValueError("fewer than two MIPAS levels remain above the observatory")
    indices = np.flatnonzero(keep)
    if np.any(np.diff(indices) != 1):
        raise ValueError("MIPAS altitude levels are not monotonic")

    # MIPAS is ground-to-top; the model interface is top-to-bottom.
    altitude_levels = altitude[keep][::-1]
    pressure_edges = np.asarray(variables["PRE"])[keep][::-1] * 1.0e-3
    temperature_levels = np.asarray(variables["TEM"])[keep][::-1]
    vmr = {}
    for name in species:
        levels = np.asarray(variables[name.upper()])[keep][::-1] * 1.0e-6
        vmr[name.upper()] = 0.5 * (levels[:-1] + levels[1:])
    return AtmosphereProfile(
        pressure_edges_bar=pressure_edges,
        temperature_k=0.5 * (temperature_levels[:-1] + temperature_levels[1:]),
        altitude_km=0.5 * (altitude_levels[:-1] + altitude_levels[1:]),
        vmr=vmr,
    )
