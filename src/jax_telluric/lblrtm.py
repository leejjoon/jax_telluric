"""Reproducible LBLRTM reference runs for validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .reference import LBLRTMSpectrum, read_tape12_single_precision
from .types import AtmosphereProfile


_LBLRTM_SPECIES = ("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2")


@dataclass(frozen=True)
class LBLRTMRunConfig:
    """Settings for a ground-to-space, user-profile LBLRTM calculation."""

    wavenumber_min_cm1: float
    wavenumber_max_cm1: float
    zenith_angle_deg: float = 0.0
    continuum_flag: int = 1
    description: str = "jax-telluric reference atmosphere"

    def __post_init__(self) -> None:
        if not 0.0 < self.wavenumber_min_cm1 < self.wavenumber_max_cm1:
            raise ValueError("wavenumber limits must be positive and increasing")
        if not 0.0 <= self.zenith_angle_deg < 90.0:
            raise ValueError("zenith angle must be in [0, 90) degrees")
        if self.continuum_flag not in range(6):
            raise ValueError("continuum_flag must be one of 0, 1, 2, 3, 4, or 5")


def write_tape5(
    path: str | Path,
    profile: AtmosphereProfile,
    config: LBLRTMRunConfig,
) -> None:
    """Write a fixed-format TAPE5 for an atmospheric transmission run.

    The layer-center pressure, temperature, altitude, and VMR values are used
    as LBLRTM user-profile levels. LBLRTM interpolates those levels to its path
    layers. Abundances use the ``A`` unit code (ppmv by volume).
    """

    available = set(profile.vmr)
    unsupported = available - set(_LBLRTM_SPECIES)
    if unsupported:
        raise ValueError(f"unsupported LBLRTM TAPE3 species: {', '.join(sorted(unsupported))}")
    nlayers = len(profile.temperature_k)
    if nlayers < 2:
        raise ValueError("LBLRTM user profiles need at least two levels")

    # LBLRTM expects user profile levels from the observer upward.
    altitude = np.asarray(profile.altitude_km)[::-1]
    pressure_hpa = np.asarray(profile.pressure_layer_bar)[::-1] * 1000.0
    temperature = np.asarray(profile.temperature_k)[::-1]
    abundance_ppmv = np.column_stack(
        [np.asarray(profile.vmr.get(name, np.zeros(nlayers)))[::-1] * 1.0e6 for name in _LBLRTM_SPECIES]
    )
    observer_altitude = float(altitude[0])
    space_altitude = float(altitude[-1])

    lines = [f"${config.description[:79]}"]
    flags = (1, 1, config.continuum_flag, 0, 1, 0, 0, 0, 0, 1)
    lines.append("".join(f"{value:5d}" for value in flags) + f"{0:5d}{0:5d}{0:5d}{0:5d}{0:5d}{0:5d}")
    lines.append(
        f"{config.wavenumber_min_cm1:10.3f}{config.wavenumber_max_cm1:10.3f}"
        f"{4.0:10.3f}{0.0:10.3f}{0.04:10.3f}{36.0:10.3f}{-1.0:10.3f}{-1.0:10.3f}"
        f"{0:5d}{0.0:15.3f}{0:5d}"
    )
    lines.append(
        f"{float(temperature[0]):10.3f}{1.0:10.3f}{0.0:10.3f}{0.0:10.3f}"
        f"{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}    s"
    )
    # MODEL=0, ITYPE=3, IBMAX=0, NMOL=7. HSPACE is the top supplied level.
    lines.append(
        f"{0:5d}{3:5d}{0:5d}{0:5d}{0:5d}{len(_LBLRTM_SPECIES):5d}{0:5d}"
        f"{0:2d} {0:2d}{0.0:10.3f}{space_altitude:10.3f}"
        f"{0.5 * (config.wavenumber_min_cm1 + config.wavenumber_max_cm1):10.3f}"
        f"{'':10s}{0.0:10.3f}"
    )
    lines.append(
        f"{observer_altitude:10.3f}{0.0:10.3f}{config.zenith_angle_deg:10.3f}"
        f"{0.0:10.3f}{0.0:10.3f}{0:5d}{'':5s}{observer_altitude:10.3f}"
    )
    lines.append(f"{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{observer_altitude:10.3f}{space_altitude:10.3f}")
    lines.append(f"{nlayers:5d}{' jax-telluric profile':24s}")
    for z_km, pressure, temp, abundances in zip(altitude, pressure_hpa, temperature, abundance_ppmv):
        lines.append(f"{z_km:10.3E}{pressure:10.3E}{temp:10.3E}     AA L AAAAAAA")
        lines.append("".join(f"{value:15.8E}" for value in abundances))
    lines.extend((f"{-1.0:4.1f}", f"{-1.0:4.1f}", "%"))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def run_lblrtm(
    workdir: str | Path,
    profile: AtmosphereProfile,
    config: LBLRTMRunConfig,
    executable: str | Path,
    tape3: str | Path,
    mt_ckd_data: str | Path,
) -> LBLRTMSpectrum:
    """Create an isolated LBLRTM run directory, execute it, and read TAPE12."""

    directory = Path(workdir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_tape5(directory / "TAPE5", profile, config)
    for source, name in ((tape3, "TAPE3"), (mt_ckd_data, "absco-ref_wv-mt-ckd.nc")):
        destination = directory / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(Path(source).resolve())
    binary = directory / "lblrtm"
    shutil.copy2(executable, binary)
    tape12 = directory / "TAPE12"
    if tape12.exists():
        tape12.unlink()
    result = subprocess.run([str(binary)], cwd=directory, capture_output=True, text=True)
    if result.returncode != 0 or not tape12.exists():
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"LBLRTM failed with status {result.returncode}:\n{details[-4000:]}")
    return read_tape12_single_precision(tape12)
