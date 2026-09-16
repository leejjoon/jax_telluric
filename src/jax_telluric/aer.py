"""AER 100-character line-file adapter for ExoJAX."""

from __future__ import annotations

from contextlib import redirect_stdout
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np


_MOLECULE_IDS = {"H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5, "CH4": 6, "O2": 7}
_MEAN_MOLAR_MASS = {
    "H2O": 18.01528,
    "CO2": 44.0095,
    "O3": 47.9982,
    "N2O": 44.0128,
    "CO": 28.0101,
    "CH4": 16.0425,
    "O2": 31.9988,
}


def _fortran_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


class AERLineDatabase:
    """Minimal HITRAN-like database backed by an AER per-molecule line file.

    The adapter lets :class:`exojax.opacity.OpaDirect` use the same ordinary
    Voigt-line parameters supplied to LNFL. AER line-coupling records are not
    exposed by this adapter. The auxiliary speed-dependence data are for
    MonoRTM and are not used by the LBLRTM configuration validated here.
    """

    dbtype = "hitran"
    isotope = 0

    def __init__(
        self,
        path: str | Path,
        molecule: str,
        wavenumber_range_cm1: tuple[float, float],
        margin_cm1: float = 25.0,
        strength_cutoff: float = 0.0,
    ) -> None:
        molecule = molecule.upper()
        if molecule not in _MOLECULE_IDS:
            raise ValueError(f"unsupported AER molecule: {molecule}")
        lower, upper = wavenumber_range_cm1
        if not 0.0 < lower < upper or margin_cm1 < 0.0 or strength_cutoff < 0.0:
            raise ValueError("invalid line-selection range, margin, or cutoff")

        records: list[tuple[float, ...]] = []
        with Path(path).open(encoding="ascii", errors="replace") as stream:
            for line in stream:
                # Some AER per-molecule files (notably CH4) omit the header
                # and % delimiter. Recognize records by their numeric fields.
                if len(line) < 67 or not line[:2].strip():
                    continue
                try:
                    molecule_id = int(line[0:2])
                    isotope_id = int(line[2:3])
                    nu = _fortran_float(line[3:15])
                    strength = _fortran_float(line[15:25])
                except ValueError:
                    continue
                if molecule_id != _MOLECULE_IDS[molecule] or not (lower - margin_cm1 <= nu <= upper + margin_cm1):
                    continue
                if strength < strength_cutoff:
                    continue
                records.append(
                    (
                        isotope_id,
                        nu,
                        strength,
                        _fortran_float(line[25:35]),
                        _fortran_float(line[35:40]),
                        _fortran_float(line[40:45]),
                        _fortran_float(line[45:55]),
                        _fortran_float(line[55:59]),
                        _fortran_float(line[59:67]),
                    )
                )
        if not records:
            raise ValueError(f"no {molecule} lines found in the requested range")
        values = np.asarray(records)
        self.simple_molecule_name = molecule
        self.molecid = _MOLECULE_IDS[molecule]
        self.isoid = values[:, 0].astype(int)
        self.uniqiso = np.unique(self.isoid)
        self.nu_lines = values[:, 1]
        self.line_strength_ref_original = values[:, 2]
        self.logsij0 = jnp.log(self.line_strength_ref_original)
        self.A = values[:, 3]
        self.gamma_air = values[:, 4]
        self.gamma_self = values[:, 5]
        self.elower = values[:, 6]
        self.n_air = values[:, 7]
        self.delta_air = values[:, 8]
        self.molmass = _MEAN_MOLAR_MASS[molecule]
        self._load_partition_functions()

    def _load_partition_functions(self) -> None:
        with redirect_stdout(open(os.devnull, "w")):
            import hapi
        self._partition_temperature = {
            iso: jnp.asarray(hapi.TIPS_2017_ISOT_HASH[(self.molecid, int(iso))]) for iso in self.uniqiso
        }
        self._partition_value = {
            iso: jnp.asarray(hapi.TIPS_2017_ISOQ_HASH[(self.molecid, int(iso))]) for iso in self.uniqiso
        }

    def qr_interp(self, isotope: int, temperature, reference_temperature):
        """Return per-line TIPS partition-function ratios."""

        del isotope
        ratio = jnp.ones(len(self.nu_lines))
        isotope_ids = jnp.asarray(self.isoid)
        for iso in self.uniqiso:
            temperatures = self._partition_temperature[iso]
            values = self._partition_value[iso]
            current = jnp.interp(temperature, temperatures, values) / jnp.interp(
                reference_temperature, temperatures, values
            )
            ratio = jnp.where(isotope_ids == iso, current, ratio)
        return ratio
