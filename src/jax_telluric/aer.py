"""AER 100-character line-file adapter for ExoJAX."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .types import AtmosphereProfile


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


_FIELD_BYTES = b"0123456789 .+-DdEe"
_NUMERIC_BYTE = np.zeros(256, dtype=bool)
_NUMERIC_BYTE[np.frombuffer(_FIELD_BYTES, dtype=np.uint8)] = True
_DIGIT_BYTE = np.zeros(256, dtype=bool)
_DIGIT_BYTE[np.frombuffer(b"0123456789", dtype=np.uint8)] = True

# One parsed index per file, keyed by identity and modification time. An AER
# per-molecule file holds up to a million records and a page needs a few
# thousand, yet the file carries no index, so every read has to look at every
# record. Doing that once per process turns a 3 s scan into a 1 ms mask for
# every page after the first. At most a handful of molecules are ever open, so
# the cache is small and is not bounded.
_FILE_INDEX: dict[tuple, object] = {}


def _build_file_index(path: Path):
    """Offset, molecule id and wavenumber of every line that could be a record.

    Returns ``None`` when the file does not fit this path, and the caller falls
    back to reading it a line at a time. Anything that cannot be classified with
    certainty is marked uncertain and always offered to the ordinary parser, so
    this narrows the work without ever making an accept or reject decision.
    """

    # Memory mapped, so repeated reads come from the page cache rather than
    # another 83 MB copy, and the index can keep it without holding the file
    # resident itself.
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    if raw.size == 0 or np.any(raw == 13):  # carriage returns change line lengths
        return None
    ends = np.flatnonzero(raw == 10)
    if ends.size == 0:
        return None
    if ends[-1] != raw.size - 1:
        ends = np.append(ends, raw.size)
    starts = np.empty(ends.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    # The parser sees the trailing newline, so its length test is one more.
    offsets = starts[(ends - starts) >= 66]

    first, second = raw[offsets], raw[offsets + 1]
    digit_first, digit_second = _DIGIT_BYTE[first], _DIGIT_BYTE[second]
    space_first, space_second = first == 32, second == 32
    # int() strips whitespace, so the field is read the same way it writes:
    # "12", " 2" and "2 " all give 2 or 12. AER pads to the right, so " 2" is
    # the common form and getting it wrong leaves every line unclassified.
    # Anything else -- a sign, a tab, a stray character -- stays uncertain.
    value_first = first.astype(np.int64) - 48
    value_second = second.astype(np.int64) - 48
    identifier = np.full(offsets.size, -1, dtype=np.int64)
    np.copyto(identifier, value_first * 10 + value_second, where=digit_first & digit_second)
    np.copyto(identifier, value_second, where=space_first & digit_second)
    np.copyto(identifier, value_first, where=digit_first & space_second)

    field = raw[offsets[:, None] + np.arange(3, 15)[None, :]]
    certain_nu = _NUMERIC_BYTE[field].all(axis=1) & _DIGIT_BYTE[field].any(axis=1)
    np.copyto(field, 69, where=(field == 68) | (field == 100))  # Fortran D exponent
    wavenumber = np.full(offsets.size, np.nan)
    if certain_nu.any():
        try:
            wavenumber[certain_nu] = (
                np.ascontiguousarray(field[certain_nu]).view("S12").ravel().astype(float)
            )
        except ValueError:
            return None
    return raw, offsets, identifier, np.where(certain_nu, wavenumber, np.nan)


def _file_index(path: Path):
    status = path.stat()
    key = (str(path.resolve()), status.st_size, status.st_mtime_ns)
    if key not in _FILE_INDEX:
        _FILE_INDEX[key] = _build_file_index(path)
    return _FILE_INDEX[key]


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

        path = Path(path)
        index = _file_index(path)
        if index is None:
            with path.open(encoding="ascii", errors="replace") as stream:
                source = stream.readlines()
            candidates = None
        else:
            raw, offsets, identifier, wavenumber = index
            # Reject only what is certainly out: an unrecognised field leaves
            # NaN or -1 here, and neither comparison is true, so it survives.
            keep = ~((identifier == _MOLECULE_IDS[molecule])
                     & ((wavenumber < lower - margin_cm1) | (wavenumber > upper + margin_cm1)))
            keep &= (identifier == _MOLECULE_IDS[molecule]) | (identifier == -1)
            candidates = offsets[keep]
        if candidates is not None:
            # Each candidate is decoded on its own; every field lies in the
            # first 67 characters and a record is at most a hundred.
            source = (raw[offset : offset + 101].tobytes().decode("ascii", errors="replace")
                      for offset in candidates)

        records: list[tuple[float, ...]] = []
        for line in source:
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

    _LINE_ARRAYS = (
        "isoid", "nu_lines", "line_strength_ref_original", "logsij0", "A",
        "gamma_air", "gamma_self", "elower", "n_air", "delta_air",
    )

    def restrict(self, keep) -> "AERLineDatabase":
        """Return a copy holding only the lines selected by a boolean mask."""

        keep = np.asarray(keep, dtype=bool)
        if keep.shape != self.nu_lines.shape:
            raise ValueError("the mask must have one entry per line")
        if not keep.any():
            raise ValueError("the mask keeps no lines")
        restricted = copy.copy(self)
        for name in self._LINE_ARRAYS:
            value = getattr(self, name)
            setattr(restricted, name, (jnp.asarray(value)[keep] if isinstance(value, jnp.ndarray)
                                       else np.asarray(value)[keep]))
        restricted.uniqiso = np.unique(restricted.isoid)
        return restricted

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


def line_optical_depth_bound(
    database: AERLineDatabase,
    profile: AtmosphereProfile,
    species: str,
    maximum_column_scale: float = 1.0,
) -> np.ndarray:
    """Upper bound on the optical depth each line can reach anywhere.

    A Voigt profile is a convolution, so its peak cannot exceed either
    component's peak. Bounding it by the smaller of the Doppler and Lorentz
    peaks and summing the layers gives a quantity no line can exceed at any
    wavenumber, which is what makes a discard budget a guarantee rather than a
    heuristic. ``maximum_column_scale`` covers the largest column a fit may
    reach, since the bound must hold there too.
    """

    from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
    from exojax.database.core.line_strength import line_strength
    from exojax.utils.constants import Tref_original

    if maximum_column_scale <= 0.0:
        raise ValueError("maximum_column_scale must be positive")
    species = species.upper()
    if species not in profile.vmr:
        raise ValueError(f"the atmosphere has no {species} profile")
    vmr = np.asarray(profile.vmr[species]) * float(maximum_column_scale)
    column = np.asarray(profile.air_column_cm2)
    total = np.zeros(np.asarray(database.nu_lines).size)
    for index, (temperature, pressure) in enumerate(
        zip(np.asarray(profile.temperature_k), np.asarray(profile.pressure_layer_bar))
    ):
        sigma = np.asarray(doppler_sigma(database.nu_lines, temperature, database.molmass))
        gamma = np.asarray(
            gamma_hitran(pressure, temperature, pressure * vmr[index], database.n_air,
                         database.gamma_air, database.gamma_self)
            + gamma_natural(database.A)
        )
        strength = np.asarray(line_strength(
            temperature, database.logsij0, database.nu_lines, database.elower,
            database.qr_interp(database.isotope, temperature, Tref_original), Tref_original,
        ))
        peak = np.minimum(1.0 / (np.sqrt(2.0 * np.pi) * sigma), 1.0 / (np.pi * gamma))
        total += strength * peak * column[index] * vmr[index]
    return total


def select_significant_lines(
    database: AERLineDatabase,
    profile: AtmosphereProfile,
    species: str,
    optical_depth_budget: float = 1.0e-3,
    maximum_column_scale: float = 1.0,
) -> AERLineDatabase:
    """Drop the weakest lines whose summed influence stays under a budget.

    AER line lists span some ten orders of magnitude in strength, and the
    opacity kernel costs the same for every line. Discarding the weakest lines
    whose bounds sum to ``optical_depth_budget`` changes the optical depth by at
    most that much anywhere, and therefore the transmission by at most that much
    -- a worst case that assumes every discarded line peaks at the same
    wavenumber, which they do not, so the realized error is far smaller.

    A budget of zero keeps every line. Nothing here depends on the fitted
    parameters, so the selection is made once, before the grid is built.
    """

    if optical_depth_budget < 0.0:
        raise ValueError("optical_depth_budget must not be negative")
    if optical_depth_budget == 0.0:
        return database
    bound = line_optical_depth_bound(database, profile, species, maximum_column_scale)
    order = np.argsort(bound)
    cumulative = np.cumsum(bound[order])
    discard = int(np.searchsorted(cumulative, optical_depth_budget, side="right"))
    if discard >= bound.size:
        # Every line together stays under the budget, yet a species with no
        # lines cannot be modelled. Keep the strongest one and let the caller's
        # own optical-depth test decide whether to free it.
        discard = bound.size - 1
    if discard == 0:
        return database
    keep = np.ones(bound.size, dtype=bool)
    keep[order[:discard]] = False
    return database.restrict(keep)
