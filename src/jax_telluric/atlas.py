"""Reader for the Hinkle, Wallace & Livingston (1995) Arcturus IR atlas.

The atlas is published as fixed-width ASCII "pages", one per plot of the
monograph, each carrying an observed spectrum, the telluric transmission the
authors divided by, and the ratio, for two epochs. Reference:

    Hinkle, K., Wallace, L., & Livingston, W. 1995, PASP, 107, 1402

A loader for the same files exists in the sibling ``differentiable_stellar_-
spectroscopy`` project (``dss/data/atlases.py``). This one is deliberately
separate and self-contained: that project is not installable, its ``mask``
marks *bad* pixels where :class:`~jax_telluric.types.SpectralOrder` marks
usable ones, and its telluric floor of 0.5 is meant for spectra that have
already been divided -- here the deep lines are the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .types import SpectralOrder


ARCTURUS_REFERENCE = "Hinkle, K., Wallace, L., & Livingston, W. 1995, PASP, 107, 1402"

# Byte offsets of the 89-character records, from the atlas table.doc: one
# D11.4 wavenumber followed by six D13.5 columns.
_FIELD_EDGES = (0, 11, 24, 37, 50, 63, 76, 89)
_EPOCH_COLUMNS = {
    "summer": {"observed": 1, "telluric": 2, "ratioed": 3},
    "winter": {"observed": 4, "telluric": 5, "ratioed": 6},
}

# Doppler factors from the atlas readme.dat plotting example, which divides the
# wavenumber scale by these to reach the stellar rest frame. They are a useful
# starting value and an independent check, not a measurement: readme.dat says
# to "change ... Doppler correction as needed" per page.
READ_ME_DOPPLER_FACTOR = {"summer": 1.0000478, "winter": 0.9999129}


def epoch_velocity_kms(epoch: str) -> float:
    """Starting value for the stellar velocity of one epoch, in km/s.

    Positive means receding, which is the sign convention
    :meth:`TelluricModel.predict` uses for ``stellar_velocity_kms``.

    The magnitude comes from the readme.dat Doppler factor; the **sign is
    measured, not inferred**. Reading the factor as a division put the summer
    epoch at -14.33 km/s, but the deepest unmodelled absorption features in a
    telluric-only fit sit *below* the wavenumbers of the atlas's own atomic
    line identifications (appendix.a), which requires a redshift. Four matched
    lines in 5015-5020 cm-1 give +13.5 +- 0.5 km/s, and an independent
    cross-correlation of the two epochs gives +13.57. So the star is receding
    in summer and approaching in winter.

    Treat this as an initial value good to about 1 km/s, not a measurement.
    """

    if epoch not in READ_ME_DOPPLER_FACTOR:
        raise ValueError(f"unknown epoch: {epoch}")
    return 299792.458 * (READ_ME_DOPPLER_FACTOR[epoch] - 1.0)


def _fortran_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True)
class ArcturusPage:
    """One page of the atlas, for one epoch, ascending in wavenumber."""

    path: Path
    epoch: str
    wavenumber_vacuum_cm1: np.ndarray
    observed: np.ndarray
    telluric: np.ndarray
    ratioed: np.ndarray
    sha256: str

    def __post_init__(self) -> None:
        if self.epoch not in _EPOCH_COLUMNS:
            raise ValueError(f"epoch must be one of {', '.join(sorted(_EPOCH_COLUMNS))}")
        nu = np.asarray(self.wavenumber_vacuum_cm1, dtype=float)
        columns = {
            "observed": np.asarray(self.observed, dtype=float),
            "telluric": np.asarray(self.telluric, dtype=float),
            "ratioed": np.asarray(self.ratioed, dtype=float),
        }
        if nu.ndim != 1 or nu.size < 8:
            raise ValueError("an atlas page needs at least eight samples")
        if np.any(~np.isfinite(nu)) or np.any(np.diff(nu) <= 0.0):
            raise ValueError("atlas wavenumbers must be finite and strictly increasing")
        for name, values in columns.items():
            if values.shape != nu.shape:
                raise ValueError(f"the {name} column must match the wavenumber grid")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "wavenumber_vacuum_cm1", nu)
        object.__setattr__(self, "path", Path(self.path))

    @property
    def spacing_cm1(self) -> float:
        return float(np.median(np.diff(self.wavenumber_vacuum_cm1)))

    def select(self, wavenumber_min_cm1: float, wavenumber_max_cm1: float) -> "ArcturusPage":
        """Restrict to a wavenumber range.

        Pages overlap their neighbours by about 2 cm-1 and differ there by a
        per-page scalar, so a fit should be trimmed to one page's own range
        rather than blending across a seam.
        """

        if not wavenumber_min_cm1 < wavenumber_max_cm1:
            raise ValueError("the selected range must be increasing")
        keep = (self.wavenumber_vacuum_cm1 >= wavenumber_min_cm1) & (
            self.wavenumber_vacuum_cm1 <= wavenumber_max_cm1
        )
        if np.count_nonzero(keep) < 8:
            raise ValueError("the selected range leaves too few samples")
        return ArcturusPage(
            self.path, self.epoch, self.wavenumber_vacuum_cm1[keep], self.observed[keep],
            self.telluric[keep], self.ratioed[keep], self.sha256,
        )


def read_arcturus_page(path: str | Path, epoch: str = "summer") -> ArcturusPage:
    """Read one ``abNNNN`` page file for one epoch."""

    if epoch not in _EPOCH_COLUMNS:
        raise ValueError(f"epoch must be one of {', '.join(sorted(_EPOCH_COLUMNS))}")
    path = Path(path)
    content = path.read_bytes()
    rows = []
    for line in content.decode("ascii", errors="replace").splitlines():
        if len(line) < _FIELD_EDGES[-1] - 1 or not line[: _FIELD_EDGES[1]].strip():
            continue
        try:
            rows.append(
                [
                    _fortran_float(line[start:stop])
                    for start, stop in zip(_FIELD_EDGES[:-1], _FIELD_EDGES[1:])
                ]
            )
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no atlas records found in {path}")
    values = np.asarray(rows)
    columns = _EPOCH_COLUMNS[epoch]
    return ArcturusPage(
        path=path,
        epoch=epoch,
        wavenumber_vacuum_cm1=values[:, 0],
        observed=values[:, columns["observed"]],
        telluric=values[:, columns["telluric"]],
        ratioed=values[:, columns["ratioed"]],
        sha256=hashlib.sha256(content).hexdigest(),
    )


def robust_noise(values: np.ndarray) -> float:
    """Estimate white noise from second differences.

    For evenly sampled data the second difference of pure noise has six times
    its variance. The atlas is oversampled about 1.6 times per resolution
    element, so neighbouring samples are correlated and this is a lower bound.
    """

    second = np.diff(np.asarray(values, dtype=float), n=2)
    second = second[np.isfinite(second)]
    if second.size < 8:
        raise ValueError("too few samples to estimate the noise")
    estimate = float(1.4826 * np.median(np.abs(second - np.median(second))) / np.sqrt(6.0))
    if not np.isfinite(estimate) or estimate <= 0.0:
        # A noiseless or heavily quantized spectrum has no scatter to measure.
        raise ValueError("the spectrum has no measurable noise; supply an uncertainty")
    return estimate


def arcturus_spectral_order(
    page: ArcturusPage,
    *,
    column: str = "observed",
    source_flux_model_grid: np.ndarray | None = None,
    uncertainty: float | np.ndarray | None = None,
    saturation_floor: float = 0.02,
    telluric_ceiling: float = 1.05,
    zenith_angle_deg: float = 0.0,
) -> SpectralOrder:
    """Convert one of a page's columns into a fittable order.

    ``column`` selects what is fitted. ``"observed"`` is the raw spectrum and
    the only physically complete target. ``"telluric"`` is the transmission the
    atlas authors divided by, useful as a star-free consistency target, but it
    is a scaled transmission from a *different* observation, so agreement with
    it is a floor rather than a truth test. ``"ratioed"`` must not be fitted:
    it was made by dividing the observed column by the telluric one and then
    smoothing, which destroys the band limit and, in the authors' own words,
    "result[s] from mathematics and need not convey physical information".

    The atlas ascends in wavenumber and :class:`SpectralOrder` requires
    ascending vacuum wavelength, so every column is reversed here. Getting that
    wrong mirrors the spectrum and still looks plausible.

    ``mask`` is ``True`` where a pixel is usable, the opposite of the sibling
    project's convention. Pixels are dropped where the observed flux is not
    positive and finite, and where the atlas telluric column is either below
    ``saturation_floor`` -- a line core carrying no information -- or above
    ``telluric_ceiling``, which marks where the authors' own division blew up.
    That upper cut matters: the column reaches 2e5 on this page.

    ``uncertainty`` defaults to the page's robust noise estimate. The atlas
    ships no error array.
    """

    if not 0.0 < saturation_floor < telluric_ceiling:
        raise ValueError("require 0 < saturation_floor < telluric_ceiling")
    if column not in ("observed", "telluric"):
        raise ValueError("column must be 'observed' or 'telluric'; see the docstring on 'ratioed'")
    order = np.argsort(1.0e7 / page.wavenumber_vacuum_cm1)
    wavelength_nm = (1.0e7 / page.wavenumber_vacuum_cm1)[order]
    source = getattr(page, column)
    observed = source[order]
    telluric = page.telluric[order]

    if uncertainty is None:
        sigma = np.full(observed.size, robust_noise(source))
    else:
        sigma = np.broadcast_to(np.asarray(uncertainty, dtype=float), observed.shape).copy()
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("uncertainties must be finite and positive")

    mask = (
        np.isfinite(observed)
        & (observed > 0.0)
        & np.isfinite(telluric)
        & (telluric >= saturation_floor)
        & (telluric <= telluric_ceiling)
    )
    if not np.any(mask):
        raise ValueError("every pixel was masked")
    # SpectralOrder validates the flux array itself, so masked-out pixels still
    # need a finite positive placeholder.
    flux = np.where(mask, observed, 1.0)

    return SpectralOrder(
        wavelength_vacuum_nm=wavelength_nm,
        flux=flux,
        uncertainty=sigma,
        mask=mask,
        zenith_angle_deg=zenith_angle_deg,
        source_flux_model_grid=source_flux_model_grid,
    )
