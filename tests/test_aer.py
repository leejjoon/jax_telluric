import os
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jax_telluric import (
    AERLineDatabase,
    AtmosphereProfile,
    line_optical_depth_bound,
    select_significant_lines,
)
from jax_telluric.direct import SparseCoreDirect


def _line(nu, isotope=1):
    gamma_air = f"{0.07:.4f}"[1:]
    gamma_self = f"{0.08:.4f}"[1:]
    return (
        f"{5:2d}{isotope:1d}{nu:12.6f}{1.0e-22:10.3E}{2.0e-5:10.3E}"
        f"{gamma_air}{gamma_self}{100.0:10.4f}{0.7:4.2f}{-0.001:8.6f}"
    )


@pytest.mark.parametrize("header", ["> header\n%%%%%%%%\n", ""])
def test_aer_database_reads_only_selected_lines(tmp_path, header):
    path = tmp_path / "05_CO"
    path.write_text(header + _line(4299.0) + "\n" + _line(4301.0) + "\n")
    database = AERLineDatabase(path, "CO", (4300.5, 4301.5), margin_cm1=0.0)
    assert database.simple_molecule_name == "CO"
    assert database.nu_lines.tolist() == [4301.0]
    assert database.gamma_self.tolist() == [0.08]
    assert database.qr_interp(0, 296.0, 296.0).tolist() == [1.0]


def _record(nu, strength, elower=200.0, isotope=1):
    gamma_air = f"{0.07:.4f}"[1:]
    gamma_self = f"{0.35:.4f}"[1:]
    return (
        f"{5:2d}{isotope:1d}{nu:12.6f}{strength:10.3E}{2.0e-5:10.3E}"
        f"{gamma_air}{gamma_self}{elower:10.4f}{0.7:4.2f}{0.0:8.6f}"
    )


def _synthetic_database(tmp_path, count=400, seed=5):
    """A real database spanning ten orders of magnitude in line strength."""
    rng = np.random.default_rng(seed)
    nu = np.sort(rng.uniform(4995.0, 5005.0, count))
    strength = 10.0 ** rng.uniform(-30.0, -20.0, count)
    elower = rng.uniform(0.0, 1500.0, count)
    path = tmp_path / "05_CO"
    path.write_text("\n".join(_record(a, b, c) for a, b, c in zip(nu, strength, elower)) + "\n")
    return AERLineDatabase(path, "CO", (4990.0, 5010.0), margin_cm1=0.0)


def _profile():
    return AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 0.9],
        temperature_k=[240.0, 285.0], altitude_km=[10.0, 2.0],
        vmr={"CO": [3.0e-4, 6.0e-3]},
    )


def test_line_bound_is_never_below_the_optical_depth_a_line_reaches(tmp_path):
    """The discard budget is a guarantee only if the bound really bounds."""
    profile = _profile()
    database = _synthetic_database(tmp_path, count=40)
    bound = line_optical_depth_bound(database, profile, "CO")
    grid = np.linspace(4990.0, 5010.0, 6001)
    column = np.asarray(profile.air_column_cm2)
    for index in (0, 7, 19, 33):
        keep = np.arange(bound.size) == index
        calculator = SparseCoreDirect(database.restrict(keep), grid,
                                      minimum_temperature_k=200.0, maximum_temperature_k=320.0)
        optical_depth = np.zeros(grid.size)
        for layer, (temperature, pressure) in enumerate(
            zip(profile.temperature_k, profile.pressure_layer_bar)
        ):
            vmr = profile.vmr["CO"][layer]
            optical_depth += np.asarray(
                calculator.xsvector(temperature, pressure, pressure * vmr)
            ) * column[layer] * vmr
        assert optical_depth.max() <= bound[index] * (1.0 + 1.0e-9), (
            f"line {index} reached {optical_depth.max():.3e} against bound {bound[index]:.3e}"
        )


def test_discarded_lines_stay_inside_the_budget(tmp_path):
    profile = _profile()
    database = _synthetic_database(tmp_path)
    bound = line_optical_depth_bound(database, profile, "CO")
    budget = 0.05 * bound.sum()
    kept = select_significant_lines(database, profile, "CO", budget)
    assert kept.nu_lines.size < database.nu_lines.size
    keep = np.isin(np.asarray(database.nu_lines), np.asarray(kept.nu_lines))
    assert bound[~keep].sum() <= budget
    # Nothing stronger may be dropped than the weakest line that survives.
    assert bound[~keep].max() <= bound[keep].min()


def test_zero_budget_keeps_every_line(tmp_path):
    profile = _profile()
    database = _synthetic_database(tmp_path)
    assert select_significant_lines(database, profile, "CO", 0.0) is database


def test_selection_never_empties_a_species(tmp_path):
    profile = _profile()
    database = _synthetic_database(tmp_path)
    assert select_significant_lines(database, profile, "CO", 1.0e30).nu_lines.size == 1


def test_restrict_filters_every_per_line_array(tmp_path):
    database = _synthetic_database(tmp_path, count=10)
    keep = np.zeros(10, dtype=bool)
    keep[[1, 4, 7]] = True
    restricted = database.restrict(keep)
    for name in AERLineDatabase._LINE_ARRAYS:
        assert np.asarray(getattr(restricted, name)).shape[0] == 3, name
    np.testing.assert_allclose(np.asarray(restricted.nu_lines),
                               np.asarray(database.nu_lines)[keep])
    with pytest.raises(ValueError, match="one entry per line"):
        database.restrict(np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="keeps no lines"):
        database.restrict(np.zeros(10, dtype=bool))


def test_maximum_column_scale_only_raises_the_bound(tmp_path):
    profile = _profile()
    database = _synthetic_database(tmp_path, count=50)
    base = line_optical_depth_bound(database, profile, "CO")
    scaled = line_optical_depth_bound(database, profile, "CO", maximum_column_scale=4.0)
    assert np.all(scaled >= base)
    with pytest.raises(ValueError):
        line_optical_depth_bound(database, profile, "CO", maximum_column_scale=0.0)
    with pytest.raises(ValueError, match="no XX profile"):
        line_optical_depth_bound(database, profile, "XX")


def test_reader_agrees_with_a_plain_line_by_line_scan(tmp_path):
    """The fast index may only narrow the work, never make a decision."""
    from jax_telluric.aer import _MOLECULE_IDS, _fortran_float

    rng = np.random.default_rng(11)
    nu = np.sort(rng.uniform(4990.0, 5010.0, 500))
    body = [_record(value, 1.0e-22) for value in nu]
    # A header, a delimiter and a short line, as the real files carry.
    text = "> 05_CO    : copyright notice\n%%%%%%%%\nshort\n" + "\n".join(body) + "\n"
    path = tmp_path / "05_CO"
    path.write_text(text)

    def scan(lower, upper, margin):
        kept = []
        for line in text.splitlines(keepends=True):
            if len(line) < 67 or not line[:2].strip():
                continue
            try:
                identifier = int(line[0:2])
                value = _fortran_float(line[3:15])
            except ValueError:
                continue
            if identifier != _MOLECULE_IDS["CO"] or not (lower - margin <= value <= upper + margin):
                continue
            kept.append(value)
        return np.asarray(kept)

    for lower, upper, margin in ((4995.0, 5005.0, 0.0), (4995.0, 5005.0, 25.0),
                                 (4999.9, 5000.1, 0.0), (4980.0, 4985.0, 1.0)):
        expected = scan(lower, upper, margin)
        if expected.size == 0:
            with pytest.raises(ValueError, match="no CO lines"):
                AERLineDatabase(path, "CO", (lower, upper), margin_cm1=margin)
            continue
        actual = AERLineDatabase(path, "CO", (lower, upper), margin_cm1=margin).nu_lines
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_reader_falls_back_for_a_file_the_index_cannot_handle(tmp_path):
    """Carriage returns change line lengths, so the byte offsets would be wrong."""
    from jax_telluric.aer import _build_file_index

    body = "\r\n".join(_record(nu, 1.0e-22) for nu in (4999.0, 5000.0, 5001.0)) + "\r\n"
    path = tmp_path / "05_CO"
    path.write_bytes(body.encode("ascii"))
    assert _build_file_index(path) is None
    database = AERLineDatabase(path, "CO", (4999.5, 5000.5), margin_cm1=0.0)
    np.testing.assert_allclose(np.asarray(database.nu_lines), [5000.0])


def test_the_file_index_notices_a_rewritten_file(tmp_path):
    path = tmp_path / "05_CO"
    path.write_text(_record(5000.0, 1.0e-22) + "\n")
    first = AERLineDatabase(path, "CO", (4990.0, 5010.0), margin_cm1=0.0)
    np.testing.assert_allclose(np.asarray(first.nu_lines), [5000.0])
    path.write_text(_record(5000.0, 1.0e-22) + "\n" + _record(5002.0, 2.0e-22) + "\n")
    os.utime(path, (0, 0))  # a changed size is enough, but do not rely on it
    second = AERLineDatabase(path, "CO", (4990.0, 5010.0), margin_cm1=0.0)
    np.testing.assert_allclose(np.asarray(second.nu_lines), [5000.0, 5002.0])


@pytest.mark.parametrize("identifier", [" 5", "05"])
def test_both_padded_forms_of_the_molecule_field_are_read(identifier, tmp_path):
    """AER right-justifies the id, so ' 5' is the common form, not '05'."""
    record = _record(5000.0, 1.0e-22)
    path = tmp_path / "05_CO"
    path.write_text(identifier + record[2:] + "\n")
    database = AERLineDatabase(path, "CO", (4999.0, 5001.0), margin_cm1=0.0)
    np.testing.assert_allclose(np.asarray(database.nu_lines), [5000.0])
