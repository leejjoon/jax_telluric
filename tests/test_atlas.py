import numpy as np
import pytest

from jax_telluric import (
    arcturus_spectral_order,
    epoch_velocity_kms,
    read_arcturus_page,
)


FIXTURE = "tests/data/arcturus_page_sample.txt"


def test_reads_ascending_wavenumbers_and_the_requested_epoch():
    summer = read_arcturus_page(FIXTURE, "summer")
    winter = read_arcturus_page(FIXTURE, "winter")

    assert summer.wavenumber_vacuum_cm1[0] == 5000.0
    assert np.all(np.diff(summer.wavenumber_vacuum_cm1) > 0.0)
    np.testing.assert_allclose(summer.spacing_cm1, 0.02, rtol=1e-9)
    # The epochs must not be interchangeable, or a column mix-up passes silently.
    assert summer.observed[0] == pytest.approx(0.90, abs=0.02)
    assert winter.observed[0] == pytest.approx(0.80, abs=0.02)
    assert summer.telluric[0] == pytest.approx(0.95)
    assert winter.telluric[0] == pytest.approx(0.97)
    assert summer.ratioed[0] == pytest.approx(0.94, abs=0.02)
    assert winter.ratioed[0] == pytest.approx(0.82, abs=0.02)
    assert summer.sha256 == winter.sha256


def test_unknown_epoch_is_rejected():
    with pytest.raises(ValueError, match="epoch"):
        read_arcturus_page(FIXTURE, "autumn")


def test_order_is_reversed_into_ascending_wavelength():
    page = read_arcturus_page(FIXTURE, "summer")
    order = arcturus_spectral_order(page)

    assert np.all(np.diff(order.wavelength_vacuum_nm) > 0.0)
    # Ascending wavelength is descending wavenumber, so the columns reverse.
    np.testing.assert_allclose(
        order.wavelength_vacuum_nm, (1.0e7 / page.wavenumber_vacuum_cm1)[::-1]
    )
    np.testing.assert_allclose(order.flux[order.mask], page.observed[::-1][order.mask])


def test_mask_marks_usable_pixels_and_cuts_both_telluric_tails():
    page = read_arcturus_page(FIXTURE, "summer")
    order = arcturus_spectral_order(page)

    reversed_observed = page.observed[::-1]
    reversed_telluric = page.telluric[::-1]
    # True means "use this pixel", the opposite of the sibling project.
    assert order.mask.sum() == order.mask.size - 3
    assert not order.mask[reversed_observed < 0.0][0]
    assert not order.mask[reversed_telluric > 1.05][0]
    assert not order.mask[reversed_telluric < 0.02][0]
    assert np.all(np.isfinite(order.flux)) and np.all(order.flux > 0.0)


def test_uncertainty_defaults_to_the_robust_noise_estimate():
    page = read_arcturus_page(FIXTURE, "summer")
    order = arcturus_spectral_order(page)
    assert np.all(order.uncertainty > 0.0)
    assert len(set(order.uncertainty.tolist())) == 1

    explicit = arcturus_spectral_order(page, uncertainty=0.004)
    np.testing.assert_allclose(explicit.uncertainty, 0.004)


def test_select_trims_to_a_range_without_crossing_a_page_seam():
    page = read_arcturus_page(FIXTURE, "summer")
    trimmed = page.select(5000.10, 5000.30)

    assert trimmed.wavenumber_vacuum_cm1[0] >= 5000.10
    assert trimmed.wavenumber_vacuum_cm1[-1] <= 5000.30
    assert trimmed.observed.size == trimmed.wavenumber_vacuum_cm1.size
    with pytest.raises(ValueError, match="too few samples"):
        page.select(5000.00, 5000.05)


def test_epoch_velocities_match_the_measured_sign_and_separation():
    """The sign is set by matching the atlas's own atomic line identifications.

    Four lines in 5015-5020 cm-1 give +13.5 +- 0.5 km/s for summer, and an
    independent two-epoch cross-correlation gives +13.57, so summer is
    receding. Reading the readme.dat factor as a division gives the opposite.
    """
    summer = epoch_velocity_kms("summer")
    winter = epoch_velocity_kms("winter")
    assert summer == pytest.approx(14.33, abs=0.05)
    assert winter == pytest.approx(-26.11, abs=0.05)
    assert summer - winter == pytest.approx(40.44, abs=0.1)
    assert summer == pytest.approx(13.5, abs=1.0)


def test_column_selection_switches_the_fitted_target():
    page = read_arcturus_page(FIXTURE, "summer")
    observed = arcturus_spectral_order(page)
    # The fixture's telluric column is a clean ramp, so its noise estimator
    # correctly refuses it; a real page has scatter.
    telluric = arcturus_spectral_order(page, column="telluric", uncertainty=0.004)

    np.testing.assert_allclose(observed.flux[observed.mask], page.observed[::-1][observed.mask])
    np.testing.assert_allclose(telluric.flux[telluric.mask], page.telluric[::-1][telluric.mask])
    # The telluric floor and ceiling apply either way, but the positivity cut
    # follows whichever column is being fitted: the fixture's negative pixel is
    # negative in 'observed' and perfectly fine in 'telluric'.
    negative = page.observed[::-1] < 0.0
    assert not observed.mask[negative][0]
    assert telluric.mask[negative][0]
    both_bad = ~observed.mask & ~telluric.mask
    assert both_bad.sum() == 2  # the planted telluric ceiling and floor pixels


def test_ratioed_column_is_refused():
    page = read_arcturus_page(FIXTURE, "summer")
    with pytest.raises(ValueError, match="ratioed"):
        arcturus_spectral_order(page, column="ratioed")
