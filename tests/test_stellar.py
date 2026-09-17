import numpy as np
import pytest

from jax_telluric import (
    ArrayOpacityBackend,
    AtmosphereProfile,
    StellarSpectrum,
    TelluricModel,
    broaden_stellar_source,
    prepare_stellar_source,
    resample_stellar_source,
)


def model_on(nu):
    profile = AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 1.0],
        temperature_k=[240.0, 280.0],
        altitude_km=[10.0, 2.0],
        vmr={"H2O": [2.0e-4, 5.0e-3]},
    )
    return TelluricModel(profile, nu, ArrayOpacityBackend({"H2O": np.zeros((2, nu.size))}))


def test_rejects_descending_or_nonpositive_input():
    nu = np.geomspace(4300.0, 4310.0, 32)
    with pytest.raises(ValueError, match="strictly increasing"):
        StellarSpectrum(nu[::-1], np.ones(32))
    with pytest.raises(ValueError, match="finite and positive"):
        StellarSpectrum(nu, np.concatenate([[0.0], np.ones(31)]))


def test_from_wavelength_nm_sorts_into_ascending_wavenumber():
    wavelength = np.linspace(2000.0, 2010.0, 32)
    flux = np.linspace(1.0, 2.0, 32)
    spectrum = StellarSpectrum.from_wavelength_nm(wavelength, flux)
    assert np.all(np.diff(spectrum.wavenumber_cm1) > 0.0)
    # Ascending wavelength is descending wavenumber, so the flux reverses.
    np.testing.assert_allclose(spectrum.flux, flux[::-1])


def test_resampling_requires_full_coverage_of_the_model_grid():
    grid = np.geomspace(4300.0, 4310.0, 256)
    narrow = StellarSpectrum(np.geomspace(4302.0, 4308.0, 256), np.ones(256))
    with pytest.raises(ValueError, match="needs"):
        resample_stellar_source(narrow, grid)


def test_resampling_refuses_a_source_coarser_than_the_model_grid():
    grid = np.geomspace(4300.0, 4310.0, 4096)
    coarse = StellarSpectrum(np.geomspace(4299.0, 4311.0, 64), np.ones(64))
    with pytest.raises(ValueError, match="coarser"):
        resample_stellar_source(coarse, grid)


def test_broadening_is_the_identity_at_zero_width():
    values = 1.0 - 0.3 * np.exp(-0.5 * ((np.arange(512) - 256) / 4.0) ** 2)
    np.testing.assert_allclose(
        broaden_stellar_source(values, 1.0, vsini_kms=0.0, macroturbulence_kms=0.0), values
    )


@pytest.mark.parametrize("vsini,zeta", [(10.0, 0.0), (0.0, 8.0), (6.0, 5.0)])
def test_broadening_conserves_equivalent_width_and_widens_the_line(vsini, zeta):
    step = 0.5
    velocity = (np.arange(2048) - 1024) * step
    values = 1.0 - 0.3 * np.exp(-0.5 * (velocity / 3.0) ** 2)
    widened = broaden_stellar_source(values, step, vsini_kms=vsini, macroturbulence_kms=zeta)

    absorbed = np.sum(1.0 - values) * step
    widened_absorbed = np.sum(1.0 - widened) * step
    np.testing.assert_allclose(widened_absorbed, absorbed, rtol=1e-6)

    def second_moment(profile):
        weight = 1.0 - profile
        return np.sum(weight * velocity**2) / np.sum(weight)

    assert second_moment(widened) > second_moment(values)


def test_prepare_normalizes_and_matches_the_model_grid():
    nu = np.geomspace(4300.0, 4310.0, 1024)
    model = model_on(nu)
    spectrum = StellarSpectrum(np.geomspace(4299.0, 4311.0, 4096), np.full(4096, 7.5))
    prepared = prepare_stellar_source(spectrum, model, vsini_kms=2.0, macroturbulence_kms=2.15)
    assert prepared.shape == nu.shape
    np.testing.assert_allclose(np.median(prepared), 1.0, rtol=1e-12)

    unnormalized = prepare_stellar_source(spectrum, model, normalize=False)
    np.testing.assert_allclose(np.median(unnormalized), 7.5, rtol=1e-12)


def test_flat_source_is_a_unit_spectrum():
    nu = np.geomspace(4300.0, 4310.0, 64)
    np.testing.assert_allclose(StellarSpectrum.flat(nu).flux, 1.0)
