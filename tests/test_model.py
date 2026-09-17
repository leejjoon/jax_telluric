import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_telluric import (
    ArrayOpacityBackend,
    BoxcarFTSInstrumentProfile,
    AtmosphereProfile,
    SpectralOrder,
    ReferenceWaterContinuum,
    LBLRTMOpticalDepthCorrection,
    LinearizedOpacityBackend,
    MTCKDWaterContinuum,
    OrderObjective,
    TelluricModel,
    TelluricParameters,
    fit_order,
    igrins_wavenumber_grid,
    trim_wavenumber_grid,
)


def make_model():
    nu = np.geomspace(4300.0, 4310.0, 512)
    profile = AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 1.0],
        temperature_k=[240.0, 280.0],
        altitude_km=[10.0, 2.0],
        vmr={"H2O": [2.0e-4, 5.0e-3]},
    )
    line = np.exp(-0.5 * ((nu - 4305.0) / 0.035) ** 2) * 2.0e-23
    opacity = ArrayOpacityBackend({"H2O": np.vstack([line, 1.1 * line])})
    return TelluricModel(profile, nu, opacity), nu


def test_igrins_grid_is_padded_and_oversampled():
    grid = igrins_wavenumber_grid(2200.0, 2220.0)
    assert grid[0] <= 1.0e7 / 2220.0 - 25.0
    assert grid[-1] >= 1.0e7 / 2200.0 + 25.0
    velocity_step = np.diff(np.log(grid)) * 299792.458
    np.testing.assert_allclose(velocity_step, velocity_step[0], rtol=1e-9)
    assert velocity_step[0] <= 299792.458 / 45000.0 / 4.0


def params(scale=0.0, sigma=2.8):
    return TelluricParameters(
        log_column_scales={"H2O": scale},
        velocity_kms=0.0,
        wavelength_stretch=0.0,
        lsf_sigma_kms=sigma,
        continuum_coeffs=jnp.array([0.0, 0.0, 0.0]),
        log_jitter=np.log(1.0e-4),
    )


def test_transmission_matches_column_sum_and_airmass():
    model, _ = make_model()
    actual = np.asarray(model.transmission(params(), zenith_angle_deg=60.0))
    xs = model.opacity.values["H2O"]
    absorber_column = model.profile.air_column_cm2 * model.profile.vmr["H2O"]
    expected = np.exp(-np.sum(xs * absorber_column[:, None], axis=0) / 0.5)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_column_scale_gradient_is_finite_and_has_expected_sign():
    model, _ = make_model()

    def line_flux(log_scale):
        return jnp.min(model.transmission(params(log_scale)))

    derivative = jax.grad(line_flux)(jnp.asarray(0.0))
    assert jnp.isfinite(derivative)
    assert derivative < 0.0


def test_model_gradient_matches_central_difference():
    model, nu = make_model()
    wavelength = np.linspace(1.0e7 / nu[-30], 1.0e7 / nu[30], 70)
    order = SpectralOrder(wavelength, np.ones(70), np.full(70, 0.01))

    def statistic(vector):
        parameters = TelluricParameters(
            {"H2O": vector[0]},
            vector[1],
            vector[2],
            vector[3],
            vector[4:6],
            np.log(1.0e-4),
        )
        weights = jnp.linspace(0.5, 1.5, wavelength.size)
        return jnp.sum(weights * model.predict(order, parameters))

    point = np.asarray([0.15, 0.31, 1.7e-5, 3.1, 0.02, -0.01])
    steps = np.asarray([1e-5, 1e-4, 1e-8, 1e-4, 1e-5, 1e-5])
    automatic = np.asarray(jax.grad(statistic)(jnp.asarray(point)))
    finite = np.empty_like(point)
    for index, step in enumerate(steps):
        delta = np.zeros_like(point)
        delta[index] = step
        finite[index] = (float(statistic(point + delta)) - float(statistic(point - delta))) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite, rtol=1.0e-4, atol=2.0e-7)


def test_water_continuum_uses_linear_foreign_and_quadratic_self_scaling():
    model, nu = make_model()
    shape = (2, len(nu))
    continuum = ReferenceWaterContinuum(
        reference_vmr=model.profile.vmr["H2O"],
        self_optical_depth=np.full(shape, 0.01),
        foreign_optical_depth=np.full(shape, 0.02),
    )
    continuum_model = TelluricModel(
        model.profile,
        np.asarray(model.wavenumber_cm1),
        ArrayOpacityBackend({"H2O": np.zeros(shape)}),
        continuum=continuum,
    )
    actual_tau = -np.log(np.asarray(continuum_model.transmission(params(np.log(2.0)))))
    np.testing.assert_allclose(actual_tau, 2 * (0.01 * 4.0 + 0.02 * 2.0))


def test_runtime_mt_ckd_matches_reference_formula_and_is_differentiable():
    model, nu = make_model()
    coefficient_nu = np.arange(4250.0, 4370.0, 10.0)
    self_ref = 2.0e-28 * (1.0 + 1.0e-4 * (coefficient_nu - 4300.0))
    foreign_ref = 3.0e-28 * (1.0 + 2.0e-4 * (coefficient_nu - 4300.0))
    exponent = np.full_like(coefficient_nu, 4.2)
    continuum = MTCKDWaterContinuum(
        nu, coefficient_nu, self_ref, foreign_ref, exponent
    )
    continuum_model = TelluricModel(
        model.profile,
        nu,
        ArrayOpacityBackend({"H2O": np.zeros((2, len(nu)))}),
        continuum=continuum,
        accuracy_mode="mt_ckd",
    )

    actual_tau = -np.log(np.asarray(continuum_model.transmission(params())))
    pressure_hpa = 0.5 * (
        model.profile.pressure_edges_bar[:-1] + model.profile.pressure_edges_bar[1:]
    ) * 1000.0
    temperature = model.profile.temperature_k
    water = model.profile.vmr["H2O"]
    radiation = nu[None, :] * np.tanh(
        0.5 * nu[None, :] * 1.4387752 / temperature[:, None]
    )
    density = pressure_hpa / 1013.0 * 296.0 / temperature
    expected_self_ref = np.interp(nu, coefficient_nu, self_ref)
    expected_foreign_ref = np.interp(nu, coefficient_nu, foreign_ref)
    cross_section = (
        expected_self_ref[None, :]
        * (296.0 / temperature[:, None]) ** 4.2
        * water[:, None]
        + expected_foreign_ref[None, :] * (1.0 - water[:, None])
    ) * density[:, None] * radiation
    expected_tau = np.sum(
        cross_section * (model.profile.air_column_cm2 * water)[:, None], axis=0
    )
    np.testing.assert_allclose(actual_tau, expected_tau, rtol=2.0e-12, atol=2.0e-12)

    derivative = jax.grad(
        lambda scale: jnp.sum(continuum_model.transmission(params(scale)))
    )(jnp.asarray(0.0))
    assert jnp.isfinite(derivative)
    assert derivative < 0.0


def test_lblrtm_corrected_mode_scales_continuum_and_line_residuals(tmp_path):
    model, nu = make_model()
    layers, samples = 2, len(nu)
    correction = LBLRTMOpticalDepthCorrection(
        wavenumber_cm1=nu,
        pressure_layer_bar=model.profile.pressure_layer_bar,
        temperature_k=model.profile.temperature_k,
        air_column_cm2=model.profile.air_column_cm2,
        reference_vmr=model.profile.vmr,
        water_self_optical_depth=np.full(samples, 0.01),
        water_foreign_optical_depth=np.full(samples, 0.02),
        reference_background_optical_depth=np.zeros(samples),
        line_residual_optical_depth={"H2O": np.full(samples, -0.003)},
    )
    path = tmp_path / "correction.npz"
    correction.save(path)
    correction = LBLRTMOpticalDepthCorrection.load(path)
    corrected = TelluricModel(
        model.profile, nu, model.opacity,
        accuracy_mode="lblrtm_corrected", correction=correction,
    )
    scale = 2.0
    parameters = params(np.log(scale))
    fast_tau = -np.log(np.asarray(model.transmission(parameters)))
    corrected_tau = -np.log(np.asarray(corrected.transmission(parameters)))
    expected_extra = 0.01 * scale**2 + 0.02 * scale - 0.003 * scale
    np.testing.assert_allclose(corrected_tau - fast_tau, expected_extra, rtol=2e-12, atol=2e-12)
    derivative = jax.grad(
        lambda log_scale: jnp.sum(corrected.transmission(params(log_scale)))
    )(jnp.asarray(np.log(scale)))
    assert jnp.isfinite(derivative)

    physics = TelluricModel(
        model.profile, nu, model.opacity,
        accuracy_mode="mt_ckd", correction=correction,
    )
    physics_tau = -np.log(np.asarray(physics.transmission(parameters)))
    expected_mt_ckd = 0.01 * scale**2 + 0.02 * scale
    np.testing.assert_allclose(
        physics_tau - fast_tau, expected_mt_ckd, rtol=2e-12, atol=2e-12
    )
    assert physics.species == model.opacity.species


def test_lblrtm_corrected_mode_requires_matching_correction():
    model, nu = make_model()
    with np.testing.assert_raises(ValueError):
        TelluricModel(model.profile, nu, model.opacity, accuracy_mode="lblrtm_corrected")
    with np.testing.assert_raises(ValueError):
        TelluricModel(model.profile, nu, model.opacity, accuracy_mode="mt_ckd")
    with np.testing.assert_raises(ValueError):
        TelluricModel(model.profile, nu, model.opacity, accuracy_mode="unknown")


def test_pressure_shifted_correction_rejects_unshifted_backend(tmp_path):
    model, nu = make_model()
    samples = len(nu)
    correction = LBLRTMOpticalDepthCorrection(
        wavenumber_cm1=nu,
        pressure_layer_bar=model.profile.pressure_layer_bar,
        temperature_k=model.profile.temperature_k,
        air_column_cm2=model.profile.air_column_cm2,
        reference_vmr=model.profile.vmr,
        water_self_optical_depth=np.zeros(samples),
        water_foreign_optical_depth=np.zeros(samples),
        reference_background_optical_depth=np.zeros(samples),
        line_residual_optical_depth={"H2O": np.zeros(samples)},
        requires_pressure_shift=True,
    )
    path = tmp_path / "shifted.npz"
    correction.save(path)
    loaded = LBLRTMOpticalDepthCorrection.load(path)
    assert loaded.requires_pressure_shift
    with np.testing.assert_raises_regex(ValueError, "pressure-shifted opacity backend"):
        TelluricModel(
            model.profile, nu, model.opacity,
            accuracy_mode="lblrtm_corrected", correction=loaded,
        )



def test_prediction_applies_lsf_sampling_and_positive_continuum():
    model, nu = make_model()
    wavelength = np.linspace(1.0e7 / nu[-20], 1.0e7 / nu[20], 80)
    order = SpectralOrder(wavelength, np.ones(80), np.full(80, 0.01))
    predicted = np.asarray(model.predict(order, params()))
    assert predicted.shape == wavelength.shape
    assert np.all(np.isfinite(predicted))
    assert np.all(predicted > 0.0)
    assert np.min(predicted) < 1.0


def test_fit_recovers_injected_water_column():
    model, nu = make_model()
    wavelength = np.linspace(1.0e7 / nu[-20], 1.0e7 / nu[20], 100)
    blank = SpectralOrder(wavelength, np.ones(100), np.full(100, 0.002))
    truth = params(scale=0.3)
    flux = np.asarray(model.predict(blank, truth))
    order = SpectralOrder(wavelength, flux, np.full(100, 0.002))
    fixed = 1.0e-12
    bounds = {
        "H2O": (-1.0, 1.0),
        "velocity_kms": (-fixed, fixed),
        "wavelength_stretch": (-fixed, fixed),
        "lsf_sigma_kms": (2.8 - fixed, 2.8 + fixed),
        "continuum_0": (-fixed, fixed),
        "continuum_1": (-fixed, fixed),
        "continuum_2": (-fixed, fixed),
        "log_jitter": (np.log(1.0e-4) - fixed, np.log(1.0e-4) + fixed),
    }
    result = fit_order(model, order, params(scale=0.0), bounds)
    assert result.success
    assert abs(float(result.parameters.log_column_scales["H2O"]) - 0.3) < 2.0e-3


def test_fit_converges_with_small_uncertainties_and_exact_fixed_bounds():
    """A success result must not stop far from a clear, noiseless optimum."""
    model, nu = make_model()
    wavelength = np.linspace(1.0e7 / nu[-20], 1.0e7 / nu[20], 160)
    blank = SpectralOrder(wavelength, np.ones(160), np.full(160, 1.0e-3))
    truth = TelluricParameters(
        {"H2O": 0.2}, 0.4, 0.0, 3.0, jnp.array([0.01, 0.0, 0.0]), np.log(1.0e-5)
    )
    flux = np.asarray(model.predict(blank, truth))
    order = SpectralOrder(wavelength, flux, np.full(160, 1.0e-3))
    initial = params(scale=0.0)
    bounds = {
        "H2O": (-1.0, 1.0),
        "velocity_kms": (-3.0, 3.0),
        "wavelength_stretch": (0.0, 0.0),
        "lsf_sigma_kms": (1.0, 5.0),
        "continuum_0": (-0.1, 0.1),
        "continuum_1": (0.0, 0.0),
        "continuum_2": (0.0, 0.0),
        "log_jitter": (np.log(1.0e-5), np.log(1.0e-5)),
    }

    result = fit_order(model, order, initial, bounds)

    assert result.success, result.message
    recovered = np.asarray(
        [result.parameters.log_column_scales["H2O"], result.parameters.velocity_kms,
         result.parameters.lsf_sigma_kms, result.parameters.continuum_coeffs[0]]
    )
    error = np.abs(recovered - np.asarray([0.2, 0.4, 3.0, 0.01]))
    assert np.all(error <= np.asarray([1.0e-6, 1.0e-5, 1.0e-5, 1.0e-7])), (recovered, error)
    assert result.covariance.shape == (8, 8)
    np.testing.assert_array_equal(result.covariance[[2, 5, 6, 7]], 0.0)
    variance = order.uncertainty**2 + np.exp(2.0 * truth.log_jitter)
    expected_objective = 0.5 * np.sum(
        (order.flux - result.model_flux) ** 2 / variance + np.log(variance)
    )
    np.testing.assert_allclose(result.objective, expected_objective, rtol=1.0e-12)


def flat_model():
    """A model with no absorption, so only the source and instrument act."""
    nu = np.geomspace(4300.0, 4310.0, 512)
    profile = AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 1.0],
        temperature_k=[240.0, 280.0],
        altitude_km=[10.0, 2.0],
        vmr={"H2O": [2.0e-4, 5.0e-3]},
    )
    opacity = ArrayOpacityBackend({"H2O": np.zeros((2, len(nu)))})
    return TelluricModel(profile, nu, opacity), nu


def source_with_dip(nu, center_cm1, width_cm1=0.02, depth=0.5):
    return 1.0 - depth * np.exp(-0.5 * ((nu - center_cm1) / width_cm1) ** 2)


def test_shift_log_uniform_reproduces_the_analytic_sample_shift():
    model, nu = make_model()
    # Catmull-Rom is exact for a linear function, so the recovered offset is
    # the shift itself rather than an interpolation artefact.
    values = jnp.asarray(1.0 + 0.001 * np.arange(nu.size))
    velocity = 12.0
    shifted = np.asarray(model._shift_log_uniform(values, jnp.asarray(velocity)))
    expected_samples = np.log1p(velocity / 299792.458) * (299792.458 / model.velocity_step_kms)
    interior = slice(10, -10)
    expected = 1.0 + 0.001 * (np.arange(nu.size) + expected_samples)
    np.testing.assert_allclose(shifted[interior], expected[interior], rtol=0.0, atol=1e-10)


def test_model_grid_source_resolves_structure_the_pixel_grid_cannot():
    """A stellar line falling between two pixels still reaches them via the LSF.

    The pixel-grid hook has to sample the source at the pixels themselves, so
    such a line is lost before the convolution ever sees it.
    """
    nu = np.geomspace(4300.0, 4310.0, 4096)
    profile = AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 1.0],
        temperature_k=[240.0, 280.0],
        altitude_km=[10.0, 2.0],
        vmr={"H2O": [2.0e-4, 5.0e-3]},
    )
    model = TelluricModel(profile, nu, ArrayOpacityBackend({"H2O": np.zeros((2, nu.size))}))
    wavelength = np.linspace(1.0e7 / nu[-40], 1.0e7 / nu[40], 40)
    between = 0.5 * (wavelength[19] + wavelength[20])
    narrow = source_with_dip(nu, 1.0e7 / between, width_cm1=0.03, depth=0.5)

    on_model_grid = SpectralOrder(
        wavelength, np.ones(40), np.full(40, 1e-3), source_flux_model_grid=narrow
    )
    at_pixels = np.interp(1.0e7 / wavelength[::-1], nu, narrow)[::-1]
    on_pixel_grid = SpectralOrder(wavelength, np.ones(40), np.full(40, 1e-3), source_flux=at_pixels)

    fine = np.asarray(model.predict(on_model_grid, params(sigma=8.0)))
    blunt = np.asarray(model.predict(on_pixel_grid, params(sigma=8.0)))
    assert 1.0 - fine.min() > 5.0 * (1.0 - blunt.min())


def test_source_flux_and_model_grid_source_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        SpectralOrder(
            np.linspace(2000.0, 2001.0, 16),
            np.ones(16),
            np.full(16, 1e-3),
            source_flux=np.ones(16),
            source_flux_model_grid=np.ones(16),
        )


def test_model_grid_source_must_match_the_model_grid():
    model, nu = make_model()
    wavelength = np.linspace(1.0e7 / nu[-30], 1.0e7 / nu[30], 60)
    order = SpectralOrder(
        wavelength, np.ones(60), np.full(60, 1e-3), source_flux_model_grid=np.ones(nu.size - 1)
    )
    with pytest.raises(ValueError, match="must match the model wavenumber grid"):
        model.predict(order, params())


def test_stellar_velocity_moves_the_star_and_not_the_atmosphere():
    star, nu = flat_model()
    wavelength = np.linspace(1.0e7 / nu[-40], 1.0e7 / nu[40], 200)
    order = SpectralOrder(
        wavelength, np.ones(200), np.full(200, 1e-3),
        source_flux_model_grid=source_with_dip(nu, 4305.0),
    )
    velocity = 40.0
    at_rest = np.asarray(star.predict(order, params(sigma=0.5)))
    moved = np.asarray(star.predict(order, params(sigma=0.5)._replace(stellar_velocity_kms=velocity)))
    # The dip is in wavelength space, so a positive velocity moves it redward.
    shift_nm = wavelength[np.argmin(moved)] - wavelength[np.argmin(at_rest)]
    expected_nm = wavelength[np.argmin(at_rest)] * velocity / 299792.458
    assert abs(shift_nm - expected_nm) < 2.0 * (wavelength[1] - wavelength[0])

    # A flat source leaves the telluric spectrum untouched by the same parameter.
    telluric, tnu = make_model()
    flat_order = SpectralOrder(wavelength, np.ones(200), np.full(200, 1e-3),
                               source_flux_model_grid=np.ones(tnu.size))
    still = np.asarray(telluric.predict(flat_order, params(sigma=0.5)))
    also = np.asarray(telluric.predict(flat_order, params(sigma=0.5)._replace(stellar_velocity_kms=velocity)))
    np.testing.assert_allclose(still, also, rtol=0.0, atol=1e-12)


def test_stellar_velocity_gradient_matches_central_difference():
    star, nu = flat_model()
    wavelength = np.linspace(1.0e7 / nu[-40], 1.0e7 / nu[40], 80)
    order = SpectralOrder(
        wavelength, np.ones(80), np.full(80, 1e-3),
        source_flux_model_grid=source_with_dip(nu, 4305.0),
    )

    def statistic(velocity):
        weights = jnp.linspace(0.5, 1.5, wavelength.size)
        return jnp.sum(weights * star.predict(order, params(sigma=0.8)._replace(stellar_velocity_kms=velocity)))

    point, step = 3.0, 1e-3
    automatic = float(jax.grad(statistic)(jnp.asarray(point)))
    finite = (float(statistic(point + step)) - float(statistic(point - step))) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-9)


def test_stellar_velocity_gradient_is_continuous_across_sample_boundaries():
    """Linear interpolation would make this gradient a staircase."""
    star, nu = flat_model()
    wavelength = np.linspace(1.0e7 / nu[-40], 1.0e7 / nu[40], 80)
    order = SpectralOrder(
        wavelength, np.ones(80), np.full(80, 1e-3),
        source_flux_model_grid=source_with_dip(nu, 4305.0),
    )

    def statistic(velocity):
        weights = jnp.linspace(0.5, 1.5, wavelength.size)
        return jnp.sum(weights * star.predict(order, params(sigma=0.8)._replace(stellar_velocity_kms=velocity)))

    gradient = jax.jit(jax.grad(statistic))
    # Sweep across roughly three sample boundaries of the model grid.
    velocities = np.linspace(0.0, 3.0 * star.velocity_step_kms, 240)
    values = np.asarray([float(gradient(jnp.asarray(v))) for v in velocities])
    jumps = np.abs(np.diff(values))
    assert jumps.max() < 0.05 * (values.max() - values.min())


def test_point_sampling_returns_the_interpolated_value_itself():
    model, nu = make_model()
    wavelength = np.linspace(1.0e7 / nu[-30], 1.0e7 / nu[30], 60)
    order = SpectralOrder(wavelength, np.ones(60), np.full(60, 1e-3))
    point_model = TelluricModel(
        model.profile, np.asarray(model.wavenumber_cm1), model.opacity, pixel_integration="point"
    )
    parameters = params()

    wavelength_hi = 1.0e7 / np.asarray(model.wavenumber_cm1)
    raw_hi = np.asarray(model.transmission(parameters, order.zenith_angle_deg))
    convolved_hi = np.asarray(model._convolve_lsf(jnp.asarray(raw_hi), jnp.asarray(parameters.lsf_sigma_kms)))
    expected = np.interp(wavelength, wavelength_hi[::-1], convolved_hi[::-1])

    point = np.asarray(point_model.predict(order, parameters))
    np.testing.assert_allclose(point, expected, rtol=0.0, atol=1e-12)
    # Simpson averages the pixel edges in as well, so it must differ.
    simpson = np.asarray(model.predict(order, parameters))
    assert not np.allclose(simpson, point, atol=1e-10)


def test_pixel_integration_mode_is_validated():
    model, nu = make_model()
    with pytest.raises(ValueError, match="pixel_integration"):
        TelluricModel(model.profile, np.asarray(model.wavenumber_cm1), model.opacity,
                      pixel_integration="boxcar")


def test_boxcar_fts_profile_reports_its_measured_line_shape():
    profile = BoxcarFTSInstrumentProfile(mopd_cm=15.167, wavenumber_center_cm1=5012.0)
    np.testing.assert_allclose(profile.fwhm_cm1, 1.20671 / (2.0 * 15.167), rtol=1e-12)
    np.testing.assert_allclose(profile.resolving_power, 125990.0, rtol=1e-3)
    np.testing.assert_allclose(profile.first_zero_kms, 299792.458 / (2 * 15.167 * 5012.0), rtol=1e-12)


def test_boxcar_fts_profile_rings_where_a_gaussian_cannot():
    """An unapodized sinc undershoots beside a sharp edge; a Gaussian never does."""
    star, nu = flat_model()
    instrument = BoxcarFTSInstrumentProfile(
        mopd_cm=15.167, wavenumber_center_cm1=float(nu[nu.size // 2]), max_residual_sigma_kms=0.0
    )
    spectrum = jnp.asarray(np.where(np.arange(nu.size) < nu.size // 2, 1.0, 0.0))
    convolved = np.asarray(instrument.convolve(spectrum, params(), star.velocity_step_kms))
    interior = convolved[20:-20]
    assert interior.min() < -1e-3
    gaussian = np.asarray(star._convolve_lsf(spectrum, jnp.asarray(2.0)))
    assert gaussian[20:-20].min() >= -1e-12


def stellar_fit_setup():
    nu = np.geomspace(4300.0, 4310.0, 4096)
    profile = AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 1.0],
        temperature_k=[240.0, 280.0],
        altitude_km=[10.0, 2.0],
        vmr={"H2O": [2.0e-4, 5.0e-3]},
    )
    line = np.exp(-0.5 * ((nu - 4303.0) / 0.05) ** 2) * 2.0e-23
    model = TelluricModel(profile, nu, ArrayOpacityBackend({"H2O": np.vstack([line, 1.1 * line])}))
    source = np.ones_like(nu)
    for center in (4302.0, 4305.0, 4306.5, 4308.0):
        source = source * source_with_dip(nu, center, width_cm1=0.06, depth=0.4)
    wavelength = np.linspace(1.0e7 / nu[-60], 1.0e7 / nu[60], 300)
    return model, nu, wavelength, source


def test_fit_recovers_an_injected_stellar_velocity():
    model, nu, wavelength, source = stellar_fit_setup()
    truth = params(scale=0.1, sigma=3.0)._replace(stellar_velocity_kms=7.0)
    blank = SpectralOrder(
        wavelength, np.ones(300), np.full(300, 1e-3), source_flux_model_grid=source
    )
    flux = np.asarray(model.predict(blank, truth))
    order = SpectralOrder(
        wavelength, flux, np.full(300, 1e-3), source_flux_model_grid=source
    )
    fixed = 1.0e-12
    bounds = {
        "H2O": (0.1 - fixed, 0.1 + fixed),
        "velocity_kms": (-fixed, fixed),
        "stellar_velocity_kms": (-20.0, 20.0),
        "wavelength_stretch": (-fixed, fixed),
        "lsf_sigma_kms": (3.0 - fixed, 3.0 + fixed),
        "continuum_0": (-fixed, fixed),
        "continuum_1": (-fixed, fixed),
        "continuum_2": (-fixed, fixed),
        "log_jitter": (np.log(1.0e-4) - fixed, np.log(1.0e-4) + fixed),
    }
    initial = params(scale=0.1, sigma=3.0)._replace(stellar_velocity_kms=0.0)

    result = fit_order(model, order, initial, bounds)

    assert result.success, result.message
    assert abs(float(result.parameters.stellar_velocity_kms) - 7.0) < 0.05
    assert result.covariance.shape == (9, 9)


def test_fit_requires_stellar_velocity_bounds_for_a_model_grid_source():
    model, nu, wavelength, source = stellar_fit_setup()
    order = SpectralOrder(
        wavelength, np.ones(300), np.full(300, 1e-3), source_flux_model_grid=source
    )
    bounds = {
        "H2O": (-1.0, 1.0),
        "velocity_kms": (-3.0, 3.0),
        "wavelength_stretch": (0.0, 0.0),
        "lsf_sigma_kms": (1.0, 5.0),
        "continuum_0": (-0.1, 0.1),
        "continuum_1": (0.0, 0.0),
        "continuum_2": (0.0, 0.0),
        "log_jitter": (np.log(1.0e-5), np.log(1.0e-5)),
    }
    with pytest.raises(ValueError, match="stellar_velocity_kms"):
        fit_order(model, order, params(), bounds)


class SelfBroadenedBackend:
    """Toy opacity whose line width responds to the self-broadening pressure.

    ``ArrayOpacityBackend`` ignores partial pressure, so it cannot exercise the
    one route a fitted column scale takes into a real opacity kernel. This one
    does, while staying cheap enough for a unit test.
    """

    species = ("H2O",)

    def __init__(self, wavenumber_cm1, center=4305.0, strength=2.0e-23, sensitivity=60.0):
        self.wavenumber_cm1 = np.asarray(wavenumber_cm1, dtype=float)
        self.center = center
        self.strength = strength
        self.sensitivity = sensitivity

    def cross_sections(self, temperature_k, pressure_bar, partial_pressure_bar):
        fraction = jnp.asarray(partial_pressure_bar["H2O"]) / jnp.asarray(pressure_bar)
        width = 0.035 * (1.0 + self.sensitivity * fraction)
        offset = jnp.asarray(self.wavenumber_cm1)[None, :] - self.center
        return {"H2O": self.strength * jnp.exp(-0.5 * (offset / width[:, None]) ** 2)}


def make_self_broadened_model():
    nu = np.geomspace(4300.0, 4310.0, 512)
    profile = AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 1.0],
        temperature_k=[240.0, 280.0],
        altitude_km=[10.0, 2.0],
        vmr={"H2O": [2.0e-4, 5.0e-3]},
    )
    return TelluricModel(profile, nu, SelfBroadenedBackend(nu)), nu


def test_precomputed_opacity_is_exact_at_its_own_reference():
    model, _ = make_self_broadened_model()
    for mode in ("linear", "frozen"):
        precomputed = model.precompute_opacity(params(scale=0.25), self_broadening=mode)
        exact = np.asarray(model.transmission(params(scale=0.25)))
        actual = np.asarray(precomputed.transmission(params(scale=0.25)))
        np.testing.assert_allclose(actual, exact, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "sensitivity,tolerance,ratio", [(4.0, 5.0e-4, 0.05), (60.0, 5.0e-2, 0.25)]
)
def test_linear_self_broadening_beats_freezing_away_from_the_reference(
    sensitivity, tolerance, ratio
):
    """The first-order term is why a fit using this needs no outer iteration.

    ``gamma_self / gamma_air`` is about five for water, so a real line width
    grows roughly like ``1 + 4 * vmr``. The second case is a stress test
    fifteen times beyond that, where a first-order expansion is expected to
    degrade -- it still beats freezing fourfold, but it no longer hides under
    the noise, which is the boundary of the approximation's usefulness.
    """
    model, nu = make_self_broadened_model()
    model.opacity = SelfBroadenedBackend(nu, sensitivity=sensitivity)
    linear = model.precompute_opacity(self_broadening="linear")
    frozen = model.precompute_opacity(self_broadening="frozen")
    for scale in (-0.7, -0.3, 0.3, 0.7):
        exact = np.asarray(model.transmission(params(scale=scale)))
        linear_error = np.max(np.abs(np.asarray(linear.transmission(params(scale=scale))) - exact))
        frozen_error = np.max(np.abs(np.asarray(frozen.transmission(params(scale=scale))) - exact))
        assert linear_error < ratio * frozen_error
        assert linear_error < tolerance


def test_precomputed_opacity_keeps_the_column_scale_differentiable():
    model, _ = make_self_broadened_model()
    precomputed = model.precompute_opacity()

    def total(scale):
        return jnp.sum(precomputed.transmission(params(scale=scale)))

    step = 1.0e-5
    expected = (float(total(0.2 + step)) - float(total(0.2 - step))) / (2.0 * step)
    np.testing.assert_allclose(float(jax.grad(total)(0.2)), expected, rtol=1e-6)


def test_precomputed_opacity_leaves_the_rest_of_the_model_alone():
    model, nu = make_self_broadened_model()
    precomputed = model.precompute_opacity()
    assert precomputed.continuum is model.continuum
    assert precomputed.accuracy_mode == model.accuracy_mode
    assert precomputed.species == model.species
    assert precomputed.velocity_step_kms == model.velocity_step_kms
    assert model.opacity is not precomputed.opacity


@pytest.mark.parametrize(
    "kwargs", [{"self_broadening": "quadratic"}, {"relative_step": 0.0}, {"relative_step": 1.5}]
)
def test_precompute_opacity_validates_its_arguments(kwargs):
    model, _ = make_self_broadened_model()
    with pytest.raises(ValueError):
        model.precompute_opacity(**kwargs)


def test_linearized_backend_rejects_inconsistent_arrays():
    values = {"H2O": np.ones((2, 8))}
    with pytest.raises(ValueError):
        LinearizedOpacityBackend(values, {"H2O": np.ones((2, 8))}, {"H2O": np.ones(3)})
    with pytest.raises(ValueError):
        LinearizedOpacityBackend(values, {"CO2": np.ones((2, 8))}, {"H2O": np.ones(2)})
    with pytest.raises(ValueError):
        LinearizedOpacityBackend(values, {"H2O": np.full((2, 8), np.nan)}, {"H2O": np.ones(2)})


def test_linearized_backend_cannot_return_a_negative_cross_section():
    backend = LinearizedOpacityBackend(
        {"H2O": np.ones((1, 4))}, {"H2O": np.full((1, 4), -10.0)}, {"H2O": np.zeros(1)}
    )
    actual = backend.cross_sections(jnp.ones(1), jnp.ones(1), {"H2O": jnp.array([1.0])})
    assert np.all(np.asarray(actual["H2O"]) == 0.0)


def test_shared_objective_matches_a_fit_that_compiles_per_call():
    """One compilation must serve every stage of a staged fit."""
    model, nu = make_model()
    wavelength = np.linspace(1.0e7 / nu[-20], 1.0e7 / nu[20], 120)
    blank = SpectralOrder(wavelength, np.ones(120), np.full(120, 2.0e-3))
    flux = np.asarray(model.predict(blank, params(scale=0.3)))
    order = SpectralOrder(wavelength, flux, np.full(120, 2.0e-3))
    stages = [
        {"H2O": (0.0, 0.0), "velocity_kms": (-2.0, 2.0)},
        {"H2O": (-1.0, 1.0), "velocity_kms": (0.0, 0.0)},
    ]
    common = {
        "wavelength_stretch": (0.0, 0.0), "lsf_sigma_kms": (2.8, 2.8),
        "continuum_0": (-0.2, 0.2), "continuum_1": (0.0, 0.0), "continuum_2": (0.0, 0.0),
        "log_jitter": (np.log(1.0e-4), np.log(1.0e-4)),
    }
    shared = OrderObjective(model, order, 3)
    current_shared = current_plain = params(scale=0.0)
    for stage in stages:
        bounds = {**common, **stage}
        current_shared = fit_order(model, order, current_shared, bounds, objective=shared).parameters
        current_plain = fit_order(model, order, current_plain, bounds).parameters
    np.testing.assert_allclose(
        float(current_shared.log_column_scales["H2O"]),
        float(current_plain.log_column_scales["H2O"]), rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        float(current_shared.velocity_kms), float(current_plain.velocity_kms),
        rtol=1e-10, atol=1e-12,
    )


def test_shared_objective_rejects_a_different_model_or_order():
    model, nu = make_model()
    other, _ = make_model()
    wavelength = np.linspace(1.0e7 / nu[-20], 1.0e7 / nu[20], 60)
    order = SpectralOrder(wavelength, np.ones(60), np.full(60, 1.0e-3))
    bounds = {
        "H2O": (-1.0, 1.0), "velocity_kms": (0.0, 0.0), "wavelength_stretch": (0.0, 0.0),
        "lsf_sigma_kms": (2.8, 2.8), "continuum_0": (0.0, 0.0), "continuum_1": (0.0, 0.0),
        "continuum_2": (0.0, 0.0), "log_jitter": (np.log(1.0e-4), np.log(1.0e-4)),
    }
    shared = OrderObjective(model, order, 3)
    with pytest.raises(ValueError, match="different model or order"):
        fit_order(other, order, params(), bounds, objective=shared)
    with pytest.raises(ValueError, match="different continuum degree"):
        initial = TelluricParameters(
            {"H2O": 0.0}, 0.0, 0.0, 2.8, jnp.array([0.0, 0.0]), np.log(1.0e-4)
        )
        fit_order(model, order, initial, {**bounds}, objective=shared)


def test_trimming_keeps_the_original_samples_and_covers_the_window():
    """Trimming must not move the samples, or it changes what is modelled."""
    full = igrins_wavenumber_grid(2200.0, 2220.0, resolving_power=100_000.0,
                                  samples_per_resolution=4.0, margin_cm1=25.0)
    nu_min, nu_max = 1.0e7 / 2220.0, 1.0e7 / 2200.0
    trimmed = trim_wavenumber_grid(full, nu_min, nu_max, 2.0)
    assert trimmed.size < full.size
    assert np.all(np.isin(trimmed, full))
    # Same spacing and the same phase: a contiguous run of the original.
    start = int(np.flatnonzero(full == trimmed[0])[0])
    np.testing.assert_array_equal(trimmed, full[start : start + trimmed.size])
    assert trimmed[0] <= nu_min and trimmed[-1] >= nu_max
    # The bracketing sample may sit one step beyond the requested margin.
    step = trimmed[1] - trimmed[0]
    assert trimmed[0] >= nu_min - 2.0 - 2.0 * step
    assert trimmed[-1] <= nu_max + 2.0 + 2.0 * step


def test_trimming_refuses_to_lose_the_window_or_the_grid():
    full = igrins_wavenumber_grid(2200.0, 2220.0, margin_cm1=25.0)
    nu_min, nu_max = 1.0e7 / 2220.0, 1.0e7 / 2200.0
    with pytest.raises(ValueError, match="does not cover"):
        trim_wavenumber_grid(full[: full.size // 2], nu_min, nu_max, 5.0)
    with pytest.raises(ValueError, match="too few samples"):
        narrow = np.geomspace(4500.0, 4500.01, 10)
        trim_wavenumber_grid(narrow, 4500.004, 4500.005, 0.0)
    with pytest.raises(ValueError, match="invalid window or margin"):
        trim_wavenumber_grid(full, nu_max, nu_min, 5.0)
    with pytest.raises(ValueError, match="increasing"):
        trim_wavenumber_grid(full[::-1], nu_min, nu_max, 5.0)


def test_a_trimmed_grid_predicts_what_the_untrimmed_one_did():
    """Only the line-wing margin is dropped, not anything the model reaches."""
    full = igrins_wavenumber_grid(2200.0, 2220.0, resolving_power=100_000.0,
                                  samples_per_resolution=4.0, margin_cm1=25.0)
    nu_min, nu_max = 1.0e7 / 2220.0, 1.0e7 / 2200.0
    profile = AtmosphereProfile(
        pressure_edges_bar=[0.1, 0.5, 1.0], temperature_k=[240.0, 280.0],
        altitude_km=[10.0, 2.0], vmr={"H2O": [2.0e-4, 5.0e-3]},
    )
    wavelength = np.linspace(2201.0, 2219.0, 200)
    flux = np.ones(200)
    predictions = {}
    for margin in (25.0, 3.0):
        grid = trim_wavenumber_grid(full, nu_min, nu_max, margin)
        # The same lines on both grids; only where they are evaluated changes.
        line = np.exp(-0.5 * ((grid - 4530.0) / 0.02) ** 2) * 3.0e-22
        model = TelluricModel(profile, grid, ArrayOpacityBackend(
            {"H2O": np.vstack([line, 1.2 * line])}))
        order = SpectralOrder(wavelength, flux, np.full(200, 0.005))
        predictions[margin] = np.asarray(model.predict(order, params(sigma=2.0)))
    difference = np.abs(predictions[25.0] - predictions[3.0])
    assert difference.max() < 1.0e-6, difference.max()
