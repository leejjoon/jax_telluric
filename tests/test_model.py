import jax
import jax.numpy as jnp
import numpy as np

from jax_telluric import (
    ArrayOpacityBackend,
    AtmosphereProfile,
    SpectralOrder,
    ReferenceWaterContinuum,
    LBLRTMOpticalDepthCorrection,
    MTCKDWaterContinuum,
    TelluricModel,
    TelluricParameters,
    fit_order,
    igrins_wavenumber_grid,
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
