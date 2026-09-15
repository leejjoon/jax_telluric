"""Compare optimized opacity with the actual ExoJAX kernel and its JVP."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_telluric  # enable float64 before importing ExoJAX
from exojax.opacity import OpaDirect
from jax_telluric.direct import SparseCoreDirect, _wing, _mixed_wing
from jax_telluric import ExoJAXOpacityBackend


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_sparse_core_matches_exojax_values_and_derivatives(mixed_precision):
    database = SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.array([5000.0, 5000.1, 5001.0]),
        logsij0=jnp.log(jnp.array([1e-22, 2e-23, 1e-24])),
        elower=np.array([100., 300., 50.]),
        n_air=np.array([0.7, 0.6, 0.8]),
        gamma_air=np.array([0.07, 0.08, 0.09]),
        gamma_self=np.array([0.3, 0.4, 0.2]),
        A=np.array([0.1, 0.2, 0.3]),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference)**1.5,
    )
    grid = np.linspace(4998., 5003., 801)
    original = OpaDirect(database, grid)
    optimized = SparseCoreDirect(database, grid, mixed_precision=mixed_precision)
    value_rtol = 1e-6 if mixed_precision else 2e-12
    # Mixed precision targets 1e-5 relative gradient accuracy; a few ppm
    # difference is expected when summing signed pressure derivatives.
    gradient_rtol = 1e-5 if mixed_precision else 1e-9
    weights = jnp.linspace(0.3, 1.2, len(grid))
    for temperature in (180., 275., 400., 401., 650.):
        for pressure in (0.001, 0.7):
            point = jnp.array([temperature, pressure, pressure * 0.02])
            reference = original.xsvector(*point)
            actual = optimized.xsvector(*point)
            np.testing.assert_allclose(actual, reference, rtol=value_rtol, atol=1e-35)
            def statistic(calculator, parameters):
                return jnp.sum(weights * calculator.xsvector(*parameters)) * 1e22
            old_grad = jax.grad(lambda p: statistic(original, p))(point)
            new_grad = jax.grad(lambda p: statistic(optimized, p))(point)
            np.testing.assert_allclose(new_grad, old_grad, rtol=gradient_rtol, atol=1e-9)

    # The public batched adapter must preserve each layer's self pressure.
    adapter = ExoJAXOpacityBackend({"H2O": optimized}, vectorize_layers=True)
    temperatures = jnp.array([200., 275., 450.])
    pressures = jnp.array([0.01, 0.5, 0.9])
    self_pressures = pressures * jnp.array([0.001, 0.02, 0.1])
    expected = jnp.stack([original.xsvector(t, p, s) for t, p, s in
                          zip(temperatures, pressures, self_pressures)])
    actual = adapter.cross_sections(temperatures, pressures, {"H2O": self_pressures})["H2O"]
    np.testing.assert_allclose(actual, expected, rtol=value_rtol, atol=1e-35)


def test_mixed_wing_derivative_survives_far_wing_cancellation():
    points = jnp.array([[11., 0.001], [100., 1.], [10000., 0.1], [10000., 100.]])
    original = jax.vmap(jax.grad(lambda p: _wing(*p)))(points)
    mixed = jax.vmap(jax.grad(lambda p: _mixed_wing(*p)))(points)
    np.testing.assert_allclose(mixed, original, rtol=2e-6, atol=1e-15)
    # A naive float32 subtraction loses the positive damping derivative here.
    assert mixed[2, 1] > 5e-9
