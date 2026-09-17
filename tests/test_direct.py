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


def test_sparse_direct_applies_hitran_air_pressure_shift():
    database = SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.array([5000.0]),
        logsij0=jnp.log(jnp.array([1e-22])),
        elower=np.array([100.]),
        n_air=np.array([0.7]),
        gamma_air=np.array([0.01]),
        gamma_self=np.array([0.01]),
        delta_air=np.array([0.1]),
        A=np.array([0.1]),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference)**1.5,
    )
    grid = np.linspace(4999.7, 5000.3, 1201)
    shifted = SparseCoreDirect(database, grid, pressure_shift=True)
    at_zero_pressure = shifted.xsvector(296.0, 0.0)
    at_one_atmosphere = shifted.xsvector(296.0, 1.01325)

    zero_peak = grid[int(jnp.argmax(at_zero_pressure))]
    pressure_peak = grid[int(jnp.argmax(at_one_atmosphere))]
    assert zero_peak == pytest.approx(5000.0, abs=0.0005)
    assert pressure_peak == pytest.approx(5000.1, abs=0.0005)
    cold_half_atmosphere = shifted.xsvector(148.0, 0.506625)
    cold_peak = grid[int(jnp.argmax(cold_half_atmosphere))]
    assert cold_peak == pytest.approx(5000.1, abs=0.0005)
    gradient = jax.grad(lambda pressure: jnp.sum(
        jnp.asarray(grid) * shifted.xsvector(296.0, pressure)
    ))(0.7)
    assert jnp.isfinite(gradient)


def test_sparse_direct_holds_no_line_by_grid_matrix():
    """The dense offset matrix is what puts a fine grid out of memory.

    ExoJAX's ``OpaDirect`` keeps ``nu_grid[None, :] - nu_lines[:, None]`` on the
    device. At terrestrial line densities that is hundreds of megabytes, re-read
    once per layer. Everything here is rebuilt from the two 1-D vectors instead,
    so nothing of that shape may survive construction.
    """

    database = SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.linspace(4999.0, 5002.0, 400),
        logsij0=jnp.log(jnp.full(400, 1e-23)),
        elower=np.full(400, 200.0),
        n_air=np.full(400, 0.7), gamma_air=np.full(400, 0.07),
        gamma_self=np.full(400, 0.3), A=np.full(400, 0.1),
        delta_air=np.full(400, -0.01),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference) ** 1.5,
    )
    grid = np.linspace(4998.0, 5003.0, 2000)
    calculator = SparseCoreDirect(database, grid, pressure_shift=True)
    assert calculator.opainfo is None
    stored = [
        value for value in vars(calculator).values()
        if hasattr(value, "ndim") and hasattr(value, "shape") and value.ndim == 2
    ]
    assert stored == [], f"a 2-D array survived: {[v.shape for v in stored]}"
    # The core list is the one thing allowed to scale with line by grid, and
    # only because it is sparse. This fixture crowds 400 lines into 3 cm-1, so
    # its 6% is far denser than a real order; the measured IGRINS figure is
    # 0.14%.
    assert calculator.core_line.size < 0.1 * database.nu_lines.size * grid.size
    # Rebuilding the offsets must not change the answer. OpaDirect applies no
    # pressure shift, so the comparison uses the unshifted calculator.
    np.testing.assert_allclose(
        SparseCoreDirect(database, grid).xsvector(275.0, 0.7, 0.01),
        OpaDirect(database, grid).xsvector(275.0, 0.7, 0.01), rtol=2e-12, atol=1e-35,
    )


def test_core_list_matches_the_dense_selection_it_replaced():
    """Binary search must reproduce the dense scan exactly, order included.

    The core and wing branches are evaluated by different code, and the scatter
    add sums in list order, so a different core list is a different answer.
    """

    from exojax.database.core.broadening import doppler_sigma
    from exojax.utils.constants import Tref_original

    rng = np.random.default_rng(3)
    count = 300
    database = SimpleNamespace(
        dbtype="hitran", isotope=1, molmass=18.0,
        nu_lines=np.sort(rng.uniform(4999.0, 5002.0, count)),
        logsij0=jnp.log(jnp.full(count, 1e-23)), elower=np.full(count, 200.0),
        n_air=np.full(count, 0.7), gamma_air=np.full(count, 0.07),
        gamma_self=np.full(count, 0.3), A=np.full(count, 0.1),
        delta_air=rng.uniform(-0.02, 0.0, count),
        qr_interp=lambda isotope, temperature, reference: (temperature / reference) ** 1.5,
    )
    grid = np.linspace(4998.0, 5003.0, 1500)
    for pressure_shift in (False, True):
        calculator = SparseCoreDirect(
            database, grid, minimum_temperature_k=200.0, maximum_temperature_k=320.0,
            pressure_shift=pressure_shift, maximum_pressure_bar=1.1,
        )
        sigma = np.asarray(doppler_sigma(database.nu_lines, 320.0, database.molmass))
        margin = (np.abs(database.delta_air) * 1.1 / 1.01325 * Tref_original / 200.0
                  if pressure_shift else 0.0)
        half_width = np.sqrt(222.0) * sigma + margin
        offsets = grid[None, :] - database.nu_lines[:, None]
        line, column = np.nonzero(np.abs(offsets) <= half_width[:, None])
        np.testing.assert_array_equal(calculator.core_line, line)
        np.testing.assert_array_equal(calculator.core_grid, column)
