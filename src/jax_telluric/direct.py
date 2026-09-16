"""Direct Voigt opacity with static selection of possible line-core pairs.

ExoJAX's vectorized ``where`` evaluates Algorithm 916 even in distant wings.
Here only pairs that can enter that branch below a configured temperature
are sent through it. No lines or wings are removed.
"""

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import hjert, xsvector as lpf_xsvector
from exojax.special.faddeeva import asymptotic_wofz
from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.utils.constants import Tref_original


@jax.custom_jvp
def _wing(x, a):
    return jnp.real(asymptotic_wofz(x, a))


@_wing.defjvp
def _wing_jvp(primals, tangents):
    # Match ExoJAX's analytic Hjerting JVP, including its wing approximation.
    x, a = primals
    dx, da = tangents
    value = asymptotic_wofz(x, a)
    h, l = jnp.real(value), jnp.imag(value)
    return h, (2 * a * l - 2 * x * h) * dx + (
        2 * x * l + 2 * a * h - 2 / jnp.sqrt(jnp.pi)
    ) * da


@jax.custom_jvp
def _mixed_wing(x, a):
    """Float32 wing evaluation after forming accurate float64 coordinates."""
    return jnp.real(asymptotic_wofz(x.astype(jnp.float32), a.astype(jnp.float32))).astype(x.dtype)


@_mixed_wing.defjvp
def _mixed_wing_jvp(primals, tangents):
    # Algebraically simplify ExoJAX's derivative identity for its truncated
    # wing series. Direct evaluation of 2*x*Im(w)-2/sqrt(pi) catastrophically
    # cancels in float32. With z=x+i*a and u=1/z**2, the same identity is
    # dH/dx=Im(u+1.5*u**2+3.75*u**3)/sqrt(pi), dH/da=Re(...)/sqrt(pi).
    x, a = primals
    dx, da = tangents
    z = x.astype(jnp.float32) + 1j * a.astype(jnp.float32)
    u = 1 / (z * z)
    derivative = (u * (1 + u * (1.5 + 3.75 * u))) / jnp.sqrt(jnp.pi)
    return _mixed_wing(x, a), (
        jnp.imag(derivative).astype(x.dtype) * dx
        + jnp.real(derivative).astype(a.dtype) * da
    )


class SparseCoreDirect(OpaDirect):
    """ExoJAX Direct with identical branch selection and a compact core list.

    Above ``maximum_temperature_k`` a full Direct calculation is used. HITRAN
    air-pressure line shifts can be enabled explicitly; the sparse core list
    then covers every shift up to ``maximum_pressure_bar`` and falls back to
    the full calculation above it. HITRAN databases are supported.
    ``mixed_precision=True`` evaluates wing values and stable wing derivative
    coefficients in float32. Coordinates, line physics, cores, accumulation,
    and the returned cross sections retain the input precision.
    """

    def __init__(
        self,
        mdb,
        nu_grid,
        minimum_temperature_k=150.0,
        maximum_temperature_k=400.0,
        mixed_precision=False,
        pressure_shift=False,
        maximum_pressure_bar=2.0,
    ):
        if mdb.dbtype != "hitran":
            raise ValueError("SparseCoreDirect requires a HITRAN-style database")
        if (not np.isfinite(minimum_temperature_k) or minimum_temperature_k <= 0
                or not np.isfinite(maximum_temperature_k)
                or maximum_temperature_k < minimum_temperature_k):
            raise ValueError("temperature bounds must be finite, positive, and ordered")
        if not np.isfinite(maximum_pressure_bar) or maximum_pressure_bar <= 0:
            raise ValueError("maximum pressure must be finite and positive")
        if pressure_shift and not hasattr(mdb, "delta_air"):
            raise ValueError("pressure_shift requires HITRAN delta_air coefficients")
        super().__init__(mdb, nu_grid)
        self.minimum_temperature_k = float(minimum_temperature_k)
        self.maximum_temperature_k = float(maximum_temperature_k)
        self.maximum_pressure_bar = float(maximum_pressure_bar)
        self.mixed_precision = bool(mixed_precision)
        self.pressure_shift = bool(pressure_shift)
        self.delta_air = jnp.asarray(mdb.delta_air) if pressure_shift else None
        offsets = np.asarray(self.opainfo)
        sigma = np.asarray(doppler_sigma(mdb.nu_lines, maximum_temperature_k, mdb.molmass))
        shift_margin = (
            np.abs(np.asarray(mdb.delta_air)) * maximum_pressure_bar / 1.01325
            * Tref_original / minimum_temperature_k
            if pressure_shift else 0.0
        )
        possible_core = np.abs(offsets) <= (
            np.sqrt(222.0) * sigma[:, None] + np.asarray(shift_margin)[:, None]
            if pressure_shift else np.sqrt(222.0) * sigma[:, None]
        )
        self.core_line, self.core_grid = np.nonzero(possible_core)
        self.wing_mask = jnp.asarray(~possible_core)

    def xsvector(self, T, P, Pself=0.0):
        within_pressure_range = (
            (P >= 0.0) & (P <= self.maximum_pressure_bar)
            if self.pressure_shift else True
        )
        return jax.lax.cond(
            (T >= self.minimum_temperature_k) & (T <= self.maximum_temperature_k)
            & within_pressure_range,
            lambda: self._sparse_xsvector(T, P, Pself),
            lambda: self._full_xsvector(T, P, Pself),
        )

    def _offsets(self, T, P):
        if not self.pressure_shift:
            return self.opainfo
        # Match LBLRTM's RHORAT scaling: AER/HITRAN shifts are referenced to
        # one atmosphere at Tref, so their magnitude follows number density.
        density_ratio = (P / 1.01325) * (Tref_original / T)
        return self.opainfo - self.delta_air[:, None] * density_ratio

    def _line_parameters(self, T, P, Pself):
        mdb = self.mdb
        sigma = doppler_sigma(mdb.nu_lines, T, mdb.molmass)
        gamma = gamma_hitran(P, T, Pself, mdb.n_air, mdb.gamma_air, mdb.gamma_self) + gamma_natural(mdb.A)
        strength = line_strength(T, mdb.logsij0, mdb.nu_lines, mdb.elower,
                                 mdb.qr_interp(mdb.isotope, T, Tref_original), Tref_original)
        return sigma, gamma, strength

    def _full_xsvector(self, T, P, Pself):
        if not self.pressure_shift:
            return super(SparseCoreDirect, self).xsvector(T, P, Pself)
        sigma, gamma, strength = self._line_parameters(T, P, Pself)
        return lpf_xsvector(self._offsets(T, P), sigma, gamma, strength)

    def _sparse_xsvector(self, T, P, Pself):
        sigma, gamma, strength = self._line_parameters(T, P, Pself)
        scale = 1 / (jnp.sqrt(2.0) * sigma)
        a = scale * gamma
        weights = strength * scale / jnp.sqrt(jnp.pi)
        offsets = self._offsets(T, P)
        # Safe dummy coordinates prevent singular asymptotic evaluations in
        # excluded pairs and keep their reverse-mode derivatives finite.
        x = jnp.where(self.wing_mask, offsets * scale[:, None], 20.0)
        wing_function = _mixed_wing if self.mixed_precision else _wing
        wings = jnp.where(self.wing_mask, wing_function(x, a[:, None]), 0.0)
        spectrum = jnp.sum(wings * weights[:, None], axis=0)
        core_offsets = offsets[self.core_line, self.core_grid]
        cores = jax.vmap(hjert)(core_offsets * scale[self.core_line], a[self.core_line])
        return spectrum.at[self.core_grid].add(cores * weights[self.core_line])
