"""Bounded MAP fitting with JAX derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .model import TelluricModel
from .types import SpectralOrder, TelluricParameters


_MAX_PROJECTED_SCALED_GRADIENT = 1.0


@dataclass(frozen=True)
class FitResult:
    parameters: TelluricParameters
    covariance: np.ndarray | None
    transmission: np.ndarray
    model_flux: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    objective: float
    iterations: int


class _ParameterCodec:
    def __init__(self, species: Sequence[str], continuum_size: int) -> None:
        self.species = tuple(species)
        self.continuum_size = continuum_size

    def pack(self, params: TelluricParameters) -> np.ndarray:
        return np.asarray(
            [*(params.log_column_scales[name] for name in self.species), params.velocity_kms,
             params.wavelength_stretch, params.lsf_sigma_kms,
             *np.asarray(params.continuum_coeffs), params.log_jitter],
            dtype=float,
        )

    def unpack(self, vector: jnp.ndarray) -> TelluricParameters:
        nspecies = len(self.species)
        start_continuum = nspecies + 3
        return TelluricParameters(
            log_column_scales={name: vector[index] for index, name in enumerate(self.species)},
            velocity_kms=vector[nspecies],
            wavelength_stretch=vector[nspecies + 1],
            lsf_sigma_kms=vector[nspecies + 2],
            continuum_coeffs=vector[start_continuum : start_continuum + self.continuum_size],
            log_jitter=vector[-1],
        )


def _parameter_scale(name: str, lower: float, upper: float) -> float:
    """Return a useful physical step represented by one optimizer unit."""
    natural = 1.0e-5 if name == "wavelength_stretch" else 0.01 if name.startswith("continuum_") else 1.0
    if np.isfinite(lower) and np.isfinite(upper):
        natural = min(natural, 0.5 * (upper - lower))
    return natural


def _projected_gradient(vector, gradient, lower, upper):
    """Gradient after removing directions forbidden by active bounds."""
    projected = np.asarray(gradient, dtype=float).copy()
    tolerance = 1.0e-10
    projected[(vector <= lower + tolerance) & (projected > 0.0)] = 0.0
    projected[(vector >= upper - tolerance) & (projected < 0.0)] = 0.0
    return projected

def fit_order(
    model: TelluricModel,
    order: SpectralOrder,
    initial: TelluricParameters,
    bounds: Mapping[str, tuple[float, float]],
) -> FitResult:
    """Fit one spectral order using a heteroscedastic Gaussian likelihood.

    Bound keys are molecule names plus ``velocity_kms``,
    ``wavelength_stretch``, ``lsf_sigma_kms``, ``continuum_0`` ... and
    ``log_jitter``.
    """

    codec = _ParameterCodec(model.opacity.species, len(np.asarray(initial.continuum_coeffs)))
    names = [*codec.species, "velocity_kms", "wavelength_stretch", "lsf_sigma_kms"]
    names.extend(f"continuum_{index}" for index in range(codec.continuum_size))
    names.append("log_jitter")
    missing = [name for name in names if name not in bounds]
    if missing:
        raise ValueError(f"missing parameter bounds: {', '.join(missing)}")

    full_initial = codec.pack(initial)
    bound_array = np.asarray([bounds[name] for name in names], dtype=float)
    if np.any(np.isnan(bound_array)) or np.any(bound_array[:, 0] > bound_array[:, 1]):
        raise ValueError("parameter bounds must be ordered and not NaN")
    if np.any(~np.isfinite(full_initial)):
        raise ValueError("initial parameters must be finite")
    full_initial = np.clip(full_initial, bound_array[:, 0], bound_array[:, 1])
    free = bound_array[:, 0] < bound_array[:, 1]
    fixed = ~free
    full_initial[fixed] = bound_array[fixed, 0]
    free_indices = np.flatnonzero(free)
    scales = np.asarray(
        [_parameter_scale(names[index], *bound_array[index]) for index in free_indices]
    )
    center = full_initial.copy()
    initial_scaled = np.zeros(len(free_indices))
    scaled_bounds = [
        ((bound_array[index, 0] - center[index]) / scale,
         (bound_array[index, 1] - center[index]) / scale)
        for index, scale in zip(free_indices, scales)
    ]

    flux = jnp.asarray(order.flux)
    uncertainty = jnp.asarray(order.uncertainty)
    mask = jnp.asarray(order.mask)
    flux = jnp.where(mask, flux, 0.0)

    center_jax = jnp.asarray(center)
    scales_jax = jnp.asarray(scales)

    def full_vector(scaled_vector: jnp.ndarray) -> jnp.ndarray:
        return center_jax.at[free_indices].set(
            center_jax[free_indices] + scales_jax * scaled_vector
        )

    def objective(scaled_vector: jnp.ndarray) -> jnp.ndarray:
        parameters = codec.unpack(full_vector(scaled_vector))
        prediction = model.predict(order, parameters)
        variance = uncertainty**2 + jnp.exp(2.0 * jnp.asarray(parameters.log_jitter))
        # Dividing inside the logarithm removes a parameter-independent
        # constant. L-BFGS-B uses relative objective reduction to stop, and
        # the large negative normalization term otherwise causes premature
        # convergence for high-S/N orders.
        terms = (flux - prediction) ** 2 / variance + jnp.log(variance / uncertainty**2)
        return 0.5 * jnp.sum(jnp.where(mask, terms, 0.0))

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=float)

    if free_indices.size:
        result = minimize(
            scipy_objective,
            initial_scaled,
            method="L-BFGS-B",
            jac=True,
            bounds=scaled_bounds,
            options={"ftol": 1.0e-12, "gtol": 1.0e-6, "maxiter": 1000, "maxls": 50},
        )
        final_scaled = np.asarray(result.x)
        _, final_gradient = scipy_objective(final_scaled)
        scaled_lower = np.asarray([value[0] for value in scaled_bounds])
        scaled_upper = np.asarray([value[1] for value in scaled_bounds])
        gradient_norm = float(np.max(np.abs(_projected_gradient(
            final_scaled, final_gradient, scaled_lower, scaled_upper
        ))))
        # Float32 wing values have a small quantization floor even though
        # their custom derivative remains smooth. One optimizer unit is a
        # physically useful parameter step, so this still rejects the former
        # false convergence by more than three orders of magnitude.
        converged = bool(result.success) and gradient_norm <= _MAX_PROJECTED_SCALED_GRADIENT
        message = str(result.message)
        if result.success and not converged:
            message = (
                f"{message}; projected scaled gradient {gradient_norm:.3g} "
                f"exceeds {_MAX_PROJECTED_SCALED_GRADIENT:g}"
            )
        final_vector = center.copy()
        final_vector[free_indices] += scales * final_scaled
    else:
        final_scaled = initial_scaled
        final_vector = center
        converged = True
        message = "all parameters fixed"
        gradient_norm = 0.0

        class FixedResult:
            fun = float(objective(jnp.asarray(initial_scaled)))
            nit = 0
            hess_inv = None

        result = FixedResult()

    parameters = codec.unpack(jnp.asarray(final_vector))
    model_flux = np.asarray(model.predict(order, parameters))
    residuals = np.asarray(order.flux) - model_flux
    covariance = None
    if hasattr(result.hess_inv, "todense"):
        scaled_covariance = np.asarray(result.hess_inv.todense())
        covariance = np.zeros((len(names), len(names)))
        covariance[np.ix_(free_indices, free_indices)] = (
            scales[:, None] * scaled_covariance * scales[None, :]
        )
    normalization = 0.5 * np.sum(np.where(np.asarray(order.mask), np.log(np.asarray(order.uncertainty) ** 2), 0.0))
    return FitResult(
        parameters=parameters,
        covariance=covariance,
        transmission=np.asarray(model.transmission(parameters, order.zenith_angle_deg)),
        model_flux=model_flux,
        residuals=residuals,
        success=converged,
        message=message,
        objective=float(result.fun + normalization),
        iterations=int(result.nit),
    )
