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
    """Ordered mapping between named parameters and the optimizer's vector.

    ``stellar_velocity_kms`` is present only when the order carries a
    model-grid source. Without one it cannot affect the prediction, so freeing
    it would contribute an identically zero gradient column and a singular
    Hessian rather than an unconstrained parameter.
    """

    def __init__(
        self,
        species: Sequence[str],
        continuum_size: int,
        include_stellar_velocity: bool = False,
    ) -> None:
        self.species = tuple(species)
        self.continuum_size = continuum_size
        self.include_stellar_velocity = include_stellar_velocity
        names = [*self.species, "velocity_kms"]
        if include_stellar_velocity:
            names.append("stellar_velocity_kms")
        names.extend(("wavelength_stretch", "lsf_sigma_kms"))
        names.extend(f"continuum_{index}" for index in range(continuum_size))
        names.append("log_jitter")
        self.names = tuple(names)
        self._offset = {name: index for index, name in enumerate(self.names)}

    @property
    def continuum_start(self) -> int:
        return self._offset["continuum_0"] if self.continuum_size else 0

    def pack(self, params: TelluricParameters) -> np.ndarray:
        values = {
            **{name: params.log_column_scales[name] for name in self.species},
            "velocity_kms": params.velocity_kms,
            "stellar_velocity_kms": params.stellar_velocity_kms,
            "wavelength_stretch": params.wavelength_stretch,
            "lsf_sigma_kms": params.lsf_sigma_kms,
            "log_jitter": params.log_jitter,
            **{
                f"continuum_{index}": value
                for index, value in enumerate(np.asarray(params.continuum_coeffs))
            },
        }
        return np.asarray([values[name] for name in self.names], dtype=float)

    def unpack(self, vector: jnp.ndarray) -> TelluricParameters:
        start = self.continuum_start
        return TelluricParameters(
            log_column_scales={name: vector[self._offset[name]] for name in self.species},
            velocity_kms=vector[self._offset["velocity_kms"]],
            wavelength_stretch=vector[self._offset["wavelength_stretch"]],
            lsf_sigma_kms=vector[self._offset["lsf_sigma_kms"]],
            continuum_coeffs=vector[start : start + self.continuum_size],
            log_jitter=vector[self._offset["log_jitter"]],
            stellar_velocity_kms=(
                vector[self._offset["stellar_velocity_kms"]]
                if self.include_stellar_velocity
                else 0.0
            ),
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

class OrderObjective:
    """Compiled value and gradient for one model and one order, reusable.

    XLA compilation of a telluric forward model costs seconds while one
    evaluation costs milliseconds, so what a staged fit spends is dominated by
    how many times it compiles. ``fit_order`` builds a fresh ``jax.jit``
    closure per call, and a new Python function object is a new cache entry, so
    four stages over one page compile the same graph four times.

    This compiles over the *full* parameter vector, leaving the bounds, the
    free set, and the optimizer's rescaling outside. Those are exactly what
    changes between stages, so one instance passed to every stage of a page
    replaces N compilations with one.
    """

    def __init__(
        self,
        model: TelluricModel,
        order: SpectralOrder,
        continuum_size: int,
    ) -> None:
        self.model = model
        self.order = order
        self.codec = _ParameterCodec(
            model.species,
            int(continuum_size),
            include_stellar_velocity=order.source_flux_model_grid is not None,
        )
        uncertainty = jnp.asarray(order.uncertainty)
        mask = jnp.asarray(order.mask)
        flux = jnp.where(mask, jnp.asarray(order.flux), 0.0)

        def objective(vector: jnp.ndarray) -> jnp.ndarray:
            parameters = self.codec.unpack(vector)
            prediction = model.predict(order, parameters)
            variance = uncertainty**2 + jnp.exp(2.0 * jnp.asarray(parameters.log_jitter))
            # Dividing inside the logarithm removes a parameter-independent
            # constant. L-BFGS-B uses relative objective reduction to stop, and
            # the large negative normalization term otherwise causes premature
            # convergence for high-S/N orders.
            terms = (flux - prediction) ** 2 / variance + jnp.log(variance / uncertainty**2)
            return 0.5 * jnp.sum(jnp.where(mask, terms, 0.0))

        self._value_and_grad = jax.jit(jax.value_and_grad(objective))

    def __call__(self, vector: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = self._value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=float)


def fit_order(
    model: TelluricModel,
    order: SpectralOrder,
    initial: TelluricParameters,
    bounds: Mapping[str, tuple[float, float]],
    objective: OrderObjective | None = None,
) -> FitResult:
    """Fit one spectral order using a heteroscedastic Gaussian likelihood.

    Bound keys are molecule names plus ``velocity_kms``,
    ``wavelength_stretch``, ``lsf_sigma_kms``, ``continuum_0`` ... and
    ``log_jitter``, and additionally ``stellar_velocity_kms`` when the order
    carries ``source_flux_model_grid``. Setting a bound's lower and upper
    values equal pins that parameter.

    Pass ``objective`` -- an :class:`OrderObjective` built once for this model
    and order -- to reuse one compilation across the stages of a staged fit.
    """

    if objective is None:
        objective = OrderObjective(model, order, len(np.asarray(initial.continuum_coeffs)))
    elif objective.model is not model or objective.order is not order:
        raise ValueError("the compiled objective was built for a different model or order")
    codec = objective.codec
    if codec.continuum_size != len(np.asarray(initial.continuum_coeffs)):
        raise ValueError("the compiled objective was built for a different continuum degree")
    names = list(codec.names)
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

    def expand(scaled_vector: np.ndarray) -> np.ndarray:
        full = center.copy()
        full[free_indices] = center[free_indices] + scales * np.asarray(scaled_vector)
        return full

    def scipy_objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        # The rescaling is affine, so the chain rule is one multiplication and
        # the compiled function never has to know which parameters are free.
        value, gradient = objective(expand(vector))
        return value, gradient[free_indices] * scales

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
            fun = float(objective(expand(initial_scaled))[0])
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
