"""Adapter from prepared ExoJAX opacity calculators to the model seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import jax


@dataclass(frozen=True)
class ExoJAXOpacityBackend:
    """Evaluate one prepared ExoJAX opacity calculator per molecule.

    Direct calculators preserve terrestrial self pressure, with optional
    layer batching via ``vmap``. `OpaPremodit` uses its matrix calculation and therefore
    applies air broadening only, matching ExoJAX 2.5's public interface.
    """

    calculators: Mapping[str, object]
    vectorize_layers: bool = False
    layer_chunk_size: int | None = None

    @classmethod
    def prepare(
        cls,
        databases: Mapping[str, object],
        wavenumber_cm1,
        methods: str | Mapping[str, str] = "direct",
        temperature_range_k: tuple[float, float] | None = None,
        maximum_pressure_bar: float = 2.0,
        vectorize_layers: bool = False,
        mixed_precision: bool = False,
        pressure_shift: bool = False,
        layer_chunk_size: int | None = None,
    ) -> "ExoJAXOpacityBackend":
        """Construct Direct, sparse-core Direct, or PreMODIT calculators.

        A method mapping supports the recommended terrestrial combination of
        Direct for H2O (self broadening) and PreMODIT for trace gases.
        ``direct_sparse`` preserves the ExoJAX formulas but restricts expensive
        core evaluation to potentially contributing line/grid pairs. It supports
        HITRAN-style databases; outside its configured temperature or pressure
        bounds it falls back to a full Direct calculation.
        ``vectorize_layers=True`` reduces GPU compile time for fixed profiles.
        Its wing array is dense in lines by grid by layer, so on a fine grid it
        can exceed device memory; ``layer_chunk_size`` then vectorizes that many
        layers at a time. Equal-sized chunks compile once and reuse, so peak
        memory falls by the chunk fraction while compile time barely moves.
        A Python loop over single layers (``vectorize_layers=False``) also fits
        in memory but unrolls the whole calculation once per layer, which
        multiplies compile time instead.
        ``mixed_precision=True`` applies only to ``direct_sparse``: evaluate
        wings and stable wing derivative coefficients in float32 while keeping
        cores, line physics, and cross-section accumulation in float64.
        ``pressure_shift=True`` applies HITRAN air-pressure line shifts in
        ``direct_sparse`` calculators. It is opt-in so the default remains
        numerically compatible with ExoJAX 2.5 Direct.
        Supply the fixed profile's temperature range and maximum pressure to
        keep the pressure-shifted sparse core list as compact as possible.
        """

        from exojax.opacity import OpaDirect, OpaPremodit

        calculators = {}
        for species, database in databases.items():
            method = (
                methods.get(species, methods.get(species.upper(), "direct"))
                if isinstance(methods, Mapping)
                else methods
            )
            if method == "direct":
                calculators[species] = OpaDirect(database, nu_grid=wavenumber_cm1)
            elif method == "direct_sparse":
                from .direct import SparseCoreDirect
                sparse_bounds = {} if temperature_range_k is None else {
                    "minimum_temperature_k": temperature_range_k[0],
                    "maximum_temperature_k": temperature_range_k[1],
                }
                calculators[species] = SparseCoreDirect(
                    database,
                    wavenumber_cm1,
                    mixed_precision=mixed_precision,
                    pressure_shift=pressure_shift,
                    maximum_pressure_bar=maximum_pressure_bar,
                    **sparse_bounds,
                )
            elif method == "premodit":
                calculators[species] = OpaPremodit(
                    database,
                    nu_grid=wavenumber_cm1,
                    auto_trange=temperature_range_k,
                    allow_32bit=False,
                    delete_mdb_after_init=False,
                )
            else:
                raise ValueError(f"unknown opacity method for {species}: {method}")
        if layer_chunk_size is not None and layer_chunk_size < 1:
            raise ValueError("layer_chunk_size must be a positive number of layers")
        return cls(
            calculators,
            vectorize_layers=vectorize_layers,
            layer_chunk_size=layer_chunk_size,
        )

    def __post_init__(self) -> None:
        if not self.calculators:
            raise ValueError("at least one ExoJAX opacity calculator is required")
        normalized = {name.upper(): calculator for name, calculator in self.calculators.items()}
        for name, calculator in normalized.items():
            if not hasattr(calculator, "xsvector") or not hasattr(calculator, "xsmatrix"):
                raise TypeError(f"{name} calculator does not satisfy the ExoJAX opacity interface")
        object.__setattr__(self, "calculators", normalized)

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(self.calculators)

    def _vectorized(self, calculator, temperature_k, pressure_bar, self_pressure_bar):
        layers = temperature_k.shape[0]
        chunk = self.layer_chunk_size or layers
        if chunk >= layers:
            return jax.vmap(calculator.xsvector)(temperature_k, pressure_bar, self_pressure_bar)
        parts = [
            jax.vmap(calculator.xsvector)(
                temperature_k[start : start + chunk],
                pressure_bar[start : start + chunk],
                self_pressure_bar[start : start + chunk],
            )
            for start in range(0, layers, chunk)
        ]
        return jnp.concatenate(parts, axis=0)

    def cross_sections(self, temperature_k, pressure_bar, partial_pressure_bar):
        result = {}
        for species, calculator in self.calculators.items():
            # OpaDirect is the only ExoJAX 2.5 calculator with a Pself argument.
            if getattr(calculator, "method", None) == "lpf":
                if self.vectorize_layers:
                    result[species] = self._vectorized(
                        calculator, temperature_k, pressure_bar, partial_pressure_bar[species]
                    )
                    continue
                result[species] = jnp.stack(
                    [
                        calculator.xsvector(temperature_k[index], pressure_bar[index], partial_pressure_bar[species][index])
                        for index in range(temperature_k.shape[0])
                    ]
                )
            else:
                result[species] = calculator.xsmatrix(temperature_k, pressure_bar)
        return result
