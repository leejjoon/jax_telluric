#!/usr/bin/env python
"""Benchmark a representative water-dominated IGRINS order."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform as platform_module
import statistics
import time


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "median_ms": 1000.0 * statistics.median(samples),
        "mean_ms": 1000.0 * statistics.mean(samples),
        "minimum_ms": 1000.0 * min(samples),
        "p95_ms": 1000.0 * ordered[p95_index],
    }


def timed_calls(function, argument, iterations: int):
    start = time.perf_counter()
    first = function(argument)
    import jax
    jax.block_until_ready(first)
    compile_and_first = time.perf_counter() - start
    for _ in range(2):
        jax.block_until_ready(function(argument))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(function(argument))
        samples.append(time.perf_counter() - start)
    return compile_and_first, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("direct", "direct_sparse"), default="direct")
    parser.add_argument("--vectorize-layers", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--v2", type=float, default=5020.0)
    parser.add_argument("--pixels", type=int, default=512)
    args = parser.parse_args()
    if args.iterations < 3:
        raise ValueError("at least three timed iterations are required")

    # These must be selected before importing JAX or jax_telluric.
    os.environ["JAX_PLATFORMS"] = "cuda" if args.platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jax_telluric import (
        AERLineDatabase,
        ExoJAXOpacityBackend,
        SpectralOrder,
        TelluricModel,
        TelluricParameters,
        igrins_wavenumber_grid,
        load_atmosphere_csv,
    )

    root = Path(__file__).resolve().parents[1]
    profile = load_atmosphere_csv(root / "data/profiles/example_midlatitude.csv")
    # A 20 cm-1 sub-order keeps Direct LPF practical on CPU while retaining
    # almost two thousand water lines after the required wing margin.
    limits = (5000.0, args.v2)
    nu_grid = igrins_wavenumber_grid(1.0e7 / limits[1], 1.0e7 / limits[0])
    database = AERLineDatabase(
        root / "data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule/01_H2O/01_H2O",
        "H2O",
        limits,
    )
    backend = ExoJAXOpacityBackend.prepare({"H2O": database}, nu_grid, methods=args.method,
                                         vectorize_layers=args.vectorize_layers,
                                         mixed_precision=args.mixed_precision)
    model = TelluricModel(profile, nu_grid, backend)
    wavelength = np.linspace(1.0e7 / limits[1], 1.0e7 / limits[0], args.pixels)
    order = SpectralOrder(wavelength, np.ones(args.pixels), np.full(args.pixels, 0.01), zenith_angle_deg=30.0)

    def parameters(vector):
        return TelluricParameters(
            {"H2O": vector[0]}, vector[1], vector[2], vector[3], vector[4:7], vector[7]
        )

    def forward(vector):
        return model.predict(order, parameters(vector))

    def objective(vector):
        prediction = forward(vector)
        return jnp.mean((prediction - 0.9) ** 2)

    vector = jnp.asarray([0.0, 0.2, 1.0e-5, 2.8, 0.0, 0.0, 0.0, -9.0])
    compiled_forward = jax.jit(forward)
    compiled_gradient = jax.jit(jax.value_and_grad(objective))

    forward_first, forward_samples = timed_calls(compiled_forward, vector, args.iterations)

    gradient_first, gradient_samples = timed_calls(compiled_gradient, vector, args.iterations)
    result = {
        "platform": args.platform,
        "method": args.method,
        "vectorize_layers": args.vectorize_layers,
        "mixed_precision": args.mixed_precision,
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "python": platform_module.python_version(),
        "case": {
            "molecule": "H2O",
            "line_source": "AER 3.9",
            "wavenumber_cm1": list(limits),
            "atmospheric_layers": len(profile.temperature_k),
            "spectral_lines_with_25_cm-1_margin": len(database.nu_lines),
            "high_resolution_samples": len(nu_grid),
            "detector_pixels": len(wavelength),
            "resolving_power": 45_000,
        },
        "forward": {"compile_and_first_s": forward_first, **summarize(forward_samples)},
        "value_and_gradient": {"compile_and_first_s": gradient_first, **summarize(gradient_samples)},
        "iterations": args.iterations,
        "x64": jax.config.x64_enabled,
        "device_kind": jax.devices()[0].device_kind,
        "core_pairs": len(backend.calculators["H2O"].core_line) if args.method == "direct_sparse" else None,
    }
    # Several parameter points allow cross-method/device numerical checks.
    points = [vector, vector.at[0].set(-0.7), vector.at[0].set(0.7)]
    fluxes = [np.asarray(compiled_forward(p)) for p in points]
    gradients = [np.asarray(compiled_gradient(p)[1]) for p in points]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), flux=fluxes, gradient=gradients)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
