# jax-telluric

`jax-telluric` is an experimental differentiable forward model for terrestrial
absorption in high-resolution spectra. It wraps ExoJAX opacity calculators with
a layered Earth atmosphere, slant-path transmission, and an observation model
for wavelength corrections, continuum, and a Gaussian line-spread function.

The first validation target is LBLRTM 12.17 in representative IGRINS H- and
K-band intervals. See `docs/lblrtm_exojax_research.md` for the rationale and
`docs/validation.md` for the reference-data workflow.

## Environment

```bash
UV_CACHE_DIR=.uv-cache uv sync --dev
UV_CACHE_DIR=.uv-cache uv run pytest
```

JAX runs in 64-bit mode inside this package. Database downloads and LBLRTM
binaries belong under the ignored `data/` directories.

To reproduce the reference compilers, builds, and line-file download:

```bash
./scripts/bootstrap_lblrtm.sh
```

After bootstrapping, generate the committed-style LBLRTM reference fixture
from the bundled atmospheric profile with:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_lblrtm_reference.py
```

The high-resolution TAPE files remain under `data/lblrtm/run_reference/`; the
script writes an IGRINS-resolution spectrum and provenance JSON under
`tests/data/`.

Run the first end-to-end comparison, using AER 3.9 CO parameters in both
LBLRTM and ExoJAX, with:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_aer_co.py
```

The script records the error metrics in `tests/data/aer_co_validation.json`.
The `AERLineDatabase` adapter can likewise load the H2O, CO2, O3, N2O, CH4,
and O2 files extracted by the bootstrap script.

## Performance benchmark

The benchmark measures a six-layer, water-dominated 5000--5020 cm-1 sub-order,
including about 1,900 AER lines after wing padding, ExoJAX Direct opacity,
slant transmission, LSF convolution, and 512 detector pixels. It reports
compilation separately from synchronized steady-state calls:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py \
  --platform cpu --output benchmarks/results/cpu.json
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py \
  --platform gpu --output benchmarks/results/gpu.json
```

See `docs/performance.md` for the measured CPU/GPU results and interpretation.

For faster terrestrial HITRAN/AER Direct opacity with the same line and wing
formulas, prepare the backend with:

```python
backend = ExoJAXOpacityBackend.prepare(
    databases, nu_grid, methods="direct_sparse", vectorize_layers=True
)
```

This selects a compact list of possible line-core pairs and batches the
atmospheric layers. Temperatures above 400 K automatically use original Direct.
Use `methods="direct"` to retain the reference path.

Add `mixed_precision=True` to this preparation call for float32 wing
calculations with float64 line centers, cores, accumulation, and instrument
model. On the tested full H2O order this gives another roughly 11x forward
and 8x gradient speedup with a maximum flux change of 3.5e-8. See the
mixed-precision section in `docs/performance.md` for scope and reproduction,
and [extended accuracy validation](docs/mixed_precision_validation.md) for
multi-species, atmospheric-state, spectral-derivative, and fit-recovery checks.
The extended checks pass, including paired mixed-precision and float64 fits.

## Optional LBLRTM-corrected mode

The default `accuracy_mode="fast"` preserves the optimized behavior above.
For a fixed atmospheric profile and order grid, an offline LBLRTM correction
template can add MT_CKD H2O continuum, line coupling, speed-dependent line
shape residuals, distant-line absorption, and the remaining reference
background while retaining JAX derivatives during fitting:

```python
correction = LBLRTMOpticalDepthCorrection.load("data/corrections/order.npz")
model = TelluricModel(
    profile, nu_grid, backend,
    accuracy_mode="lblrtm_corrected", correction=correction,
)
```

Generate a template with `scripts/build_lblrtm_correction.py`. In the tested
5000--5020 cm-1 order, the correction reduced the 99th-percentile absolute
error against full LBLRTM from 0.102 to 6.61e-5 with negligible steady-state
GPU overhead. See [the corrected-mode guide](docs/lblrtm_corrected_mode.md)
for usage, assumptions, and reproduction.
