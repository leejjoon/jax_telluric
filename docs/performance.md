# Performance benchmark

## Mixed precision (opt-in)

On the 5000--5100 cm-1 H2O order (4,669 lines, six layers, 2,048 pixels),
adding `mixed_precision=True` to sparse Direct gives:

| GPU operation | Sparse Direct float64 | Mixed precision | Further speedup |
|---|---:|---:|---:|
| Forward | 61.5 ms | 5.53 ms | 11.1x |
| Objective + gradient | 73.3 ms | 9.12 ms | 8.0x |

Both runs used 30 synchronized timed calls on the RTX 5000 Ada.
Compile + first call took 11.1 s forward and 14.1 s for value + gradient.
Raw measurements and accuracy metrics are in `docs/mixed_precision_results.json`.

**Accuracy versus float64:** across three H2O scales (exp(-0.7), 1,
exp(0.7)), maximum absolute flux difference was 3.53e-8. Maximum absolute
objective-gradient difference was 7.53e-7; maximum relative difference among
nonzero gradient components was 3.17e-6. These checks cover this benchmark
profile and wavelength interval; they do not establish a universal error
bound or fitted-parameter recovery accuracy.

The subsequent [extended validation](mixed_precision_validation.md) checks
six molecules, 588 molecular states, 45 combined spectra, and three paired
fits. Its machine-readable report records the measured errors and acceptance
criteria separately from these timing measurements.
All checks passed after improving the fitter's scaling and convergence guard;
the largest mixed-versus-float64 fitted-parameter shift was 1.09e-4 of its
local uncertainty. See the validation report for the tested scope.

### Precision boundaries

- Float64: spectral grid, line-center subtraction, line strengths, Doppler
  and Lorentz widths, partition functions, line cores, opacity accumulation,
  transmission, and the entire instrument model.
- Float32: the asymptotic wing value and its derivative coefficients only.
- Above the sparse calculator's 400 K threshold: original float64 Direct.

The important detail is to subtract nearby line centers and spectral samples
**before** converting to float32. Globally disabling JAX float64 would lose
precision in these coordinates and is not the tested configuration.

A first experiment retaining the original float64 wing derivatives measured
5.5 ms forward but 67.1 ms for value + gradient. Computing those derivatives
naively in float32 is unsafe: their original formula subtracts nearly equal
terms in distant wings. We instead simplify ExoJAX's derivative identity
algebraically. For `z=x+i*a`, `u=1/z**2`, define
`d=(u+1.5*u**2+3.75*u**3)/sqrt(pi)`. The same truncated-series identity gives
`dH/dx=imag(d)` and `dH/da=real(d)` without the catastrophic subtraction.
Its coefficients can then be evaluated in float32, with tangent propagation
and accumulation in float64. This retains ExoJAX's analytic-JVP convention.

Tests cover opacity and first derivatives in temperature, total pressure,
and self pressure over 180--650 K, layer batching, and far-wing derivative
cancellation. Mixed-mode test tolerances are 1e-6 relative opacity and 1e-5
relative derivatives, reflecting the measured few-ppm derivative changes.

```python
backend = ExoJAXOpacityBackend.prepare(
    databases, nu_grid, methods="direct_sparse",
    vectorize_layers=True, mixed_precision=True,
)
```

To reproduce, run the full-order benchmark with and without `--mixed-precision`:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/benchmark.py --platform gpu \
  --method direct_sparse --vectorize-layers --mixed-precision \
  --v2 5100 --pixels 2048 --iterations 30 --output benchmarks/results/mixed.json
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/benchmark.py --platform gpu \
  --method direct_sparse --vectorize-layers \
  --v2 5100 --pixels 2048 --iterations 30 --output benchmarks/results/float64.json
.venv/bin/python benchmarks/compare.py benchmarks/results/float64.json \
  benchmarks/results/mixed.json --rtol 1e-5 --atol 1e-7
```

## Optimized Direct path (2026-09-15)

For the same 20 cm-1 workload below, use `methods="direct_sparse"` and
`vectorize_layers=True` in `ExoJAXOpacityBackend.prepare`. Original Direct
remains available as the reference implementation and the default.

| GPU operation | Original Direct | Sparse core, unrolled layers | Sparse core, batched layers |
|---|---:|---:|---:|
| Forward median | 109.9 ms | 15.4 ms | 15.6 ms |
| Objective + gradient median | 183.7 ms | 20.5 ms | 17.7 ms |
| Forward compile + first | 30.1 s | 17.0 s | 5.7 s |
| Gradient compile + first | 45.1 s | 24.6 s | 6.9 s |

The final GPU comparison gives **7.0x faster forward evaluation and 10.4x
faster value + gradient**. At three H2O column scales (exp(-0.7), 1,
exp(0.7)), maximum absolute flux and gradient differences versus original
ExoJAX are 3.33e-16 and 7.11e-14, respectively. Machine-readable measurements
are saved in `docs/performance_results.json`.

The wider **5000--5100 cm-1, 2048-pixel order** also completed: 4,669 lines,
5,348 high-resolution samples, six layers, 61.4 ms forward and 73.2 ms
value + gradient on GPU (20 timed calls). Compile + first call took 11.6 s
and 14.5 s. Only 35,862 of its 24,969,812 line/grid pairs (0.144%) can enter
the expensive core branch. The wider case is a scaling measurement; its
full-order agreement with original Direct was not separately measured.

The optimized CPU path with unrolled layers measured 118 ms forward and
409 ms for objective + gradient, versus the original 1,168 and 3,405 ms.
The timings are workload-specific; CPU and GPU runs can be affected by other
work on this shared machine.

### What changed

ExoJAX's `hjert` selects Algorithm 916 for `x*x + a*a < 111` and an
asymptotic expression otherwise. Under vectorization, its `where` computes
both expressions before selecting. Algorithm 916 sums 27 terms per
line/grid pair, even though most pairs lie in distant wings.

`SparseCoreDirect` preselects every pair that could enter the core branch at
temperatures up to 400 K. Only those pairs use the original `hjert`. All
other pairs use ExoJAX's original asymptotic expression and analytic JVP.
Every line and wing remains included. A larger Lorentz width cannot create
an omitted core pair because its square only increases the branch test.
Above 400 K the original Direct implementation is used automatically.

The second improvement replaces our expanded Python layer loop with `vmap`.
This substantially reduces compilation time for a fixed terrestrial profile.
Dynamic temperatures spanning the fallback boundary may evaluate both branches
under `vmap`; use unrolled layers if that workload matters.

No ExoJAX installation files were changed. Tests compare the actual ExoJAX
values and temperature, total-pressure, and self-pressure derivatives at
180--650 K, including the fallback, and verify the public batched adapter.

### Reproduce and compare

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/benchmark.py --platform gpu \
  --method direct --iterations 20 --output benchmarks/results/reference.json
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/benchmark.py --platform gpu \
  --method direct_sparse --vectorize-layers --iterations 20 \
  --output benchmarks/results/optimized.json
.venv/bin/python benchmarks/compare.py benchmarks/results/reference.json \
  benchmarks/results/optimized.json
```

The comparison checks spectra and gradients at three H2O abundance scales and
requires at least a 1.5x steady-state speedup. Timings synchronize the entire
output tree, including both objective and gradient. The objective is the
original benchmark's mean-squared residual; it is not an end-to-end optimizer
timing and does not exercise fitted jitter. All calculations remain float64.

Use `--v2 5100 --pixels 2048` to measure a wider order. Initialization of the
line database and core index list is outside compile/execute timings.

## Original implementation measurements

Measured on 2026-09-15 with JAX/JAXlib 0.6.2 in 64-bit mode.

## Workload

The benchmark evaluates a water-dominated 5000--5020 cm-1 sub-order using:

- 1,938 AER 3.9 H2O lines, including a 25 cm-1 margin on both sides
- six atmospheric layers with H2O self pressure
- 2,517 high-resolution samples
- a 30 degree zenith angle
- Gaussian LSF convolution and integration onto 512 detector pixels
- ExoJAX `OpaDirect`

Each steady-state number is the median of synchronized calls after compilation
and two warmups. The CPU used seven measured calls and the GPU used twenty.
`compile + first` includes XLA compilation and the first synchronized execution.

## Hardware

- CPU: two Intel Xeon Gold 6526Y sockets, 32 physical cores and 64 threads
- GPU: NVIDIA RTX 5000 Ada Generation, 32 GB, driver 580.82.07
- GPU selection: physical GPU 1, idle before the run

## Results

| Operation | CPU median | GPU median | GPU speedup | CPU compile + first | GPU compile + first |
|---|---:|---:|---:|---:|---:|
| Forward model | 1.168 s | 109.5 ms | 10.66x | 9.27 s | 29.08 s |
| Objective + gradient | 3.405 s | 183.3 ms | 18.58x | 21.52 s | 43.67 s |

The corresponding steady-state rates are 0.86 versus 9.13 forward models per
second and 0.29 versus 5.46 value-and-gradient evaluations per second.

The GPU's extra compilation cost is recovered after about 19 forward calls or
7 gradient calls for this fixed model shape. A fit normally performs many more
gradient calls, so the GPU is the effective target for interactive fitting.
CPU execution remains useful for tests and short validation runs.

For context, LBLRTM 12.17 took a median 266 ms over ten process-level runs for
the same spectral interval, six-level H2O profile, and continuum-disabled path.
That is not a strict kernel comparison: LBLRTM produces its own denser
monochromatic grid and does not compute autodiff gradients or this package's
detector model. It shows that the current Direct CPU path is slower than an
LBLRTM reference call, while the warmed GPU forward path is about 2.4 times
faster and also supports gradients.

## Interpretation

The main cost is direct Voigt evaluation across every line, layer, and spectral
sample. Runtime and compilation grow quickly with interval width. Keep a
prepared model alive and reuse fixed shapes across optimizer iterations; doing
otherwise pays the compilation cost repeatedly. Splitting an IGRINS order into
fixed padded segments can also cap compile time and memory use.

PreMODIT should improve throughput for trace gases, but ExoJAX 2.5 does not
accept H2O self pressure through its public PreMODIT interface. The benchmark
therefore measures the scientifically safer Direct H2O path used by this
project.

## Reproduction

```bash
UV_CACHE_DIR=.uv-cache uv sync --dev --extra gpu
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py \
  --platform cpu --iterations 7 --output benchmarks/results/cpu.json
CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py \
  --platform gpu --iterations 20 --output benchmarks/results/gpu.json
```

GPU numbering inside JAX is relative to `CUDA_VISIBLE_DEVICES`; the selected
physical GPU therefore appears as `cuda:0` in the recorded result.
