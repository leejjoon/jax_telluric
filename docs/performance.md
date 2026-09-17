# Performance benchmark

## Per-page cost of a batch fit (2026-09-17)

With the opacity out of the optimizer loop, what a page costs is no longer the
line-by-line kernel. Measured per phase by the batch driver on `ab3255_`
(3,067 samples, 4 species, 9,139 lines, 1,301 pixels), the numbers below are
from `arcturus_batch_timing.json`, which the driver writes on every run.

| Phase | Before | After |
|---|---:|---:|
| Read AER line files | 4.0 s | 0.45 s |
| Build sparse core lists | 2.1 s | 0.8 s |
| Precompute opacity | 20.4 s | 7.1 s cold, 2.3 s cached |
| Compile objective | 3.7 s | 3.4 s |
| Fit four stages | 12.1 s | 6.4 s |
| Write products | 37.1 s | 0.1 s |
| **Total** | **~57 s** | **~13 s** |

The atlas is 598 page-epochs, so this projects to about 45 minutes across three
GPUs, from roughly three hours. Five changes got there.

Fitted values reproduce the committed atlas results: on `ab3255_` and `ab4950_`
residual rms over noise moves by at most 0.003 and the water column by 1e-4
relative. `ab2021_` moves more -- water column 0.23% -- because it is the
narrowest page in the atlas and its grid shrinks 3.9x, so the trimming below
bites hardest there; its residual rms improves slightly, from 6.5276 to 6.5261.

### The grid margin is not the line margin

`igrins_wavenumber_grid` pads by 25 cm-1 so that lines outside a window still
contribute their wings. That is a property of the *line list*. The grid only has
to cover the window plus what the forward model reaches back for -- the LSF
kernel, the Doppler shifts, and the instrument profile's edge padding -- which
is under 2 cm-1 anywhere in this atlas. Sharing one margin between the two put
**70% of every grid where there is no data**, and 90% on the narrow 2 micron
pages, which are 4 cm-1 wide with 50 cm-1 of padding.

`trim_wavenumber_grid` cuts the grid back to a separate, smaller margin. It
trims rather than regenerating, so the samples keep their original spacing *and*
phase and only the extent changes. Across the atlas this is **3.3x fewer grid
points**, up to 9x on the narrow pages, and it is what stopped `ab2021_` from
exhausting device memory. Measured against an untrimmed grid at a 2 cm-1 margin:
maximum pixel-flux difference 2.5e-4, confined to the outermost pixels, against
5.5e-3 of photon noise. The drivers default to 5 cm-1.

### Layer chunking is obsolete

`layer_chunk_size` existed because the wing array is dense in lines by grid by
layer. With ExoJAX's offset matrix gone, XLA fuses the whole wing sum into one
reduction and never materializes it, so peak memory is the same at every chunk
size -- while chunking multiplies compile time:

| Layer chunk | Compile | Run | Peak memory |
|---|---:|---:|---:|
| 2 | 40.8 s | 23.3 ms | 1.1 GB |
| 3 | 24.0 s | 23.9 ms | 1.1 GB |
| 6 | 11.9 s | 24.6 ms | 1.1 GB |
| none | 8.0 s | 24.5 ms | 1.1 GB |

Both drivers now default to no chunking. The option remains for a grid fine
enough to break that fusion.

### The core list by binary search

The core/wing split was built with `np.nonzero` over a dense (line, grid)
boolean -- the last place an array of that size was built, at 2.1 s and a
gigabyte on a wide page. Both arrays are sorted, so each line's core region is
one contiguous run of samples, found by `searchsorted`. A pair at the boundary
has `x*x >= 111`, where ExoJAX's `hjert` already takes its asymptotic branch, so
which side it falls on cannot change a value; the core list comes out identical,
order included, and `tests/test_direct.py` asserts that against the dense scan.

### Final products without an eager forward model

Writing a page's products re-evaluated the forward model without the
precomputation, which ran the uncompiled kernel operation by operation: 37 s,
more than the rest of the page together. Refreezing at the *fitted* parameters
is exact there -- the expansion's offset from its own reference is zero, so it
returns the kernel's own values -- and reuses the cached compilation. Same
numbers, 0.2 s.

### What did not pay

**Padding shapes into buckets so pages share a compilation.** Measured across
the 598 page-epochs, the compile key is the product of four dimensions --
species set, grid size, pixel count, and per-species line count -- and it
shatters: at a 1.25x bucket ratio only 8% of pages reuse a compilation, and even
a brutal 2x ratio (43% padding waste) reaches 36%. With the opacity precomputed,
the *objective's* key is only (grid, pixels), where 1.125x buckets give 18
distinct shapes and 97% reuse -- but that compile is 3.7 s of a 21 s page, and
collecting it would mean carrying every page-dependent array through `jit` as a
traced argument.

**A line-strength budget.** `select_significant_lines` discards the weakest
lines whose bounds sum to a stated optical depth, which is a guarantee rather
than a heuristic: a Voigt peak cannot exceed either component's peak, so the
summed bound over the discarded set bounds the transmission error. Realized
error is some 400x below the guarantee.

| Budget | Lines kept | Max error | Kernel run |
|---|---:|---:|---:|
| 1e-4 | 64% | 4.1e-7 | |
| 1e-3 | 43% | 4.0e-6 | 39.7 -> 24.4 ms |
| 1e-2 | 22% | 5.5e-5 | 39.7 -> 15.4 ms |

It is off by default. Once the opacity is precomputed the kernel runs three
times per page, so halving it saves about 50 ms; what it still buys is memory
headroom and a real speedup for `--no-precompute-opacity`.

**A persistent compilation cache.** `--compilation-cache` is on by default and
cuts the opacity precompute from 7.7 s to 2.3 s on a repeated run. It does not
help the objective: its graph embeds the precomputed cross sections as
constants, and a GPU scatter-add is not bit-reproducible, so the key changes
between runs.

### Reading the line files once, not once per page

An AER per-molecule file holds up to a million records and a page needs a few
thousand, but the file carries no index, so every read had to look at every
record in Python: 4 s per page, and the largest remaining item.

`AERLineDatabase` now builds one index per file -- offset, molecule id and
wavenumber of every line that could be a record -- in numpy, memory maps the
file, and caches both under the file's identity and modification time. The index
is a *superset*: anything it cannot classify with certainty is marked uncertain
and still offered to the ordinary parser, which makes every accept and reject
decision exactly as before. Verified byte-identical against a plain line-by-line
scan across six molecules and eight windows, and asserted in the tests against a
scan of a synthetic file carrying a header, a delimiter and a short line.

| | Line-by-line | Indexed, first read | Indexed, later reads |
|---|---:|---:|---:|
| H2O, 20 cm-1 window | 0.57 s | 0.66 s | 0.05 s |
| CO2, 20 cm-1 window | 1.09 s | 0.49 s | 0.05 s |

Across a batch that is 4 s per page down to 0.45 s. One detail cost an hour and
is worth recording: AER right-justifies the molecule field, so the common form
is `" 2"`, not `"02"`. A first version only read two-digit fields, classified
every line as uncertain, rejected nothing, and came out *slower* than the scan
it replaced. Files with carriage returns fall back to the line-by-line path,
because the byte offsets would not survive universal-newline translation.

The next measured candidate is compiling the objective, 3.4 s of every page.
Sharing it across pages needs the shape work described above and a way to carry
each page's arrays through `jit` as traced arguments.

## Fitting cost (2026-09-17)

`benchmark.py` times one forward evaluation. A fit spends its time differently:
XLA compilation happens once per distinct jitted function and costs seconds,
while one evaluation costs milliseconds. Three changes address that, measured on
the Arcturus workload (5005--5025 cm-1, 8,474 AER lines, 12 layers, 5,585
high-resolution samples, 1,001 pixels, `mt_ckd`, mixed precision, pressure
shifts, RTX 5000 Ada). Raw numbers are in `precomputed_opacity_results.json`.

### The dense offset matrix is gone

ExoJAX's `OpaDirect` stores `nu_grid[None, :] - nu_lines[:, None]` as a dense
(line, grid) float64 matrix, and `SparseCoreDirect` added a boolean mask of the
same shape. Both are exact functions of two 1-D vectors totalling 113 KB, so
they are now recomputed inside the kernel and XLA fuses them into the reduction.

| | Before | After |
|---|---:|---:|
| Held on the device | 426 MB | 1.78 MB |
| Opacity compile + first call | 22.4 s | 4.3 s |
| Opacity steady state | 16.7 ms | 16.7 ms |

Steady state does not move: the kernel is compute-bound, not bandwidth-bound.
What changes is compile time, and the memory ceiling that previously forced
`layer_chunk_size` on fine grids. Agreement with the previous implementation is
2e-15 relative across values *and* gradients, over both epochs' temperature and
pressure ranges, both molecules, and all four combinations of `pressure_shift`
and `mixed_precision` -- round-off, not approximation.

### The opacity leaves the optimizer loop

A fitted parameter reaches `xsvector` only through the self-broadening partial
pressure. `TelluricModel.precompute_opacity()` evaluates the kernel once and
carries the result as fixed arrays expanded to first order in that pressure, so
the column scales stay free through the linear `tau = sigma * column` factor and
through the continuum.

| GPU operation | Live opacity | Precomputed |
|---|---:|---:|
| Value and gradient | 38.3 ms | 2.53 ms |
| Compile + first call | 8.92 s | 2.36 s |

Accuracy, against the exact calculator over water columns from 0.50x to 2.72x
the reference: maximum transmission error 6.7e-4, rms 2.9e-5, against 5.5e-3 of
photon noise. `self_broadening="frozen"` drops the first-order term and is 70x
worse (1.5e-2 maximum); it is accurate only near the reference and a fit using
it must be repeated from its own result.

### One compilation per page, not one per stage

`fit_order` built a fresh `jax.jit` closure on every call, and a new Python
function object is a new cache entry, so each stage of a staged fit recompiled
the same graph. `OrderObjective` compiles over the *full* parameter vector and
leaves the bounds, the free set, and the optimizer's rescaling outside -- which
is exactly what changes between stages. Pass one to every stage.

### Together

Three stages (continuum, velocity, columns) on the synthetic page above:

| | Wall clock | Speedup |
|---|---:|---:|
| Compile per stage (previous behaviour) | 61.2 s | 1.0x |
| One shared compilation | 35.4 s | 1.7x |
| Precomputed opacity, shared compilation | 3.3 s | 18.4x |

End to end, including the 4.2 s precompute, **8.1x**. Fitted parameters agree
with the live-opacity fit to 1.8e-5 relative and the model flux to 2.3e-6.

On the real driver, `scripts/fit_arcturus_page.py` over `ab5000_` summer with
the Payne-Zero source: 80.7 s to 11.4 s, **7.1x**, with identical stage
objectives to six significant figures and identical residual rms to eight. All
fitted parameters agree to 8e-6 relative except `lsf_sigma_kms`, which differs
by 2.2% -- that parameter sits in a flat valley behind the measured FTS sinc and
must not be interpreted (see `arcturus_fit.md`); the objective and the flux do
not move with it.

Both drivers default to `--precompute-opacity`. Every saved product is
re-evaluated through the exact model, so no output carries the approximation.

```bash
CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark_fit.py
```

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

The optional [LBLRTM-corrected mode](lblrtm_corrected_mode.md) adds
precomputed MT_CKD and empirical line residuals. Adding those arrays has
negligible cost relative to opacity evaluation. Its pressure-shifted opacity
baseline has a separate, measured cost because line coordinates become
layer-dependent:

| Platform and operation | Unshifted | Pressure shifted | Ratio |
|---|---:|---:|---:|
| CPU forward | 100.9 ms | 96.6 ms | 0.96x |
| CPU objective + gradient | 148.4 ms | 153.1 ms | 1.03x |
| GPU forward | 1.89 ms | 3.88 ms | 2.06x |
| GPU objective + gradient | 3.31 ms | 3.56 ms | 1.08x |

These paired runs use the fixed profile's actual temperature and pressure
bounds, 10 CPU iterations, and 30 GPU iterations. The CPU forward difference
is benchmark noise. On the GPU, dynamic line coordinates add about 2 ms to a
forward call; the objective-gradient cost rises by about 0.25 ms.

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
