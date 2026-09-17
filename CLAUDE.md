# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`jax-telluric` (package `src/jax_telluric/`, repo directory `lblrtm/`) is a
differentiable JAX forward model for terrestrial (telluric) absorption in
high-resolution spectra such as IGRINS. It wraps ExoJAX opacity calculators
with a layered atmosphere, slant-path transmission, an instrument model, and a
bounded MAP fitter. LBLRTM 12.17 is the accuracy reference, not a runtime
dependency — LBLRTM is only executed offline to produce fixtures and
correction templates.

## Commands

```bash
UV_CACHE_DIR=.uv-cache uv sync --dev          # environment (.venv, Python 3.10)
UV_CACHE_DIR=.uv-cache uv run pytest          # full suite, ~70 s, no external data needed
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py::test_name   # one test
```

Always prefix with `UV_CACHE_DIR=.uv-cache`; the default cache location is not
used in this project. Set `JAX_PLATFORMS=cpu` to keep test/benchmark runs off
the GPU.

The Arcturus telluric-fitting pipeline (see `docs/arcturus_fit.md`). Needs the AER
line files and the MT_CKD file from `bootstrap_lblrtm.sh`, but not the LBLRTM binary:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/measure_atlas_ils.py          # FTS MOPD per page/epoch
UV_CACHE_DIR=.uv-cache uv run python scripts/make_site_profile.py          # site + epoch atmosphere CSV
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_arcturus_fit.py      # synthetic checks L0-L2
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_arcturus_page.py --stellar <npz>
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_arcturus_batch.py         # resumable, whole atlas
```

Both fit drivers default to `--precompute-opacity` (see *What a fit costs*);
`--no-precompute-opacity` restores the exact-kernel-per-iteration path and is
the control to reach for when a fitted value looks wrong.

`scripts/generate_payne_zero_arcturus.py` makes the stellar source and does **not**
run in this environment: Payne Zero needs Python >= 3.11 while this package is pinned
to 3.10 by `exojax==2.5.0`. It runs in Payne Zero's own venv and writes an npz that
`StellarSpectrum.from_npz` reads back; the script's docstring has the invocation.

Benchmarks and the offline LBLRTM workflows. All need `./scripts/bootstrap_lblrtm.sh`
to have been run first; `benchmark.py` and `validate_mixed_precision.py` use only the
AER line files it downloads, the rest also execute the LBLRTM binary:

```bash
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark.py --platform cpu --output benchmarks/results/cpu.json
UV_CACHE_DIR=.uv-cache uv run python benchmarks/benchmark_fit.py   # staged-fit cost -> docs/precomputed_opacity_results.json
UV_CACHE_DIR=.uv-cache uv run python benchmarks/compare.py baseline.json candidate.json
UV_CACHE_DIR=.uv-cache uv run python scripts/run_lblrtm_reference.py      # rebuild tests/data fixture
UV_CACHE_DIR=.uv-cache uv run python scripts/build_lblrtm_correction.py   # rebuild data/corrections template
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_aer_co.py           # writes tests/data/aer_co_validation.json
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_mt_ckd.py           # writes docs/native_mt_ckd_validation.json
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_mixed_precision.py [--resume]
```

## Two toolchains

- `.venv/` — the `uv` Python environment for the package.
- `.conda-lblrtm/` — a conda-forge gfortran/netCDF-Fortran toolchain used only
  to build LBLRTM/LNFL. Never mix the two. `scripts/bootstrap_lblrtm.sh` creates
  it, clones and builds LBLRTM 12.17 + LNFL, downloads AER Line File 3.9, and
  runs LNFL to make `data/lblrtm/run_lnfl_igrins/TAPE3`.

Scripts hardcode the bootstrapped paths relative to the repo root:
`data/lblrtm/LBLRTM/lblrtm_v12.17_linux_gnu_sgl`,
`data/lblrtm/run_lnfl_igrins/TAPE3`,
`data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc`,
`data/lblrtm/AER_Line_File/aer_v_3.9/line_files_By_Molecule/`.
All of `data/lblrtm/`, `data/databases/`, `data/corrections/`, and
`benchmarks/results/*` are gitignored — only compact fixtures under
`tests/data/` and JSON reports under `docs/` are committed.

For installed users, `jax-telluric-download-data {mt-ckd,aer-lines,all}`
(`download.py`) fetches SHA-256-pinned copies into `$JAX_TELLURIC_DATA` or
`~/.local/share/jax-telluric`.

## Architecture

The forward model is composed from three duck-typed seams declared as
`Protocol`s in `model.py` — `OpacityBackend`, `ContinuumBackend`, and
`CorrectionBackend`. Anything satisfying them can be swapped in, which is how
tests run without spectroscopic data.

```
AtmosphereProfile (types.py)      layered, top-to-bottom, validated in __post_init__
        +
OpacityBackend                    ExoJAXOpacityBackend (exojax_backend.py) -> OpaDirect /
                                  SparseCoreDirect (direct.py) / OpaPremodit;
                                  ArrayOpacityBackend for fixtures
        + optional ContinuumBackend  MTCKDWaterContinuum (mt_ckd.py), ReferenceWaterContinuum
        + optional CorrectionBackend LBLRTMOpticalDepthCorrection (corrections.py)
        ->
TelluricModel (model.py)          .transmission()  = exp(-sum_layers tau / cos z)
                                  .predict(order, params) adds velocity/stretch, LSF
                                  convolution, Simpson pixel integration, Chebyshev
                                  log-continuum
        ->
fit_order (fit.py)                L-BFGS-B over jax.value_and_grad, bounded and rescaled
```

### What a fit costs

XLA compilation happens once per distinct jitted function and costs seconds,
while one evaluation costs milliseconds, so a staged fit is dominated by how
many times it compiles. Two seams exist for that, and both drivers use them:

- `OrderObjective(model, order, continuum_size)` compiles once over the *full*
  parameter vector and is passed to `fit_order(..., objective=...)` for every
  stage. Bounds, the free set, and the rescaling stay outside the compiled
  function, which is why they may change between calls. Without it `fit_order`
  builds a fresh `jax.jit` closure per call and recompiles the same graph.
- `TelluricModel.precompute_opacity()` evaluates the line-by-line kernel once
  and carries it as fixed arrays, expanded to first order in the
  self-broadening partial pressure (`LinearizedOpacityBackend`) -- the only
  route a fitted parameter takes into the kernel, and a weak one. Column scales
  stay free and differentiable. `self_broadening="frozen"` drops that term and
  is 70x less accurate; a fit using it must be repeated from its own result.

Measured together: 61 s to 3.3 s for a three-stage fit, 8.1x end to end
including the precompute, with fitted parameters agreeing to 1.8e-5.

Two traps follow from this. **Never call the forward model eagerly on a live
opacity backend** — an uncompiled kernel dispatches operation by operation and
one call costs more than the whole fit (measured: 37 s of a 57 s page). For
final products, call `precompute_opacity(fitted_parameters)` instead: refreezing
at those parameters is *exact* there, because the expansion's offset from its
own reference is zero, and it reuses the evaluator cached on the model. And
**`velocity_step_kms` is the same for every page** at a fixed resolving power
and sampling, which is what makes one grid spacing shared across the atlas.

### Two margins, not one

`igrins_wavenumber_grid(..., margin_cm1=25)` pads so that outside lines
contribute their wings — a property of the *line list*. The grid only has to
cover the window plus what the model reaches back for (LSF kernel, Doppler
shifts, instrument edge padding), under 2 cm-1 anywhere in this atlas. Sharing
one margin put 70% of every grid where there is no data, and 90% on the narrow
2 micron pages. Select lines over `(v1, v2)` with the wide margin, then cut the
grid with `trim_wavenumber_grid(grid, v1, v2, 5.0)`, which trims rather than
regenerating so the samples keep their spacing *and* phase. 3.3x fewer grid
points atlas-wide; maximum pixel-flux difference 2.5e-4 against 5.5e-3 noise.

`select_significant_lines(database, profile, species, budget)` is available and
off by default: it discards the weakest lines whose bounded contributions sum to
`budget` of optical depth, which bounds the transmission error by the same
amount. It buys little once the opacity is precomputed — see
`docs/performance.md` for when it does.

Observed data and the stellar source enter through two adapters: `atlas.py`
(`read_arcturus_page` / `arcturus_spectral_order`, a self-contained reader for the
Hinkle, Wallace & Livingston 1995 IR atlas) and `stellar.py` (`StellarSpectrum`,
`prepare_stellar_source` — resample onto the model grid, broaden, normalize).

A source spectrum may be given on the pixel grid (`SpectralOrder.source_flux`) or on
the model's high-resolution grid (`source_flux_model_grid`); only the latter can
represent structure narrower than a pixel, and only it activates the separate
`TelluricParameters.stellar_velocity_kms`. `InstrumentProfile` is a fourth seam:
the default is the built-in Gaussian, and `BoxcarFTSInstrumentProfile` applies an
unapodized FTS sinc by FFT (a truncated kernel is invalid for a sinc at any width).
`pixel_integration="point"` turns off Simpson pixel averaging, which is wrong for
point-sampled FTS data.

Line data reaches ExoJAX through `AERLineDatabase` (`aer.py`), a minimal
HITRAN-shaped adapter over AER 100-character per-molecule line files (it
exposes `nu_lines`, `logsij0`, `gamma_air/self`, `n_air`, `delta_air`,
`qr_interp`, `dbtype="hitran"`). The files carry no index, so it builds one per
file in numpy — offset, molecule id and wavenumber of every possible record —
memory maps the file and caches both by path, size and mtime. That index is a
**superset**: anything it cannot classify with certainty stays a candidate and
the ordinary per-line parser still makes every accept/reject decision, which is
what keeps it byte-identical. Keep that property if you touch it. Note AER
right-justifies the molecule field (`" 2"`, not `"02"`); reading only two-digit
forms silently matches nothing and is *slower* than no index at all. LBLRTM inputs/outputs live in `lblrtm.py`
(TAPE5 writer, subprocess runner) and `reference.py` (binary TAPE12 reader for
the GNU single-precision build, resolution degradation, `compare_transmission`
metrics). `io.py` loads CSV layer profiles and TelFit/RFM MIPAS level profiles.

### accuracy_mode

`TelluricModel(..., accuracy_mode=...)` selects the physics, and the
constructor enforces a strict combination matrix:

- `"fast"` (default) — lines only, plus an explicitly supplied `continuum`;
  a `correction` is rejected.
- `"mt_ckd"` — exactly one of `continuum` (runtime `MTCKDWaterContinuum`,
  differentiable across pressure/temperature/abundance) or `correction`
  (only its isolated H2O self/foreign terms via `mt_ckd_optical_depth`).
- `"lblrtm_corrected"` — requires a `correction` and forbids `continuum`;
  adds the continua plus per-species empirical line residuals and the fixed
  reference background. Valid only for the exact profile and grid the template
  was built for; `validate()` re-checks the profile and
  `validate_opacity()` refuses a backend without `pressure_shift=True` when the
  template needs it.

### Performance path

`SparseCoreDirect` subclasses `OpaDirect` and statically splits line/grid pairs
into a compact core list, keeping ExoJAX's exact formulas and branch selection.
Outside its configured temperature (default 150–400 K) or pressure bounds it
falls back to full Direct via `jax.lax.cond`.
Nothing of shape (line, grid) is stored: ExoJAX's dense offset matrix and the
core/wing test are exact functions of two 1-D vectors and are recomputed inside
the kernel, which cut what the calculators hold from 426 MB to 1.8 MB and the
opacity compile from 22 s to 4 s on the Arcturus window, bit-for-bit. Keep it
that way — `tests/test_direct.py` asserts no 2-D array survives construction.
The kernel is compute-bound, not bandwidth-bound, so steady state is unchanged.
XLA now fuses the whole wing sum into one reduction and never materializes it,
so **`layer_chunk_size` is obsolete**: peak memory is the same at every chunk
size while chunking multiplies compile time (40.8 s at chunk 2 against 8.0 s
unsplit). Both drivers default to no chunking; the option remains for a grid
fine enough to break that fusion. Do not reach for `vectorize_layers=False`
either — a Python layer loop unrolls the whole calculation per layer (24 s
chunked against ~400 s looped). A JVP through `xsvector` is worse than both:
`vmap` turns the out-of-range `lax.cond` into a select that evaluates the dense
fallback branch, which is why `precompute_opacity` uses a central difference.
`mixed_precision=True` evaluates only the asymptotic wing value and its
derivative in float32 — the JVP is algebraically rewritten to avoid the
catastrophic float32 cancellation in ExoJAX's identity. Coordinates, cores,
line physics, accumulation, and the whole instrument model stay float64.
`pressure_shift=True` adds HITRAN `delta_air` shifts scaled like LBLRTM's
RHORAT (`delta_air * P/1 atm * 296 K/T`); ExoJAX 2.5 omits these, so it is
opt-in to keep the default numerically identical to ExoJAX.

## Conventions

- `jax_telluric/__init__.py` enables `jax_enable_x64` at import. Import
  `jax_telluric` **before** ExoJAX so calculators are built in float64.
- Public data classes are frozen dataclasses/NamedTuples that validate and
  normalize in `__post_init__` (species names uppercased, arrays coerced,
  physical ordering checked) and raise `ValueError` with a short message.
  Follow that pattern rather than validating at call sites.
- Atmosphere arrays are ordered top-to-bottom: pressure edges increase,
  altitude decreases. Wavenumber grids are strictly increasing and evenly
  spaced in log wavenumber (constant velocity step) — `igrins_wavenumber_grid`
  builds them with padding for line wings.
- Units are in names: `_bar`, `_hpa`, `_cm1`, `_km`, `_kms`, `_k`, `cm2`.
- Comments explain *why* (a numerical or LBLRTM-compatibility reason), not what.
  Keep that density; the existing comments are load-bearing.
- Accuracy and performance claims are backed by a committed JSON report under
  `docs/` plus a prose companion (`docs/validation.md`,
  `docs/performance.md`, `docs/lblrtm_corrected_mode.md`,
  `docs/mixed_precision_validation.md`). `tests/test_lblrtm.py` asserts on
  `tests/data/aer_co_validation.json` and `docs/native_mt_ckd_validation.json`,
  so regenerating those reports requires a working LBLRTM build. Update the
  numbers in README/docs prose whenever a report is regenerated.
- `mt_ckd.py` carries an AER copyright notice; keep it with any derived code.
- `FitResult.covariance` comes from L-BFGS-B's inverse Hessian and is **not** a
  covariance: on the Arcturus fit it overestimates the column errors by two orders
  of magnitude. Never quote it as an uncertainty; use an explicit study instead.
- Commit subjects are short imperative lines with no body (`Add native JAX
  MT_CKD continuum`).
