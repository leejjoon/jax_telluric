# Extended mixed-precision validation

This validates the additional numerical error from mixed precision, separately
from physical differences between ExoJAX and LBLRTM. The machine-readable
report is [mixed_precision_validation.json](mixed_precision_validation.json).
Only a report with `passed: true` represents a completed successful run.

## Results (RTX 5000 Ada, JAX 0.6.2)

The opacity, spectrum, and paired-fit checks all passed after hardening
`fit_order` against premature L-BFGS-B convergence.

| Check | Worst measured error | Limit | Result |
|---|---:|---:|---|
| 588 molecular states: opacity relative L2 | 4.94e-8 | 1e-6 | Pass |
| Temperature/pressure derivative relative L2 | 8.75e-8 | 1e-4 | Pass |
| 45 combined spectra: absolute flux | 2.73e-8 | 1e-6 | Pass |
| Spectral Jacobian column relative L2 | 7.23e-8 | 1e-4 | Pass |
| Three paired fits: largest parameter shift | 1.09e-4 sigma | 0.01 sigma | Pass |

The original fitter represented fixed parameters as optimizer dimensions,
used unscaled physical variables, and retained a large parameter-independent
likelihood offset. L-BFGS-B stopped on relative objective reduction while the
mixed solution still had a large gradient, producing a 10.88-sigma LSF-width
difference for seed 17.

The fitter now removes exactly fixed dimensions, scales free variables to
useful physical steps, subtracts the constant likelihood normalization during
optimization, and rejects convergence with a large projected scaled gradient.
It still reports the conventional Gaussian log-likelihood value and returns a
full-size covariance matrix in physical units.

| Noise seed | Largest parameter shift / local uncertainty | Result |
|---|---:|---|
| 17 | 2.40e-5 | Pass |
| 42 | 5.75e-5 | Pass |
| 123 | 1.09e-4 | Pass |

All six optimizations reported success, and every recovered parameter was
within 2.59 local uncertainty units of truth. The report now has `passed: true`.
The CPU regression suite passed all 23 tests, including a high-S/N regression
for the former premature-convergence pattern.

## Checks and acceptance criteria

1. **Individual molecular opacity versus original ExoJAX Direct.** Seven
   real AER 3.9 windows: H2O at 6250 and 5000 cm-1, CO2 at 6350,
   CH4 at 6000, CO at 4300, N2O at 4500, and O2 at 6300; each is 10 cm-1
   wide with 25 cm-1 line margins. Each window uses 84 Cartesian combinations:
   temperatures 180, 240, 300, 399.9, 400, 400.1, and 450 K; pressures
   1e-5, 0.01, 0.1, and 1.1 bar; self-pressure fractions 1e-5, 0.001, and
   0.03. High self fractions for trace gases are numerical stress tests.
   Compare cross sections and full spectral derivatives in T, P, and Pself.
   Limits: 1e-6 relative L2 opacity error and 1e-4 derivative error.
2. **Combined terrestrial spectra versus sparse Direct float64.** Six-layer
   example profile, H/K windows 6250--6270, 5000--5020, and 4300--4320 cm-1,
   512 pixels, Gaussian instrument broadening, wavelength shift/stretch, and
   continuum throughput. Airmasses 1, 1.5, and 2.5; water multipliers 0.05,
   0.5, 1, 2, and 5. Include the six species above wherever their AER files
   contain lines in the padded window; absent species are recorded explicitly.
   Compare flux and spectral Jacobians for six abundances, velocity, stretch,
   width, and two continuum coefficients. Limits: 1e-6 maximum absolute flux
   change and 1e-4 relative L2 change per Jacobian column.
3. **Paired fits through `fit_order`.** Generate a float64 H2O-only order at
   5000--5020 cm-1, add Gaussian noise with sigma=0.001 (three fixed seeds),
   and fit both precision modes from the same initial values. Fit H2O column,
   velocity, LSF width, and constant log throughput; fix wavelength stretch
   and jitter. Estimate local uncertainty from the float64 spectral Jacobian
   and known noise, rather than the optimizer's approximate inverse Hessian.
   Require successful optimization, less than 0.01 sigma precision-induced
   parameter shift, and recovery within 4 sigma for each seeded experiment.

Relative L2 error means `norm(candidate-reference)/norm(reference)` over a
spectrum. It avoids dividing by individual derivative pixels near zero, while
testing each parameter independently. Zero reference vectors require zero
candidate vectors under the same criterion. All arrays must be finite.

## Scope

These are first-derivative and numerical-equivalence tests. They do not validate
new atmospheric physics, full LBLRTM equivalence, independently fitted jitter,
or uncertainty calibration across a large ensemble. MT_CKD absorption is not
included in this line-opacity validation. The mixed mode does not change the
float64 continuum backend or instrument calculations.

The 400 K boundary and high-temperature fallback are tested explicitly.
Arbitrarily hot/dense atmospheres, other molecules, and higher derivatives
remain outside the validated scope.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/validate_mixed_precision.py
```

The script checkpoints completed checks to the report and exits with an error
if a tolerance is exceeded. Fit failures are collected across all seeds before
the final error. Use `--resume` to continue after an interruption
with the same inputs and environment. A complete rerun is required after
changes to the numerical kernels or validation thresholds.

## Loader issue discovered

The real CH4 AER file starts directly with numeric records, while several
other files include a header ending in a `%` delimiter. The reader previously
required that delimiter and rejected the methane file. It now recognizes
numeric records with or without the header; regression tests cover both.
