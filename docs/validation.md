# Validation workflow

## Reference software and data

Use LBLRTM 12.17, the current LNFL release, MT_CKD 4.3, and AER Line File 3.9.
Store sources, binaries, databases, and full model output under the ignored
`data/lblrtm/` directory. Commit only compact `.npz` fixtures with provenance,
inputs, checksums, and the spectral samples needed by tests.

The repository includes one such fixture for 5000--5100 cm-1. Rebuild it from
`data/profiles/example_midlatitude.csv` after bootstrapping:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_lblrtm_reference.py
```

`jax_telluric.compare_transmission` reports the median, 99th-percentile, and
maximum absolute errors plus the global line displacement in resolution
elements. It interpolates the candidate spectrum onto the reference grid and
applies the transmission threshold below.

The Python package uses a local `uv` environment. LBLRTM 12.14 and newer also
needs netCDF-Fortran, so its compiler toolchain lives in `.conda-lblrtm/` and is
independent of `.venv`.

## Comparison ladder

1. Match a single layer with continuum disabled and identical retained lines.
2. Match a dry multilayer atmosphere.
3. Enable H2O self broadening.
4. Add H2O, CO2, CH4, O2, CO, and N2O.
5. Extract self and foreign MT_CKD H2O continuum terms and enable their
   differentiable abundance scaling.
6. Compare zenith angles corresponding to airmasses 1.0, 1.5, and 2.5.
7. Convolve and integrate onto an IGRINS-like R=45,000 detector grid.

The implemented first rung is a 4290--4310 cm-1 CO-only calculation with
continuum disabled. `scripts/validate_aer_co.py` runs LBLRTM and ExoJAX from
the same AER 3.9 line file and updates the recorded metrics. This case meets
all three acceptance thresholds below. The remaining rungs are explicitly
future validation work; their effects are already isolated in the reference
fixture where possible.

An optional fixed-profile implementation now covers the correction ladder for
the 5000--5020 cm-1 example order. `scripts/build_lblrtm_correction.py`
isolates H2O self/foreign continua and per-species LBLRTM line residuals, then
stores the remaining reference background. Its corrected 99th-percentile
absolute error is 6.61e-5. Checks at 0.5--2x water abundance and airmass
1--2.5 remain below 3.10e-3 at the 99th percentile. This validates that
profile and grid; broader pressure-temperature and wavelength coverage
remains future work.

Use a water-dominated H interval and a mixed-species K interval. Within pixels
whose reference transmission exceeds 0.05, require median absolute error below
`1e-3`, 99th-percentile error below `5e-3`, and line-center displacement below
0.1 resolution element. Report continuum, line-data, and line-mixing residuals
separately rather than hiding them in a combined score.

## Gradient and recovery checks

Compare every fitted JAX derivative with a central finite difference away from
bounds. The relative difference target is `1e-4`. Synthetic noisy orders must
recover injected molecular scales, wavelength correction, LSF width, continuum,
and jitter within three estimated standard deviations.
