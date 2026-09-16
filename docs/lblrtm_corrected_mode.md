# Optional MT_CKD and LBLRTM-corrected modes

The default `fast` mode evaluates ExoJAX/AER Voigt lines and any explicitly
supplied continuum backend. The `mt_ckd` mode accepts either the native
`MTCKDWaterContinuum` backend or a fixed optical-depth template. The native
backend evaluates MT_CKD 4.3 from the current pressure-temperature-abundance
profile and should be preferred for fitting across atmospheric states.

The fixed template modes generated offline by LBLRTM 12.17 are:

- `mt_ckd` with a correction adds only its isolated H2O self and foreign
  MT_CKD continua;
- `lblrtm_corrected` adds those continua plus empirical per-species line
  residuals and the fixed reference background.

LBLRTM is not executed during prediction or fitting.

## Native runtime MT_CKD

```python
from jax_telluric import MTCKDWaterContinuum, TelluricModel

continuum = MTCKDWaterContinuum.from_netcdf(
    "data/lblrtm/LBLRTM/data/absco-ref_wv-mt-ckd.nc", nu_grid
)
model = TelluricModel(
    profile,
    nu_grid,
    backend,
    accuracy_mode="mt_ckd",
    continuum=continuum,
)
```

The implementation follows LBLRTM 12.17's `mt_ckd_h2o_module.f90`: it applies
the self-continuum temperature exponent, self and foreign collider densities,
the radiation factor, and the same four-point cubic interpolation. Coefficients
are loaded once and the fixed pressure-temperature spectral factors are
calculated when constructing `TelluricModel`; predictions perform no file I/O
or coefficient interpolation.
The layer pressure uses the hydrostatic column-weighted mean of its pressure
edges. Run `scripts/validate_mt_ckd.py` to reproduce direct comparisons with
LBLRTM over three pressure-temperature-water states.

Against isolated LBLRTM 12.17 continuum calculations at 5000--5020 cm-1, the
three cases give median relative optical-depth errors of 0.024%, 0.076%, and
0.029%. The largest 99th-percentile error is 0.107%; see
`docs/native_mt_ckd_validation.json` for the states and complete metrics.

The builder first uses the same pressure edges as the JAX layers and enables
the AER/HITRAN air-pressure line shifts that ExoJAX 2.5 Direct omits. LBLRTM
scales those shifts with density, so the JAX implementation uses
`delta_air * (P / 1 atm) * (296 K / T)`.

The resulting template contains two kinds of terms:

- MT_CKD 4.3 H2O self continuum, scaled with the square of the global H2O
  column multiplier;
- MT_CKD H2O foreign continuum, scaled linearly with H2O;
- an empirical line residual for every profile molecule, scaled linearly with
  that molecule. It is the measured difference between a species-only LBLRTM
  run and the selected JAX backend, rather than an attribution to one physical
  mechanism;
- the remaining full-atmosphere LBLRTM background at the reference profile.
  This includes other continua and small numerical decomposition residuals and
  is held fixed under abundance changes.

For this 5000--5020 cm-1 case, CO2 line coupling is outside the AER coupling
database's stated 597--2503 cm-1 range. LBLRTM also does not consume the AER
speed-dependence files used by MonoRTM. Those effects therefore do not explain
this order's correction. After fixing the profile extent, pressure shifts
reduce the fast 99th-percentile error from 0.107 to 0.0581. The remaining
discrepancy is concentrated in ordinary H2O and CO2 line calculations; its
exact parameter-level cause has not been isolated, so the residual remains
explicitly empirical.

All terms are vertical optical depths. The normal model airmass calculation
therefore scales them with zenith angle. Templates are valid only for their
exact pressure-temperature-abundance profile and high-resolution grid; model
construction rejects a mismatch. Generate another template when either
changes materially.

## Build and use a template

After running `scripts/bootstrap_lblrtm.sh`, generate a template for an order:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/build_lblrtm_correction.py \
  --v1 5000 --v2 5020 \
  --profile data/profiles/example_midlatitude.csv \
  --output data/corrections/lblrtm_5000_5020.npz
```

The builder runs three full-atmosphere continuum calculations plus one
continuum-free LBLRTM calculation per profile molecule. It saves a compressed
template and provenance JSON, reloads the result, checks first derivatives,
and measures fast and corrected transmission against full LBLRTM.

To enable line shifts, prepare the opacity backend with `pressure_shift=True`:

```python
backend = ExoJAXOpacityBackend.prepare(
    databases,
    nu_grid,
    methods="direct_sparse",
    temperature_range_k=(profile.temperature_k.min(), profile.temperature_k.max()),
    maximum_pressure_bar=profile.pressure_layer_bar.max(),
    vectorize_layers=True,
    pressure_shift=True,
)
```

Then select the continuum-only mode:

```python
from jax_telluric import LBLRTMOpticalDepthCorrection, TelluricModel

correction = LBLRTMOpticalDepthCorrection.load(
    "data/corrections/lblrtm_5000_5020.npz"
)
model = TelluricModel(
    profile,
    nu_grid,
    backend,
    accuracy_mode="mt_ckd",
    correction=correction,
)
```

Change the mode to `lblrtm_corrected` when the calibrated residual terms are
desired. The same correction file supports both modes.

Version-2 correction files record the line baseline used to derive their
empirical residuals. Full corrected-mode construction rejects an unshifted
ExoJAX backend instead of silently applying the wrong residual. The `mt_ckd`
terms are independent of that line baseline, so they can also be combined
with another opacity backend.

Omitting `accuracy_mode` retains the existing `fast` behavior. The `mt_ckd`
mode does not add `reference_background_optical_depth` or any
`line_residual_optical_depth` arrays. Generated
templates live under ignored `data/corrections/` because they depend on the
chosen profile, order grid, LBLRTM build, and line database.

## Measured result

The example 5000--5020 cm-1 order uses a padded 4975--5045 cm-1 grid with
2,517 samples and the six-layer example atmosphere. Against full LBLRTM,
among samples with transmission above 0.05:

| Mode | Median absolute error | 99th percentile | Maximum |
|---|---:|---:|---:|
| Fast, pressure-shifted float64 | 3.904e-3 | 5.811e-2 | 7.893e-1 |
| Pressure shifts + MT_CKD only | 5.607e-3 | 5.689e-2 | 7.879e-1 |
| LBLRTM corrected | 4.80e-8 | 4.75e-5 | 3.78e-4 |

MT_CKD slightly improves the 99th-percentile error but increases the median
in this order. The dominant JAX line residual is signed and partially cancels
the positive continuum opacity in the fast comparison. Adding a known
continuum therefore need not improve every aggregate agreement metric while
the ordinary H2O and CO2 line mismatch remains. The `mt_ckd` mode is the
physically attributable option; `lblrtm_corrected` is the close LBLRTM
emulator.

The same template was also checked against new LBLRTM runs away from its
calibration point:

| Case | Median absolute error | 99th percentile | Maximum |
|---|---:|---:|---:|
| H2O column 0.5x | 3.21e-4 | 7.24e-4 | 3.88e-3 |
| H2O column 2x | 5.08e-4 | 1.54e-3 | 3.19e-3 |
| Airmass 1.5 | 3.07e-4 | 7.96e-4 | 1.86e-3 |
| Airmass 2.5 | 8.98e-4 | 2.34e-3 | 2.95e-3 |

The builder requires 99th-percentile error below 1e-3 at the reference and
below 5e-3 for these four stress cases. Individual saturated-line pixels can
have larger errors under abundance changes, as shown by the maximum column.

On the RTX 5000 Ada, enabling pressure shifts changed the mixed-precision
H2O benchmark from 1.89 to 3.88 ms per forward call and from 3.31 to 3.56 ms
per objective-gradient call. Adding the precomputed correction arrays has
negligible cost relative to opacity evaluation. The bounds passed during
backend preparation must cover the fixed profile; wider bounds create more
sparse core pairs and cost more.

This test establishes calibrated, reference-profile accuracy, not universal
physical completeness under large abundance or pressure-temperature profile
changes. The H2O and per-species line terms have explicit scaling; the
empirical residual need not follow that linear approximation indefinitely and
the unattributed background remains fixed. Regenerate the
template outside the validated 0.5--2x water or 1--2.5 airmass range, when
changing the pressure-temperature profile, or when saturated-line accuracy
above the tabulated level is required.
