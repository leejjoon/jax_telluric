# Optional LBLRTM-corrected mode

The default `fast` mode evaluates ExoJAX/AER Voigt lines and any explicitly
supplied continuum backend. The opt-in `lblrtm_corrected` mode adds a compact,
differentiable optical-depth template generated offline by LBLRTM 12.17.
LBLRTM is not executed during prediction or fitting.

The template contains:

- MT_CKD 4.3 H2O self continuum, scaled with the square of the global H2O
  column multiplier;
- MT_CKD H2O foreign continuum, scaled linearly with H2O;
- a line residual for every profile molecule, scaled linearly with that
  molecule. This captures line coupling, speed-dependent line shapes, distant
  lines absent from the local ExoJAX database, and other line-model residuals;
- the remaining full-atmosphere LBLRTM background at the reference profile.
  This includes unattributed non-H2O continua and interpolation residuals and
  is held fixed under abundance changes.

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

```python
from jax_telluric import LBLRTMOpticalDepthCorrection, TelluricModel

correction = LBLRTMOpticalDepthCorrection.load(
    "data/corrections/lblrtm_5000_5020.npz"
)
model = TelluricModel(
    profile,
    nu_grid,
    opacity_backend,
    accuracy_mode="lblrtm_corrected",
    correction=correction,
)
```

Omitting `accuracy_mode` retains the existing `fast` behavior. Generated
templates live under ignored `data/corrections/` because they depend on the
chosen profile, order grid, LBLRTM build, and line database.

## Measured result

The example 5000--5020 cm-1 order uses a padded 4975--5045 cm-1 grid with
2,517 samples and the six-layer example atmosphere. Against full LBLRTM,
among samples with transmission above 0.05:

| Mode | Median absolute error | 99th percentile | Maximum |
|---|---:|---:|---:|
| Fast mixed precision | 1.653e-2 | 1.019e-1 | 7.669e-1 |
| LBLRTM corrected | 5.68e-8 | 6.61e-5 | 1.94e-4 |

The same template was also checked against new LBLRTM runs away from its
calibration point:

| Case | Median absolute error | 99th percentile | Maximum |
|---|---:|---:|---:|
| H2O column 0.5x | 2.41e-4 | 8.24e-4 | 9.67e-3 |
| H2O column 2x | 4.03e-4 | 3.09e-3 | 1.06e-2 |
| Airmass 1.5 | 1.88e-4 | 6.44e-4 | 1.91e-3 |
| Airmass 2.5 | 6.36e-4 | 1.88e-3 | 2.79e-3 |

The builder requires 99th-percentile error below 1e-3 at the reference and
below 5e-3 for these four stress cases. Individual saturated-line pixels can
have larger errors under abundance changes, as shown by the maximum column.

On the RTX 5000 Ada, the fast and corrected modes respectively measured
2.23/2.03 ms per forward call and 3.43/3.09 ms per objective-gradient call.
The difference is benchmark noise: adding precomputed arrays has negligible
cost relative to opacity evaluation.

This test establishes reference-profile accuracy, not universal scaling under
large abundance or pressure-temperature profile changes. The H2O and
per-species line terms have
explicit scaling; the unattributed background remains fixed. Regenerate the
template outside the validated 0.5--2x water or 1--2.5 airmass range, when
changing the pressure-temperature profile, or when saturated-line accuracy
above the tabulated level is required.
