# The instrument line shape of the Arcturus IR atlas, measured from the data

The Hinkle, Wallace & Livingston (1995) atlas quotes R = 100,000 and documents
no apodization. Both can be recovered from the spectra themselves, because an
FTS spectrum is the Fourier transform of an interferogram truncated at the
maximum optical path difference (MOPD): transforming a page returns that
interferogram, whose envelope is flat out to the MOPD and then cuts.

Reproduce with:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/measure_atlas_ils.py
```

Machine-readable results, per page and per epoch, are in `docs/arcturus_ils.json`.

## Method

Each page's observed column is mean-subtracted, Hann-windowed, and transformed
with a real FFT. The MOPD is located at the **steepest sustained drop** of the
smoothed log envelope, not at the last sample above a noise threshold: the atlas
stores five significant digits, and that quantization leaves a floor near 1e-5
of the peak that extends several centimetres beyond the true cut. A threshold
detector locks onto that floor and overestimates the MOPD by up to 5 cm.

The transition is sharp. For `ab5000_` summer the envelope runs at ~1e-2 of peak
out to 15.17 cm and then falls 2.65 decades within 0.1 cm.

## Results

Measured over 598 page-epoch combinations. Within each band the MOPD is constant
to better than the 0.03 cm FFT bin (MAD 0.00–0.28 cm); between bands it steps.

| Δν (cm⁻¹) | ν range (cm⁻¹) | MOPD summer (cm) | MOPD winter (cm) | median R summer | median R winter |
|---|---|---:|---:|---:|---:|
| 0.0100 | 1866–3136 | 25.29 | 25.60 | 109,600 | 107,700 |
| 0.0125 | 3134–3471 | 24.25 | 18.97 | 132,700 | 103,800 |
| 0.0200 | 3997–6677 | 15.17 | 13.60 | 137,600 | 123,700 |
| 0.0330 | 7398–8902 | 8.86 | 8.86 | 120,100 | 119,600 |
| 0.0400 | 8897–10953 | 6.74 | 6.74 | 110,800 | 110,800 |

Every measured MOPD sits comfortably inside its band's Nyquist path difference
(1/2Δν), so these are genuine interferogram truncations and not an artefact of
the per-band decimation.

### Three corrections to the documented description

1. **The ILS is a sinc, not a Gaussian.** The cut is hard: the transition from
   in-band to floor spans 0.4–0.6 cm (summer) and 0.7–1.3 cm (winter), i.e. 3–8%
   of the MOPD. Norton-Beer apodization tapers over roughly L/3 — 4–8 cm here —
   so medium and strong apodization are excluded outright. The data are
   unapodized or very weakly apodized, and a boxcar sinc is the right default.
   Independent confirmation: `ab5000_` summer contains 4 negative flux pixels in
   5005–5025 cm⁻¹, all at model transmission below 0.005. Sinc ringing produces
   those; a Gaussian ILS cannot produce a negative value anywhere.

2. **"Constant R" and "constant width in wavenumber" are both true, at different
   scales.** Within a band the MOPD is fixed, so the ILS width is constant in
   wavenumber and R varies across the band — by a factor 1.67 across the
   3997–6677 cm⁻¹ band alone. Between bands the MOPD steps so that R stays
   roughly constant. Neither description is adequate on its own.

3. **R is not 100,000.** It is 104,000–138,000 depending on band and epoch. At
   `ab5000_` specifically:

   | epoch | MOPD (cm) | ILS FWHM (cm⁻¹) | R at 5012 cm⁻¹ | samples per FWHM |
   |---|---:|---:|---:|---:|
   | summer | 15.167 | 0.03978 | 125,990 | 1.99 |
   | winter | 13.600 | 0.04436 | 112,976 | 2.22 |

   **The two epochs are effectively different instruments** in this band — an
   11.5% difference in resolving power. They must not share an LSF parameter.
   The epochs also differ in the 0.0125 band (24.25 vs 18.97 cm) but are
   identical in the 0.0330 and 0.0400 bands.

## Other findings

- **The summer epoch has no data below ~2020 cm⁻¹.** Twenty-two pages from
  `ab1867_` to `ab2013_` have an identically-zero summer column. They are
  reported as errors in the JSON rather than silently skipped.
- Sampling is adequate everywhere: at `ab5000_` the band limit 1/(2L) is
  0.0329 cm⁻¹ against 0.02 cm⁻¹ sampling, i.e. 1.65× oversampled (summer). The
  spectrum is exactly band-limited and there is no aliasing.

## Discrepancy with `differentiable_stellar_spectroscopy`

The sibling repository's `dss/calib/lsf.py` registers this atlas under
`ATLAS_LSF` as constant R = 100,000 with the apodization left open, and its
docstring infers constant R from the sampling stepping between bands. On this
measurement that inference is backwards — the sampling steps because the MOPD
steps, and within a band the width is constant in wavenumber, not in R. The
recorded value is also 10–26% low, and a single entry cannot represent the two
epochs, which differ by 11.5% in the band of interest.

Reported here only; that repository is not modified by this work.
