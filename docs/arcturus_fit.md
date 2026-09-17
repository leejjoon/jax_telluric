# Fitting telluric absorption to the Arcturus atlas

Work in progress. This records the configuration, the synthetic validation, the
telluric-only baseline, and the fit with a fixed Payne Zero stellar source.

Machine-readable companions: `docs/arcturus_ils.json` (instrument profile),
`docs/arcturus_validation.json` (synthetic checks L0–L2),
`docs/arcturus_ab5000_fit.json` (telluric only),
`docs/arcturus_ab5000_fit_star.json` and `docs/arcturus_ab5000_winter.json`
(both epochs with the star), `docs/arcturus_two_epoch.json` (L5),
`docs/arcturus_ab5025_fit.json` (L7), `docs/arcturus_ab4750_fit.json` (L8).

Reproduce the whole chain:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/measure_atlas_ils.py
UV_CACHE_DIR=.uv-cache uv run python scripts/make_site_profile.py
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_arcturus_fit.py
# the stellar source needs Payne Zero's own Python 3.11 environment
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_arcturus_page.py \
    --stellar data/stellar/arcturus_payne_zero_1960_2130nm.npz \
    --stages continuum,velocity,columns,stellar
```

## Target

Page `ab5000_` of the Hinkle, Wallace & Livingston (1995) IR atlas, summer
epoch, restricted to **5005.0–5025.0 cm⁻¹** (1990.0–1998.0 nm, 1001 pixels).

The window is chosen for one reason: the 2.0 µm CO₂ band dies at about
5010 cm⁻¹, and 5005–5010 cm⁻¹ is the only part of this band where a well-mixed
absorber is present but unsaturated. Airmass and a per-species column scale are
otherwise exactly degenerate, because the transmission is `exp(−Στ/cos z)` and
scaling every species is indistinguishable from scaling the path. That
sub-window is the only thing in the page that breaks the degeneracy.

The page is trimmed off its own seams. Adjacent pages overlap by about 2 cm⁻¹
and differ there by a per-page multiplicative scalar, so the overlap is cut
rather than blended. Below 5005 cm⁻¹ the CO₂ band core is saturated, the
observed flux goes negative, and the atlas's own telluric column reaches 2×10⁵.

## Configuration

| choice | value | why |
|---|---|---|
| accuracy mode | `mt_ckd` with the native continuum | no LBLRTM at fit time; see the bias below |
| species | H₂O, CO₂ | CH₄, CO, O₂, N₂O have mean τ ≤ 2e-4 here; freeing them only moves them to a bound |
| internal grid | R = 100,000 × 4 → 5,585 points, 0.749 km/s | L1 |
| instrument | unapodized FTS sinc, MOPD 15.167 cm | measured; L2 |
| pixel sampling | point, not Simpson | FTS data are point samples of a band-limited function |
| line shifts | HITRAN `delta_air`, density-scaled | 0.47 km/s at 0.78 atm, which a single velocity cannot absorb |
| atmosphere | `data/profiles/kitt_peak_1994.csv` | 1993–94 trace gases; CO₂ 357 ppm, not 420 |
| wavelength stretch | pinned to 0 | an FTS wavenumber scale is exactly linear |
| continuum | degree-3 Chebyshev in log | L1/L2 aside, higher degrees eat the H₂O band wing |

**The dominant systematic is the line physics, not the noise.** Without the
offline LBLRTM correction template, this model disagrees with LBLRTM 12.17 in
this window by median 0.0039 and 99th percentile 0.058
(`data/corrections/lblrtm_5000_5020.json`), against a per-pixel noise near
0.0058. That error is structured around strong lines and is partly absorbed by
the free continuum and column scales, so **the fitted scales carry a
line-physics bias of that order**. Reported, not tuned away.

## Synthetic validation

Reproduce with:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_arcturus_fit.py
```

One reference spectrum is generated on the finest grid with the measured sinc
from a known truth (H₂O × 1.20, CO₂ × 1.15, v = +0.30 km/s) plus 0.005 noise,
then fitted back with each candidate configuration, so each comparison isolates
one model choice.

**L0 — injection recovery.** H₂O recovered to −0.295%, CO₂ to −0.521%, velocity
to −2.4 m/s, reduced χ² = 1.049. **Passes.**

**L1 — grid convergence.**

| grid | points | step | H₂O shift vs finest | CO₂ shift vs finest | rms/noise |
|---|---:|---:|---:|---:|---:|
| R=45,000 × 4 | 2,514 | 1.665 km/s | 0.06% | **2.03%** | **1.528** |
| R=100,000 × 4 | 5,585 | 0.749 km/s | 0.019% | 0.077% | 1.030 |
| R=200,000 × 4 | 11,168 | 0.375 km/s | — | — | 1.024 |

R = 100,000 × 4 is converged at the 0.1% level. **The R = 45,000 × 4 grid the
package ships by default is not usable here**: its velocity step is coarser than
the 1.196 km/s data pixel, and it biases CO₂ — the species carrying the airmass
information — by 2%.

**L2 — instrument profile.** Fitting the sinc-generated reference with a
Gaussian line spread function gives rms 1.276× the noise against 1.030× for the
sinc, a 1.24× penalty, and biases CO₂ by +0.61%. Against a 1.2× threshold this
**fails**, which is the quantitative case for implementing the measured sinc
rather than approximating it.

### Wavelength alignment

One velocity is fitted (`velocity_kms`); `wavelength_stretch` is pinned at zero,
on the argument that an FTS wavenumber scale is exactly linear so its only
freedom is a constant multiplicative factor, which is a velocity. That was
checked rather than assumed. Cross-correlating the telluric component against
the model in 2.5 cm⁻¹ windows with absorption depth above 0.10 leaves a residual
lag of **−0.008 km/s (summer) and +0.030 km/s (winter), i.e. 0.01–0.02 of a
pixel**. The apparent trend across the order, +0.27 km/s in summer and −0.09 in
winter, is smaller than the window-to-window scatter and **reverses sign between
epochs**, so it is not an instrumental stretch: a real wavenumber-scale error
would be common to both. Freeing the stretch would only absorb model error.

That is the *global* shift only, and it is not the whole question. A matched
filter — regressing the residual onto the model's derivative, which weights the
steep flanks where a displacement shows — gives a global telluric shift of
**−0.003 ± 0.005 km/s (summer)** and **−0.005 ± 0.005 (winter)**, confirming
there is no wavelength-scale error at 0.004 of a pixel. But the same filter
applied line by line finds individual telluric lines displaced by up to
**±0.5 km/s**, at 10–20σ of their own errors:

| | summer | winter |
|---|---:|---:|
| global shift | −0.003 ± 0.005 km/s | −0.005 ± 0.005 |
| per-line scatter | 0.206 km/s | 0.100 |
| largest single line | −0.51 km/s (5013.26) | −0.16 (5024.66) |
| correlation of the same lines between epochs | **r = 0.54** | |

Two consequences. **Dividing by the transmission amplifies this by 1/T**, so a
displacement of a few hundredths of a pixel, nearly invisible in the fit
residual, becomes a conspicuous positive-then-negative spike in the corrected
spectrum beside every deep line. That is the dominant visible defect in the
corrected product. And **no global parameter can remove it**: freeing
`wavelength_stretch` would fit a smooth ramp to a line-by-line scatter and
absorb model error in the process.

The r = 0.54 repeatability across two epochs, taken months apart through
different atmospheres, says a good part of this is a property of the line data
rather than the observation. One caveat on the interpretation: the derivative
filter responds to any antisymmetric residual, so it cannot separate a genuine
line-position error from an asymmetric depth error or an unmodelled blend. The
+0.46 km/s at 5012.06 is visibly the latter — the model is too shallow on the
blue flank, not displaced. The measurement bounds the effect; it does not
attribute all of it to line centres.

**A plotting trap worth knowing about.** `SpectralOrder.flux` must be finite and
positive everywhere, so `arcturus_spectral_order` stores 1.0 at masked pixels.
Those placeholders sit in the core of every saturated line, and plotting the
array directly puts a spike to 1.0 there that reads convincingly as a
wavelength shift. The diagnostic products now also carry `observed_raw` and
NaN-masked plotting columns; use those.

### A caution on the reported errors

`FitResult.covariance` comes from L-BFGS-B's inverse-Hessian approximation,
which is a byproduct of its line search and not a covariance. Here it
overestimates the column uncertainties by more than two orders of magnitude —
34–38% against a measured grid-to-grid scatter near 0.1%. The validation
thresholds are therefore absolute fractions, not pulls. **Do not quote those
formal errors as uncertainties.**

## Telluric-only fit

With a flat source — no stellar model — the three staged fits converge in 168 s
on one GPU:

| quantity | value |
|---|---|
| H₂O scale | 1.206 |
| CO₂ scale | 1.212 |
| velocity | +0.264 km/s |
| residual Gaussian σ | at its 0.05 km/s lower bound |
| residual rms | 0.0674, against 0.00575 pixel noise |
| jitter / uncertainty | 11.7 |
| vs atlas telluric column | median 0.018, p99 0.070 |

Three things are worth reading off this, and one thing is not.

**The velocity is right.** +0.264 km/s against +0.30 km/s measured
independently by cross-correlating the AER line positions against the data. It
is the FTS wavenumber-scale zero point, not wind, and it is the same in both
epochs.

**The measured sinc accounts for all the instrumental broadening.** The residual
Gaussian goes to its lower bound, meaning there is nothing left for it to
absorb. That is an independent check on the MOPD measurement.

**The residual is the star.** rms 0.0674 is 11.7× the pixel noise, and it is
largest where the atmosphere is most transparent. The deepest excursions reach
−85σ at isolated wavenumbers, and four of them match the atlas's own atomic line
identifications (`appendix.a`) at a consistent Doppler offset:

| observed (cm⁻¹) | appendix.a | implied velocity |
|---|---|---:|
| 5015.040 | 5015.254 | +12.8 km/s |
| 5016.240 | 5016.464 | +13.4 |
| 5017.640 | 5017.869 | +13.7 |
| 5019.180 | 5019.418 | +14.2 |

Mean **+13.5 ± 0.5 km/s**, confirmed by an independent two-epoch
cross-correlation giving +13.57. A telluric mismatch would sit at the telluric
rest frame, near zero, so these features are stellar.

**This also fixes the sign of the epoch velocity.** Reading the `readme.dat`
Doppler factor as a division put summer at −14.33 km/s. The lines are
redshifted, so summer is *receding*: `epoch_velocity_kms` now returns +14.33,
with the measured sign recorded in its docstring and asserted in the tests.

**What must not be read off this:** the H₂O and CO₂ scales. They come out within
0.5% of each other, which is the airmass direction of the degeneracy, and with
no stellar model the degree-3 continuum has an entire stellar spectrum to
absorb. No airmass or precipitable-water value should be quoted until the star
is in the fit.

## With the Payne Zero stellar model

The source is a fixed Payne Zero synthesis at the literature labels
(T_eff 4286 K, log g 1.66, [M/H] −0.52, [α/M] 0.30, ξ 1.7 km/s), generated by
`scripts/generate_payne_zero_arcturus.py` in Payne Zero's own Python 3.11
environment and read back as an npz. It covers 1960–2130 nm at
R_grid = 600,000 (0.50 km/s, finer than the 0.749 km/s model grid).

Payne Zero's published examples stop at 1700 nm, but nothing limits it there;
it synthesized this window in 13 s on CPU.

**Recorded limitation:** `synthesize_from_labels` returns
`atmosphere_converged: False` and `atmosphere_closure_required: True`. The
spectrum rests on the learned initializer atmosphere, not a converged solve.

| | flat source | Payne Zero star |
|---|---:|---:|
| H₂O scale | 1.206 | 1.088 |
| CO₂ scale | 1.212 | 1.252 |
| telluric velocity | +0.264 km/s | +0.277 km/s |
| stellar velocity | — | **+13.67 km/s** |
| residual rms | 0.0674 | **0.0221** |
| reduced χ² | 137.3 | 14.8 |
| jitter / σ | 11.7 | 3.7 |
| continuum coefficients | (−0.015, −0.029, 0.023, −0.029) | (−0.0003, 0.0069, 0.0062, 0.0024) |

Two things stand out. The **stellar velocity is a prediction**: it started at
the readme value +14.33 and moved to +13.67, toward the +13.5 ± 0.5 measured
from `appendix.a` line matching and the +13.57 from cross-correlation. And the
**continuum went quiet** — its coefficients fall by more than an order of
magnitude, because the star now accounts for the shape the continuum was
absorbing. That also broke the airmass degeneracy: H₂O and CO₂ separate from
1.206/1.212 to 1.088/1.252.

## How to divide the telluric out

The obvious correction, `observed / transmission`, leaves a large
positive-then-negative spike beside every strong telluric line. That spike is
**not** a wavelength misalignment. It is intrinsic to the division:

    observed = continuum x Conv(T x S)

and `Conv(T x S) / Conv(T)` is not `S`, because convolution does not commute
with multiplication. Where `T` varies sharply the mismatch is derivative-shaped
and then amplified by 1/T.

Measured by applying the same division to the *model*, which contains no data
at all, **86% of the excursion is reproduced** — so it is the method, not the
fit. Correcting by the model ratio instead,

    corrected = (observed / model) x star

transfers the data's information onto the stellar model without ever dividing
by a convolved transmission, and the two agree exactly where the fit is
perfect. Median peak-to-peak within ±0.15 cm⁻¹ of the strong lines:

| | naive division | model ratio |
|---|---:|---:|
| peak-to-peak at strong lines | 0.319 | **0.068** (4.7x smaller) |
| same operation on the model alone (pure artefact) | 0.406 | **0.030** (14x smaller) |
| **two epochs differenced, stellar rest frame** | 0.0357 | **0.0183** rms |

What remains in the model-ratio product is real model error — the per-line
shifts and depth errors above — which is what should remain. The two-epoch
disagreement halving is the strongest evidence the change is a genuine
improvement rather than cosmetic: the two epochs are independent observations,
and pure noise would give 0.0078.

`scripts/fit_arcturus_page.py` therefore also writes `stellar_only_pixels`, the
same forward model with the atmosphere removed, and
`scripts/export_corrected_spectrum.py` uses the model ratio by default, keeping
the naive division alongside as `corrected_naive` for comparison.

## L5 — two epochs

Summer and winter fitted independently, same stellar spectrum, each with its
own measured MOPD (15.17 vs 13.60 cm). `docs/arcturus_two_epoch.json`.

| | summer | winter |
|---|---:|---:|
| H₂O scale | 1.088 | 0.383 |
| CO₂ scale | 1.252 | 0.950 |
| stellar velocity | +13.67 | −26.76 km/s |
| implied airmass (from CO₂) | 1.25 | **0.95** |
| implied zenith PWV | 4.34 mm | 2.01 mm |

**The stellar velocity separation is 40.43 km/s**, against 40.44 from the
readme factors and 40.5 from cross-correlation — two independent fits, on
different data with different instrument profiles, agreeing to 0.01 km/s.

**Winter's implied airmass of 0.95 is unphysical**, 5% below 1. That is the
honest systematic floor of the CO₂ route here: it folds in the AER line
strengths, the assumed 357 ppm, and the mt_ckd line-physics bias. It is
reported, not tuned away.

**Where the remaining error lives.** `scripts/compare_arcturus_epochs.py`
shifts each epoch into *both* candidate rest frames using its own fitted
velocity, so neither frame is handicapped by being left unaligned, and repeats
each with the sign reversed as a null control:

| frame | correlation | reversed sign | difference rms |
|---|---:|---:|---:|
| telluric rest | 0.152 | 0.154 | 0.0289 |
| **stellar rest** | **0.824** | **0.054** | 0.0130 |

The residual is **dominated by the stellar model**, not the atmosphere. Against
an expected 0.0078 for pure noise, the stellar-frame difference of 0.0130 puts
the remaining *telluric* error near 0.009 rms, about 1.7× the photon noise.

Two details matter for reading this. The two epochs' telluric velocities differ
by only 0.123 km/s — **0.10 of a pixel** — so aligning on the telluric frame is
below the sampling and changes the correlation by 0.001; the comparison is
insensitive to it either way. And the sign control is what rules out the
resampling manufacturing the agreement: reversing the stellar shift collapses
the correlation from 0.824 to 0.054, while reversing the telluric shift changes
nothing, exactly as it should when there is no feature there to align.

## L3 — fitting the atlas's own telluric column

The atlas ships the transmission its authors divided by. Fitting that column
directly is a star-free consistency target, run with a flat source:
`docs/arcturus_ab5000_telluric_column.json`.

| | observed + star | atlas telluric column |
|---|---:|---:|
| H₂O scale | 1.088 | 1.085 (**−0.3%**) |
| CO₂ scale | 1.252 | 1.123 (**−10.3%**) |
| telluric velocity | +0.277 | +0.029 km/s |
| residual rms | 0.0221 | 0.0315 |

**H₂O agrees to 0.3%.** Our water column and Hinkle's independent telluric
determination are the same number, which is a real cross-check: they come from
different reductions of different data.

**CO₂ disagrees by 10.3%**, missing the 10% threshold, and the rms of 0.0315
misses its 0.03 threshold. Both are marginal, but the CO₂ direction matters: it
is the species carrying the airmass, and a 10% error there is the same story as
winter's unphysical 0.95 airmass. Two independent routes now point at CO₂
rather than H₂O as the weak link — the AER CO₂ line strengths, the assumed
357 ppm, or the mt_ckd line-physics bias concentrated in the strong CO₂ band.

The velocity offset of 0.25 km/s is expected and not a failure: Hinkle's
telluric column comes from a different observation, so its wavelength zero
point need not match this page's.

## L7 and L8 — cross-page and null page

| page | window (cm⁻¹) | H₂O | CO₂ | stellar v | rms | rms/noise |
|---|---|---:|---:|---:|---:|---:|
| ab5000 summer | 5005–5025 | 1.088 | 1.252 | +13.67 | 0.0221 | 3.85 |
| ab5000 winter | 5005–5025 | 0.383 | 0.950 | −26.76 | 0.0224 | 4.21 |
| ab5025 summer (L7) | 5025–5050 | 1.151 | pinned | +13.41 | 0.0206 | 4.84 |
| ab4750 summer (L8) | 4750–4775 | 1.180 | 1.121 | +14.90 | 0.0121 | 3.74 |

**L7 fails, marginally.** With CO₂ pinned at the ab5000 value, the H₂O slant
column on `ab5025_` comes out **+5.84%** higher, against a 5% threshold. The
sub-window scan below shows what that number is actually made of.

### What L7 is measuring

`scripts/scan_arcturus_subwindows.py` refits the water column in 5 cm⁻¹
sub-windows with everything except H₂O and the continuum level held at the
page's own full-window solution.

| | H₂O scale | peak-to-peak |
|---|---|---:|
| within `ab5000_`, 4 sub-windows | 1.0773 ± 0.0411 | 9.0% |
| within `ab5025_`, 5 sub-windows | 1.1523 ± 0.0364 | 6.5% |
| between pages, full windows | 1.0875 vs 1.1510 | 5.8% |
| **identical region, both page files** | **1.0778 vs 1.0778** | **0.00%** |

**The pages are not the problem.** They overlap over 5022.10–5027.08 cm⁻¹, 250
pixels, where their fluxes differ by a constant 0.99471 ± 0.00001. Fitting that
identical region out of each file gives the same water column to four decimals:
the free continuum term absorbs the page scalar exactly, as it should.

**The line-to-line scatter is larger than the page-to-page difference.** A
single 5 cm⁻¹ sub-window returns a column anywhere in a 6.5–9% range. Comparing
the means, the two pages differ by 2.9σ of that scatter (2.8σ if the
telluric-free 5015–5020 window, where water is barely constrained, is dropped).
So the disagreement is real but small, and **the 5% threshold was tighter than
the intrinsic line-dependent systematic** — a water column from a 20 cm⁻¹
window carries roughly ±3.5% of line-dependent uncertainty before any question
of weather or calibration arises.

The residual conclusion is unchanged and now better localised: this is H₂O
line-parameter and line-physics error, of the size expected for the mt_ckd
bias, and it sets a floor of a few percent on any column retrieved from a
single short window.

**L8 is the cleanest measure of stellar-model error.** `ab4750_` is nearly
telluric-free — fitted transmission median 0.989, only 4.8% of pixels below
0.9 — yet the residual is still 0.0121, **3.7× the photon noise**, with no
parameter on a bound. With almost no atmosphere to get wrong, that residual is
essentially all stellar. It agrees with L5's verdict from a completely
different direction.

### The stellar velocity spread is the stellar model, not the atlas

The three summer pages give stellar velocities of +13.67, +13.41 and +14.90 km/s.
A per-page wavelength-scale error would move the *telluric* velocity by the same
amount, and it does not:

| page | telluric velocity | stellar velocity |
|---|---:|---:|
| ab5000 (5005–5025) | +0.277 | +13.67 |
| ab5025 (5025–5050) | +0.326 | +13.41 |
| ab4750 (4750–4775) | +0.287 | +14.90 |
| peak-to-peak | **0.050** | **1.493** |

The telluric velocity is stable to 0.050 km/s while the stellar velocity moves
1.493 km/s — a factor of 30. The atlas wavelength scale is therefore not the
cause. The outlier is `ab4750_`, the one page in a different spectral region
(2.10 µm against 2.00 µm for the other two), so this is region-dependent error
in the stellar line positions.

## Error budget as it stands

| term | rms | evidence |
|---|---:|---|
| photon noise | 0.0055 | second-difference estimator |
| telluric model | ~0.009 | L5 stellar-frame differencing |
| stellar model | ~0.012 | L8 near-telluric-free page |
| total observed | 0.022 | the fit |

The telluric side is now the smaller of the two model errors. Further progress
on this page is limited by the stellar spectrum — the CN lines a K-giant
synthesis models poorly, and the un-converged initializer atmosphere — not by
the atmosphere.

## Where it stands

| check | result |
|---|---|
| L0 injection recovery | pass |
| L1 grid convergence | pass at R=100,000 × 4 |
| L2 Gaussian vs measured sinc | fails by design — the case for the sinc |
| L3 atlas telluric column | H₂O passes at 0.3%, CO₂ fails at 10.3% |
| L5 two epochs | pass; residual localised to the stellar model |
| L7 cross-page H₂O | fails at 5.8%, but that is 2.9σ of a 6.5–9% within-page scatter |
| L8 null page | pass; confirms L5 independently |

Every failure points the same way: CO₂ and the H₂O line physics, not the
machinery. L6, the direct LBLRTM comparison, is not run — under `mt_ckd` it
would measure the known bias rather than test anything.

## The stellar limit: a converged atmosphere does not help

Every check above put the residual on the stellar model rather than the
atmosphere, and `synthesize_from_labels` reports `atmosphere_converged: False`,
so the obvious suspect was the learned initializer atmosphere. It is not.

Solving the physical atmosphere for Arcturus (`payne-zero-atmosphere`, 13.7 min,
mostly first-use Numba compilation) and re-synthesizing gives a spectrum almost
identical to the initializer's:

| window | rms difference | max difference | deep lines (converged / initializer) |
|---|---:|---:|---|
| 1.04 µm | 0.00048 | 0.0058 | 255 / 255 |
| 2.00 µm | 0.00069 | 0.0043 | 507 / 522 |
| 3.06 µm | 0.00056 | 0.0040 | 373 / 374 |

That is 50–100× smaller than the residuals it would have to explain, and
refitting confirms it:

| page | initializer | converged |
|---|---:|---:|
| ab9550 (1.04 µm) | 5.04 | 5.01 |
| ab3255 (3.06 µm) | 7.26 | 7.25 |

**The `atmosphere_converged: False` flag is a red herring.** The initializer is
accurate well below anything that matters here, and the deep-line counts are
unchanged — a converged structure adds no lines. The stellar error is in the
**line list**, not the atmospheric structure.

### What would fix it, and why it is not a quick win

Payne Zero ships exactly the right tool: `linelist_calibration` fits oscillator
strengths and three damping terms through differentiable synthesis, and the
bundled `sun_arcturus_fts_hband_shared` overlay was calibrated against *this
atlas*, both epochs. It applies cleanly to the catalog — 42,886 components
matched, none unmatched, source hash verified — and the corrections are large:
10,309 oscillator strengths changed, spanning 0.17x to 5.2x at the 5–95% range.

Two things stop it being a drop-in:

* **Scope.** `scope_wavelength_nm` is [1500, 1700] — the H band only. It does
  nothing for the pages diagnosed above at 1.04 and 3.06 µm.
* **Reach.** Neither `synthesize` nor `synthesize_from_labels` accepts a
  calibration; only `fitter/apogee/forward_model.py` does, and that hardcodes
  the APOGEE window. Using the overlay for arbitrary windows means either
  patching the raw source catalog by hand or driving Payne Zero's internal
  window-bundle machinery.

And one caveat that applies even if it were wired in: the overlay was fitted
against this same atlas, so an improvement in the H band would be partly
refitting the same data, not an independent validation.

## Still to do
- Whether an LBLRTM-corrected template closes L3's CO₂ gap and L7's H₂O gap;
  both are the right size for the mt_ckd bias.
- Real uncertainties. Every number above is a point estimate, because
  `FitResult.covariance` cannot be used (see the caution above).
