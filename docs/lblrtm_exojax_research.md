# LBLRTM versus ExoJAX for IGRINS telluric fitting

## Conclusion

ExoJAX 2.5 is a strong opacity and automatic-differentiation engine, but it is
not a drop-in differentiable LBLRTM. Its official telluric example models one
effective H2O slab. LBLRTM uses a vertical terrestrial atmosphere, AER-adjusted
line parameters, MT_CKD continuum absorption, and line coupling for selected
molecules. This project therefore keeps ExoJAX behind an opacity seam and owns
the terrestrial profile, path, instrument, and inference logic.

The numerical target is convolved IGRINS H/K transmission, rather than every
radiance and scattering mode supported by LBLRTM.

## Capability comparison

| Area | ExoJAX 2.5 | LBLRTM 12.17 | Project decision |
|---|---|---|---|
| Molecular lines | HITRAN/HITEMP/ExoMol loaders and differentiable Voigt opacity | AER-adjusted HITRAN line file | Use ExoJAX calculators and record line-source checksums |
| Atmosphere | Layer-oriented opacity and radiative-transfer helpers | Standard or explicit Earth profiles | Accept an explicit pressure, temperature, altitude, and VMR profile |
| Direct telluric model | Single-slab official tutorial; `ArtAbsPure` also supports Earth transmission | Mature terrestrial slant paths | Sum layer optical depths and expose zenith angle explicitly |
| Continuum | HITRAN CIA facilities | MT_CKD 4.3 | Import separately generated self/foreign reference terms and preserve their abundance scaling |
| Line mixing | No documented LBLRTM-equivalent implementation | O2, CO2, and CH4 coupling | Quantify separately; add only where it matters in H/K |
| Instrument | Gaussian FFT/overlap-add convolution and sampling | TelFit Gaussian resolution and resampling | Apply Gaussian LSF and pixel integration on padded order grids |
| Inference | End-to-end JAX derivatives and probabilistic-programming compatibility | Selected analytic Jacobians | Use JAX gradients in bounded Gaussian MAP fitting |

## Primary sources

- [ExoJAX telluric fitting tutorial](https://secondearths.sakura.ne.jp/exojax/tutorials/Fitting_Telluric_Lines.html)
- [ExoJAX radiative-transfer interfaces](https://secondearths.sakura.ne.jp/exojax/exojax/exojax.rt.html)
- [ExoJAX spectral operators](https://secondearths.sakura.ne.jp/exojax/userguide/sop.html)
- [ExoJAX source and license](https://github.com/HajimeKawahara/exojax)
- [LBLRTM source and model/data compatibility table](https://github.com/AER-RC/LBLRTM)
- [AER Line File download repository](https://github.com/AER-RC/AER_Line_File)
- [LNFL source](https://github.com/AER-RC/LNFL)
- [TelFit source](https://github.com/kgullikson88/Telluric-Fitter)

The AER Line File repository currently pairs LBLRTM 12.17 and MT_CKD 4.3 with
line-file release 3.9. This supersedes the older 3.8.1 value still visible in
some LBLRTM documentation.

`AERLineDatabase` reads AER's per-molecule 100-character files directly into
the interface required by ExoJAX `OpaDirect`. This makes the ordinary Voigt
line comparison use identical AER centers, strengths, lower-state energies,
and air/self broadening parameters. LBLRTM's separate coupling and
speed-dependent files are deliberately excluded and remain measurable
residual terms.
