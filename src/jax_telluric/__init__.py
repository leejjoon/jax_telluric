"""Differentiable terrestrial telluric transmission."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .fit import FitResult, OrderObjective, fit_order
from .corrections import LBLRTMOpticalDepthCorrection, build_lblrtm_correction
from .aer import (
    AERLineDatabase,
    line_optical_depth_bound,
    select_significant_lines,
)
from .atlas import (
    ArcturusPage,
    arcturus_spectral_order,
    epoch_velocity_kms,
    read_arcturus_page,
    robust_noise,
)
from .exojax_backend import ExoJAXOpacityBackend
from .download import default_data_directory, download_aer_lines, download_mt_ckd
from .io import load_atmosphere_csv, load_mipas_profile
from .lblrtm import LBLRTMRunConfig, run_lblrtm, write_tape5
from .mt_ckd import MTCKDWaterContinuum
from .reference import (
    LBLRTMSpectrum,
    ValidationMetrics,
    compare_transmission,
    degrade_to_resolving_power,
    read_tape12_single_precision,
)
from .model import (
    ArrayOpacityBackend,
    LinearizedOpacityBackend,
    BoxcarFTSInstrumentProfile,
    ReferenceWaterContinuum,
    TelluricModel,
    igrins_wavenumber_grid,
    trim_wavenumber_grid,
)
from .stellar import (
    StellarSpectrum,
    broaden_stellar_source,
    prepare_stellar_source,
    resample_stellar_source,
)
from .types import AtmosphereProfile, SpectralOrder, TelluricParameters

__all__ = [
    "ArrayOpacityBackend",
    "LinearizedOpacityBackend",
    "AERLineDatabase",
    "ArcturusPage",
    "AtmosphereProfile",
    "BoxcarFTSInstrumentProfile",
    "ExoJAXOpacityBackend",
    "FitResult",
    "LBLRTMSpectrum",
    "LBLRTMOpticalDepthCorrection",
    "LBLRTMRunConfig",
    "MTCKDWaterContinuum",
    "SpectralOrder",
    "StellarSpectrum",
    "ReferenceWaterContinuum",
    "TelluricModel",
    "TelluricParameters",
    "ValidationMetrics",
    "arcturus_spectral_order",
    "broaden_stellar_source",
    "compare_transmission",
    "default_data_directory",
    "download_aer_lines",
    "download_mt_ckd",
    "build_lblrtm_correction",
    "degrade_to_resolving_power",
    "epoch_velocity_kms",
    "OrderObjective",
    "fit_order",
    "igrins_wavenumber_grid",
    "trim_wavenumber_grid",
    "load_atmosphere_csv",
    "load_mipas_profile",
    "prepare_stellar_source",
    "read_arcturus_page",
    "read_tape12_single_precision",
    "resample_stellar_source",
    "robust_noise",
    "run_lblrtm",
    "write_tape5",
]
