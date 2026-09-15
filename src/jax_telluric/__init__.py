"""Differentiable terrestrial telluric transmission."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .fit import FitResult, fit_order
from .aer import AERLineDatabase
from .exojax_backend import ExoJAXOpacityBackend
from .io import load_atmosphere_csv, load_mipas_profile
from .lblrtm import LBLRTMRunConfig, run_lblrtm, write_tape5
from .reference import (
    LBLRTMSpectrum,
    ValidationMetrics,
    compare_transmission,
    degrade_to_resolving_power,
    read_tape12_single_precision,
)
from .model import (
    ArrayOpacityBackend,
    ReferenceWaterContinuum,
    TelluricModel,
    igrins_wavenumber_grid,
)
from .types import AtmosphereProfile, SpectralOrder, TelluricParameters

__all__ = [
    "ArrayOpacityBackend",
    "AERLineDatabase",
    "AtmosphereProfile",
    "ExoJAXOpacityBackend",
    "FitResult",
    "LBLRTMSpectrum",
    "LBLRTMRunConfig",
    "SpectralOrder",
    "ReferenceWaterContinuum",
    "TelluricModel",
    "TelluricParameters",
    "ValidationMetrics",
    "compare_transmission",
    "degrade_to_resolving_power",
    "fit_order",
    "igrins_wavenumber_grid",
    "load_atmosphere_csv",
    "load_mipas_profile",
    "read_tape12_single_precision",
    "run_lblrtm",
    "write_tape5",
]
