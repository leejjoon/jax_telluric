"""Differentiable corrections derived from offline LBLRTM calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import jax.numpy as jnp
import numpy as np

from .lblrtm import LBLRTMRunConfig, run_lblrtm
from .types import AtmosphereProfile


def _optical_depth_on_grid(spectrum, target_wavenumber_cm1: np.ndarray) -> np.ndarray:
    source_nu = np.asarray(spectrum.wavenumber_cm1, dtype=float)
    target_nu = np.asarray(target_wavenumber_cm1, dtype=float)
    if source_nu[0] > target_nu[0] or source_nu[-1] < target_nu[-1]:
        raise ValueError("LBLRTM spectrum does not cover the correction grid")
    transmission = np.clip(np.asarray(spectrum.transmission, dtype=float), np.finfo(np.float32).tiny, None)
    return np.interp(target_nu, source_nu, -np.log(transmission))


@dataclass(frozen=True)
class LBLRTMOpticalDepthCorrection:
    """Fixed-profile MT_CKD continua and empirical line corrections.

    The template stores vertical optical depths on the model grid. H2O self
    continuum scales quadratically with its global column scale; foreign
    continuum and per-species line residuals scale linearly. The residuals
    measure all remaining differences from the selected JAX opacity backend.
    They must not be interpreted as one particular omitted physical effect.
    """

    wavenumber_cm1: np.ndarray
    pressure_layer_bar: np.ndarray
    temperature_k: np.ndarray
    air_column_cm2: np.ndarray
    reference_vmr: Mapping[str, np.ndarray]
    water_self_optical_depth: np.ndarray
    water_foreign_optical_depth: np.ndarray
    reference_background_optical_depth: np.ndarray
    line_residual_optical_depth: Mapping[str, np.ndarray]
    requires_pressure_shift: bool = False

    def __post_init__(self) -> None:
        nu = np.asarray(self.wavenumber_cm1, dtype=float)
        pressure = np.asarray(self.pressure_layer_bar, dtype=float)
        temperature = np.asarray(self.temperature_k, dtype=float)
        air_column = np.asarray(self.air_column_cm2, dtype=float)
        self_tau = np.asarray(self.water_self_optical_depth, dtype=float)
        foreign_tau = np.asarray(self.water_foreign_optical_depth, dtype=float)
        background_tau = np.asarray(self.reference_background_optical_depth, dtype=float)
        vmr = {name.upper(): np.asarray(value, dtype=float) for name, value in self.reference_vmr.items()}
        residual = {
            name.upper(): np.asarray(value, dtype=float)
            for name, value in self.line_residual_optical_depth.items()
        }
        if nu.ndim != 1 or len(nu) < 2 or np.any(np.diff(nu) <= 0.0):
            raise ValueError("correction wavenumbers must be increasing")
        if pressure.shape != temperature.shape or pressure.shape != air_column.shape:
            raise ValueError("correction profile arrays must share one layer shape")
        if any(value.shape != pressure.shape for value in vmr.values()):
            raise ValueError("reference VMR arrays must match the correction profile")
        if self_tau.shape != nu.shape or foreign_tau.shape != nu.shape or background_tau.shape != nu.shape:
            raise ValueError("continuum corrections must match the wavenumber grid")
        if any(value.shape != nu.shape for value in residual.values()):
            raise ValueError("line corrections must match the wavenumber grid")
        arrays = [
            nu, pressure, temperature, air_column, self_tau, foreign_tau, background_tau,
            *vmr.values(), *residual.values(),
        ]
        if any(np.any(~np.isfinite(value)) for value in arrays):
            raise ValueError("correction arrays must be finite")
        if "H2O" not in vmr or np.sum(air_column * vmr["H2O"]) <= 0.0:
            raise ValueError("LBLRTM corrections require a positive H2O reference column")
        if np.any(self_tau < 0.0) or np.any(foreign_tau < 0.0):
            raise ValueError("continuum optical depths must be nonnegative")
        if set(residual) - set(vmr):
            raise ValueError("line corrections require matching reference VMR profiles")
        object.__setattr__(self, "wavenumber_cm1", nu)
        object.__setattr__(self, "pressure_layer_bar", pressure)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "air_column_cm2", air_column)
        object.__setattr__(self, "reference_vmr", vmr)
        object.__setattr__(self, "water_self_optical_depth", self_tau)
        object.__setattr__(self, "water_foreign_optical_depth", foreign_tau)
        object.__setattr__(self, "reference_background_optical_depth", background_tau)
        object.__setattr__(self, "line_residual_optical_depth", residual)

    def _validate_structure(self, profile, wavenumber_cm1) -> None:
        checks = (
            (self.wavenumber_cm1, np.asarray(wavenumber_cm1)),
            (self.pressure_layer_bar, profile.pressure_layer_bar),
            (self.temperature_k, profile.temperature_k),
            (self.air_column_cm2, profile.air_column_cm2),
        )
        if any(left.shape != right.shape or not np.allclose(left, right, rtol=1e-12, atol=0.0)
               for left, right in checks):
            raise ValueError("LBLRTM correction does not match the model grid and atmospheric profile")

    def validate(self, profile: AtmosphereProfile, wavenumber_cm1: np.ndarray) -> None:
        """Reject full correction use outside its reference profile and grid."""
        self._validate_structure(profile, wavenumber_cm1)
        for species, reference in self.reference_vmr.items():
            if species not in profile.vmr or not np.allclose(reference, profile.vmr[species], rtol=1e-12, atol=0.0):
                raise ValueError(f"LBLRTM correction reference VMR does not match {species}")

    def validate_mt_ckd(self, profile: AtmosphereProfile, wavenumber_cm1: np.ndarray) -> None:
        """Validate only the profile fields used by the isolated H2O continua."""
        self._validate_structure(profile, wavenumber_cm1)
        if ("H2O" not in profile.vmr
                or not np.allclose(self.reference_vmr["H2O"], profile.vmr["H2O"],
                                   rtol=1e-12, atol=0.0)):
            raise ValueError("LBLRTM correction reference VMR does not match H2O")

    def validate_opacity(self, opacity) -> None:
        """Reject a backend incompatible with the correction's line baseline."""
        if not self.requires_pressure_shift:
            return
        calculators = getattr(opacity, "calculators", None)
        if calculators is None:
            raise ValueError("this LBLRTM correction requires a pressure-shifted opacity backend")
        incompatible = [
            species for species, calculator in calculators.items()
            if species in self.line_residual_optical_depth
            and not getattr(calculator, "pressure_shift", False)
        ]
        if incompatible:
            raise ValueError(
                "this LBLRTM correction requires pressure_shift=True for: "
                + ", ".join(sorted(incompatible))
            )

    def _column_scale(self, species: str, scaled_vmr: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        reference = jnp.asarray(self.reference_vmr[species])
        weights = jnp.asarray(self.air_column_cm2)
        denominator = jnp.sum(weights * reference)
        return jnp.sum(weights * scaled_vmr[species]) / denominator

    @property
    def species(self) -> tuple[str, ...]:
        """Molecules represented by the LBLRTM line-residual templates."""
        return tuple(self.line_residual_optical_depth)

    def _mt_ckd_total(self, scaled_vmr):
        h2o_scale = self._column_scale("H2O", scaled_vmr)
        return (
            jnp.asarray(self.water_self_optical_depth) * h2o_scale**2
            + jnp.asarray(self.water_foreign_optical_depth) * h2o_scale
        )

    def mt_ckd_optical_depth(self, profile, scaled_vmr):
        """Return only the isolated H2O self and foreign MT_CKD terms."""
        total = self._mt_ckd_total(scaled_vmr)
        layers = jnp.zeros((len(profile.temperature_k), total.size), dtype=total.dtype)
        return layers.at[0].set(total)

    def optical_depth(self, profile, scaled_vmr):
        total = self._mt_ckd_total(scaled_vmr) + jnp.asarray(
            self.reference_background_optical_depth
        )
        for species, residual in self.line_residual_optical_depth.items():
            total = total + jnp.asarray(residual) * self._column_scale(species, scaled_vmr)
        layers = jnp.zeros((len(profile.temperature_k), total.size), dtype=total.dtype)
        return layers.at[0].set(total)

    def save(self, path: str | Path) -> None:
        """Save a portable correction template."""
        values = {
            "format_version": np.asarray(2),
            "requires_pressure_shift": np.asarray(self.requires_pressure_shift),
            "wavenumber_cm1": self.wavenumber_cm1,
            "pressure_layer_bar": self.pressure_layer_bar,
            "temperature_k": self.temperature_k,
            "air_column_cm2": self.air_column_cm2,
            "water_self_optical_depth": self.water_self_optical_depth,
            "water_foreign_optical_depth": self.water_foreign_optical_depth,
            "reference_background_optical_depth": self.reference_background_optical_depth,
        }
        values.update({f"reference_vmr__{name}": value for name, value in self.reference_vmr.items()})
        values.update({f"line_residual__{name}": value for name, value in self.line_residual_optical_depth.items()})
        np.savez_compressed(path, **values)

    @classmethod
    def load(cls, path: str | Path) -> "LBLRTMOpticalDepthCorrection":
        """Load a correction template created by :meth:`save`."""
        with np.load(path) as values:
            version = int(values["format_version"])
            if version not in (1, 2):
                raise ValueError("unsupported LBLRTM correction format")
            vmr = {key.split("__", 1)[1]: values[key] for key in values.files if key.startswith("reference_vmr__")}
            residual = {key.split("__", 1)[1]: values[key] for key in values.files if key.startswith("line_residual__")}
            return cls(
                values["wavenumber_cm1"], values["pressure_layer_bar"], values["temperature_k"],
                values["air_column_cm2"], vmr, values["water_self_optical_depth"],
                values["water_foreign_optical_depth"], values["reference_background_optical_depth"], residual,
                bool(values["requires_pressure_shift"]) if version >= 2 else False,
            )


def build_lblrtm_correction(
    workdir: str | Path,
    profile: AtmosphereProfile,
    wavenumber_cm1: np.ndarray,
    opacity,
    executable: str | Path,
    tape3: str | Path,
    mt_ckd_data: str | Path,
) -> LBLRTMOpticalDepthCorrection:
    """Generate a fixed-profile correction using offline LBLRTM runs.

    Three full-atmosphere runs isolate MT_CKD H2O self and foreign continua.
    One continuum-free, species-only run per profile molecule is compared with
    the JAX backend to capture the residual line physics.
    """
    nu = np.asarray(wavenumber_cm1, dtype=float)
    root = Path(workdir)

    def run(run_profile, continuum_flag: int, name: str):
        config = LBLRTMRunConfig(float(nu[0]), float(nu[-1]), continuum_flag=continuum_flag,
                                 description=f"jax-telluric correction {name}")
        spectrum = run_lblrtm(root / name, run_profile, config, executable, tape3, mt_ckd_data)
        return _optical_depth_on_grid(spectrum, nu)

    all_tau = run(profile, 1, "continuum_all")
    no_self_tau = run(profile, 2, "continuum_no_self")
    no_foreign_tau = run(profile, 3, "continuum_no_foreign")
    water_self = np.maximum(all_tau - no_self_tau, 0.0)
    water_foreign = np.maximum(all_tau - no_foreign_tau, 0.0)

    pressure = jnp.asarray(profile.pressure_layer_bar)
    partial_pressure = {
        species: pressure * jnp.asarray(profile.vmr[species]) for species in opacity.species
    }
    cross_sections = opacity.cross_sections(jnp.asarray(profile.temperature_k), pressure, partial_pressure)
    line_residual = {}
    correction_species = tuple(profile.vmr)
    for species in correction_species:
        if species not in profile.vmr or not np.any(np.asarray(profile.vmr[species]) > 0.0):
            raise ValueError(f"cannot build a {species} correction with zero reference abundance")
        species_profile = AtmosphereProfile(
            profile.pressure_edges_bar, profile.temperature_k, profile.altitude_km,
            {species: profile.vmr[species]}, profile.mean_molecular_weight_g_mol, profile.gravity_m_s2,
        )
        lblrtm_tau = run(species_profile, 0, f"lines_{species.lower()}")
        if species in cross_sections:
            jax_tau = np.sum(
                np.asarray(cross_sections[species])
                * (profile.air_column_cm2 * np.asarray(profile.vmr[species]))[:, None],
                axis=0,
            )
        else:
            jax_tau = np.zeros_like(lblrtm_tau)
        line_residual[species] = lblrtm_tau - jax_tau

    jax_line_total = sum(
        np.sum(
            np.asarray(cross_sections[species])
            * (profile.air_column_cm2 * np.asarray(profile.vmr[species]))[:, None],
            axis=0,
        )
        for species in opacity.species
    )
    background = (
        all_tau - jax_line_total - water_self - water_foreign
        - sum(line_residual.values())
    )

    return LBLRTMOpticalDepthCorrection(
        nu, profile.pressure_layer_bar, profile.temperature_k, profile.air_column_cm2,
        {species: profile.vmr[species] for species in correction_species},
        water_self, water_foreign, background, line_residual, requires_pressure_shift=True,
    )
