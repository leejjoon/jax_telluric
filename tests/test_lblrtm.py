from pathlib import Path
import hashlib
import json

import numpy as np

from jax_telluric import (
    LBLRTMRunConfig,
    LBLRTMSpectrum,
    compare_transmission,
    degrade_to_resolving_power,
    load_atmosphere_csv,
    write_tape5,
)
from jax_telluric import lblrtm as lblrtm_module


def test_tape5_writer_uses_requested_range_profile_and_continuum(tmp_path):
    profile = load_atmosphere_csv("data/profiles/example_midlatitude.csv")
    output = tmp_path / "TAPE5"
    write_tape5(output, profile, LBLRTMRunConfig(5000.0, 5100.0, 30.0))
    lines = output.read_text().splitlines()
    assert lines[1][14] == "1"
    assert float(lines[2][:10]) == 5000.0
    assert float(lines[2][10:20]) == 5100.0
    assert any("AAAAAAA" in line for line in lines)
    assert lines[-1] == "%"


def test_degrade_and_compare_reference_spectrum():
    nu = np.linspace(5000.0, 5002.0, 4001)
    flux = 1.0 - 0.5 * np.exp(-0.5 * ((nu - 5001.0) / 0.01) ** 2)
    reference = degrade_to_resolving_power(LBLRTMSpectrum(nu, flux))
    metrics = compare_transmission(reference, reference)
    assert metrics.median_absolute_error == 0.0
    assert metrics.percentile_99_absolute_error == 0.0
    assert abs(metrics.line_shift_resolution_elements) < 1.0e-12


def test_run_lblrtm_accepts_relative_workdir(tmp_path, monkeypatch):
    profile = load_atmosphere_csv("data/profiles/example_midlatitude.csv")
    fake_spectrum = LBLRTMSpectrum(np.arange(8.0), np.ones(8))

    def fake_run(command, cwd, capture_output, text):
        assert Path(command[0]).is_absolute()
        (Path(cwd) / "TAPE12").write_bytes(b"placeholder")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(lblrtm_module.subprocess, "run", fake_run)
    monkeypatch.setattr(lblrtm_module, "read_tape12_single_precision", lambda path: fake_spectrum)
    executable = tmp_path / "source-lblrtm"
    tape3 = tmp_path / "source-TAPE3"
    mt_ckd = tmp_path / "source-mt-ckd.nc"
    for path in (executable, tape3, mt_ckd):
        path.write_bytes(b"x")
    actual = lblrtm_module.run_lblrtm(
        tmp_path / "relative-run", profile, LBLRTMRunConfig(5000.0, 5001.0), executable, tape3, mt_ckd
    )
    assert actual is fake_spectrum


def test_committed_lblrtm_fixture_has_valid_ranges():
    fixture = Path("tests/data/lblrtm_k_5000_5100.npz")
    metadata = json.loads(fixture.with_suffix(".json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == metadata["sha256"]
    with np.load(fixture) as data:
        assert np.all(np.diff(data["wavenumber_cm1"]) > 0.0)
        assert np.all((data["transmission"] >= 0.0) & (data["transmission"] <= 1.0))
        assert {
            "transmission_no_self",
            "transmission_no_foreign",
            "transmission_no_water_continuum",
        } < set(data.files)


def test_recorded_aer_co_validation_meets_mvp_thresholds():
    result = json.loads(Path("tests/data/aer_co_validation.json").read_text())
    metrics = result["metrics"]
    thresholds = result["thresholds"]
    assert metrics["median_absolute_error"] < thresholds["median_absolute_error"]
    assert metrics["percentile_99_absolute_error"] < thresholds["percentile_99_absolute_error"]
    assert abs(metrics["line_shift_resolution_elements"]) < thresholds[
        "absolute_line_shift_resolution_elements"
    ]
