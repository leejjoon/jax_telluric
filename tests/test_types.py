import numpy as np
import pytest

from jax_telluric import AtmosphereProfile, SpectralOrder, load_atmosphere_csv, load_mipas_profile


def test_profile_rejects_pressure_in_wrong_order():
    with pytest.raises(ValueError, match="increase"):
        AtmosphereProfile(
            pressure_edges_bar=[1.0, 0.1],
            temperature_k=[280.0],
            altitude_km=[2.0],
            vmr={"H2O": [0.001]},
        )


def test_order_defaults_mask_and_source():
    order = SpectralOrder([1000.0, 1000.1, 1000.2], [1.0] * 3, [0.1] * 3)
    np.testing.assert_array_equal(order.mask, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(order.source_flux, np.ones(3))


def test_example_profile_loads_all_target_species():
    profile = load_atmosphere_csv("data/profiles/example_midlatitude.csv")
    assert len(profile.temperature_k) == 6
    assert set(profile.vmr) == {"H2O", "CO2", "CH4", "O2", "CO", "N2O"}
    assert np.all(profile.air_column_cm2 > 0.0)


def test_mipas_profile_is_converted_from_levels_to_top_down_layers(tmp_path):
    mipas = tmp_path / "profile.atm"
    mipas.write_text(
        """3 ! levels
*HGT [km]
0 2 10
*PRE [hPa]
1000 800 250
*TEM [K]
290 275 230
*H2O [ppmv]
10000 5000 100
*END
"""
    )
    profile = load_mipas_profile(mipas, observatory_altitude_km=2.0, species=("H2O",))
    np.testing.assert_allclose(profile.pressure_edges_bar, [0.25, 0.8])
    np.testing.assert_allclose(profile.temperature_k, [252.5])
    np.testing.assert_allclose(profile.vmr["H2O"], [0.00255])
