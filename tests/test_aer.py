from jax_telluric import AERLineDatabase
import pytest


def _line(nu, isotope=1):
    gamma_air = f"{0.07:.4f}"[1:]
    gamma_self = f"{0.08:.4f}"[1:]
    return (
        f"{5:2d}{isotope:1d}{nu:12.6f}{1.0e-22:10.3E}{2.0e-5:10.3E}"
        f"{gamma_air}{gamma_self}{100.0:10.4f}{0.7:4.2f}{-0.001:8.6f}"
    )


@pytest.mark.parametrize("header", ["> header\n%%%%%%%%\n", ""])
def test_aer_database_reads_only_selected_lines(tmp_path, header):
    path = tmp_path / "05_CO"
    path.write_text(header + _line(4299.0) + "\n" + _line(4301.0) + "\n")
    database = AERLineDatabase(path, "CO", (4300.5, 4301.5), margin_cm1=0.0)
    assert database.simple_molecule_name == "CO"
    assert database.nu_lines.tolist() == [4301.0]
    assert database.gamma_self.tolist() == [0.08]
    assert database.qr_interp(0, 296.0, 296.0).tolist() == [1.0]
