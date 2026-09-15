import struct

import numpy as np

from jax_telluric import read_tape12_single_precision


def test_read_single_precision_tape12_panel(tmp_path):
    content = bytearray(1068)
    content.extend(struct.pack("=ddfl", 4300.0, 4300.3, 0.1, 3))
    content.extend(bytes(16))
    content.extend(struct.pack("=3f", 0.0, 0.0, 0.0))
    content.extend(struct.pack("=3f", 0.9, 0.8, 0.7))
    content.extend(bytes(8))
    tape12 = tmp_path / "TAPE12"
    tape12.write_bytes(content)

    spectrum = read_tape12_single_precision(tape12)
    np.testing.assert_allclose(spectrum.wavenumber_cm1, [4300.0, 4300.1, 4300.2])
    np.testing.assert_allclose(spectrum.transmission, [0.9, 0.8, 0.7], rtol=1e-6)
