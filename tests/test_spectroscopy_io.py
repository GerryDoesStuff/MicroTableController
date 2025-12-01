import numpy as np
import h5py

from microstage_app.spectroscopy.io import (
    save_spectrum_csv,
    save_spectrum_hdf5,
    save_spectrum_jcamp,
)


def _sample_data():
    wavelengths = np.array([400.0, 500.0, 600.0])
    processed = np.array([1.0, 2.0, 3.0])
    raw = np.array([10.0, 11.0, 12.0])
    metadata = {"mode": "Absorbance", "integration_ms": 100}
    return wavelengths, processed, raw, metadata


def test_save_spectrum_csv_with_raw(tmp_path):
    wavelengths, processed, raw, metadata = _sample_data()
    path = tmp_path / "spectrum.csv"

    save_spectrum_csv(
        path,
        wavelengths,
        processed,
        metadata,
        raw_counts=raw,
        include_raw=True,
    )

    content = path.read_text().splitlines()
    assert any("# mode: Absorbance" in line for line in content)
    table = np.loadtxt(path, delimiter=",", skiprows=len(metadata) + 1)
    np.testing.assert_allclose(table[:, 0], wavelengths)
    np.testing.assert_allclose(table[:, 1], processed)
    np.testing.assert_allclose(table[:, 2], raw)


def test_save_spectrum_hdf5_with_raw(tmp_path):
    wavelengths, processed, raw, metadata = _sample_data()
    path = tmp_path / "spectrum.h5"

    save_spectrum_hdf5(
        path,
        wavelengths,
        processed,
        metadata,
        raw_counts=raw,
        include_raw=True,
    )

    with h5py.File(path, "r") as h5:
        grp = h5["spectrum"]
        np.testing.assert_allclose(grp["wavelength_nm"], wavelengths)
        np.testing.assert_allclose(grp["intensity"], processed)
        np.testing.assert_allclose(grp["raw_counts"], raw)
        assert grp.attrs["mode"] == metadata["mode"]
        assert grp.attrs["integration_ms"] == metadata["integration_ms"]


def test_save_spectrum_jcamp_preserves_axes(tmp_path):
    wavelengths, processed, raw, metadata = _sample_data()
    path = tmp_path / "spectrum.jdx"

    save_spectrum_jcamp(
        path,
        wavelengths,
        processed,
        metadata,
        raw_counts=raw,
        include_raw=True,
    )

    lines = path.read_text().splitlines()
    assert "##$mode= Absorbance" in lines
    assert "##XYDATA= (X Y)" in lines
    assert "##RAW_COUNTS= (X Y)" in lines

    def _parse_block(start_marker: str):
        idx = lines.index(start_marker)
        block = lines[idx + 1 : idx + 1 + len(wavelengths)]
        return np.array([tuple(map(float, row.split())) for row in block])

    processed_block = _parse_block("##XYDATA= (X Y)")
    raw_block = _parse_block("##RAW_COUNTS= (X Y)")

    np.testing.assert_allclose(processed_block[:, 0], wavelengths)
    np.testing.assert_allclose(processed_block[:, 1], processed)
    np.testing.assert_allclose(raw_block[:, 0], wavelengths)
    np.testing.assert_allclose(raw_block[:, 1], raw)
