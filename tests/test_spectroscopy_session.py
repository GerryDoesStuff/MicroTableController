import numpy as np
import pytest

from microstage_app.spectroscopy.session import SpectroscopySession


def test_session_calibration_invalidation(wavelengths, raw_spectrum, reference_spectrum):
    session = SpectroscopySession()
    session.set_wavelengths(wavelengths)
    session.set_acquisition_context("dev1", 10.0, 1, "Absorbance", timestamp=0.0)
    session.set_raw(raw_spectrum)
    session.set_reference(reference_spectrum)

    session.set_calibration(np.ones_like(wavelengths), created_with="mock")
    assert session.calibration_valid is True
    assert session.calibration.metadata["created_with"] == "mock"

    session.set_wavelengths(wavelengths[:-1])
    assert session.requires_recalibration()

    with pytest.raises(ValueError):
        session.set_calibration(np.ones_like(wavelengths))


def test_session_ready_state(wavelengths, raw_spectrum, dark_spectrum):
    session = SpectroscopySession()
    assert not session.is_ready_for_processing()

    session.set_wavelengths(wavelengths)
    session.set_raw(raw_spectrum)
    assert session.is_ready_for_processing()

    with pytest.raises(ValueError):
        session.set_dark(dark_spectrum[:-1])

    session.set_dark(dark_spectrum)
    assert session.is_ready_for_processing()


def test_roi_management():
    session = SpectroscopySession()
    session.add_roi(500, 600, label="band1")
    session.add_roi(450, 480)
    assert len(session.rois) == 2
    assert session.rois[0].label == "band1"

    session.remove_roi(0)
    assert len(session.rois) == 1
    session.clear_rois()
    assert session.rois == []


def test_calibration_invalidation_on_inputs(wavelengths, raw_spectrum, dark_spectrum, reference_spectrum):
    session = SpectroscopySession()
    session.set_wavelengths(wavelengths)
    session.set_acquisition_context("dev1", 10.0, 1, "Absorbance", timestamp=0.0)
    session.set_calibration(np.ones_like(wavelengths))
    assert session.calibration_valid

    session.set_dark(dark_spectrum)
    assert not session.calibration_valid

    session.set_reference(reference_spectrum)
    assert not session.calibration_valid

    session.set_mode_params(exposure=10)
    assert not session.calibration_valid


def test_roi_mapping_and_ordering():
    session = SpectroscopySession()
    roi = session.add_roi(620, 580)
    assert roi.as_tuple() == (580.0, 620.0)
    session.add_roi(500, 550, label="mid")
    assert [r.label for r in session.rois] == ["", "mid"]


def test_acquisition_metadata_and_invalidation(wavelengths, dark_spectrum, reference_spectrum, response_curve):
    session = SpectroscopySession()
    session.set_wavelengths(wavelengths)
    session.set_acquisition_context("devA", 5.0, 2, "Absorbance", timestamp=1.0)
    session.set_dark(dark_spectrum)
    session.set_reference(reference_spectrum)
    session.set_calibration(response_curve)

    assert session.dark_metadata and session.dark_metadata.device_id == "devA"
    assert session.reference_metadata and session.reference_metadata.averages == 2
    assert not session.requires_recalibration()

    session.set_acquisition_context("devB", 5.0, 2, "Absorbance", timestamp=2.0)
    assert session.requires_recalibration()
    assert session.dark_valid is False
    assert session.reference_valid is False
    assert session.calibration_valid is False
