import numpy as np
import pytest

from microstage_app.spectroscopy.session import SpectroscopySession


def test_session_calibration_invalidation(wavelengths, raw_spectrum, reference_spectrum):
    session = SpectroscopySession()
    session.set_wavelengths(wavelengths)
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
