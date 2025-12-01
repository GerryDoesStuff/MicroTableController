import numpy as np

from microstage_app.spectroscopy.processing import (
    apply_baseline,
    apply_response_correction,
    compute_absorbance,
    compute_irradiance,
    compute_reflectance,
    compute_transmittance,
    detect_saturation,
    normalize_reference,
    raman_shift_cm,
    smooth_boxcar,
    subtract_dark,
)


def test_processing_pipeline(wavelengths, raw_spectrum, dark_spectrum, reference_spectrum, response_curve):
    smoothed = smooth_boxcar(raw_spectrum, window=3)
    assert smoothed.shape == raw_spectrum.shape

    dark_corrected = subtract_dark(raw_spectrum, dark_spectrum)
    assert np.all(dark_corrected >= 0)

    normalized = normalize_reference(dark_corrected, reference_spectrum)
    assert np.isfinite(normalized).all()

    absorbance = compute_absorbance(raw_spectrum, reference_spectrum)
    assert (absorbance > 0).all()

    transmittance = compute_transmittance(raw_spectrum, reference_spectrum)
    reflectance = compute_reflectance(raw_spectrum, reference_spectrum)
    assert np.allclose(transmittance, reflectance)

    corrected = apply_response_correction(raw_spectrum, response_curve)
    assert corrected.shape == raw_spectrum.shape
    assert (corrected >= 0).all()

    irradiance = compute_irradiance(raw_spectrum, integration_time_ms=10.0, response_curve=response_curve)
    assert irradiance.shape == raw_spectrum.shape
    assert irradiance.max() > raw_spectrum.max()

    raman = raman_shift_cm(wavelengths, excitation_nm=532.0)
    assert np.all(np.diff(raman) > 0)

    baseline_corrected = apply_baseline(raw_spectrum, baseline_fn=lambda arr: np.ones_like(arr))
    assert np.allclose(baseline_corrected, raw_spectrum - 1)

    saturation_mask = detect_saturation(raw_spectrum, max_value=raw_spectrum.max() - 1)
    assert saturation_mask.any()
