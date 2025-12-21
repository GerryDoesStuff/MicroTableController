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


def test_processing_edge_cases(wavelengths):
    dark = np.zeros_like(wavelengths)
    ref = np.ones_like(wavelengths)
    raw = np.ones_like(wavelengths)

    with np.testing.assert_raises(ValueError):
        subtract_dark(raw, dark[:-1])

    absorbance = compute_absorbance(sample=raw, reference=ref * 0.0)
    assert np.isfinite(absorbance).all()

    corrected = apply_baseline(raw, baseline_fn=lambda arr: arr * 0.5)
    assert np.allclose(corrected, raw * 0.5)


def test_absorbance_with_dark_subtraction():
    sample = np.array([10.0, 20.0, 30.0])
    reference = np.array([100.0, 200.0, 300.0])
    dark = np.array([1.0, 2.0, 3.0])

    expected = np.log10((reference - dark) / (sample - dark))
    absorbance = compute_absorbance(sample, reference, dark=dark)

    assert np.allclose(absorbance, expected)


def test_beer_lambert_fit_consistency(wavelengths):
    ref = np.ones_like(wavelengths) * 100.0
    concentrations = np.array([0.1, 0.5, 1.0])
    path_length = 1.0
    epsilon = 0.2

    absorbances = []
    for c in concentrations:
        sample = ref * np.exp(-epsilon * path_length * c)
        absorbance = compute_absorbance(sample, ref)
        absorbances.append(absorbance.mean())

    slope, intercept = np.polyfit(concentrations, absorbances, 1)
    assert slope > 0
    assert abs(intercept) < 1e-3


def test_irradiance_and_color_metrics(raw_spectrum, response_curve):
    irradiance = compute_irradiance(raw_spectrum, integration_time_ms=20.0, response_curve=response_curve)
    assert np.all(irradiance > 0)
    total_energy = np.trapz(irradiance)
    assert total_energy > 0

    faster = compute_irradiance(raw_spectrum, integration_time_ms=10.0, response_curve=response_curve)
    assert faster.mean() > irradiance.mean()


def test_raman_shift_and_validation(wavelengths, raw_spectrum):
    shift = raman_shift_cm(wavelengths, excitation_nm=785.0)
    assert np.all(np.diff(shift) > 0)

    with np.testing.assert_raises(ValueError):
        raman_shift_cm(wavelengths, excitation_nm=0)

    baseline_removed = apply_baseline(raw_spectrum, baseline_fn=None)
    assert np.array_equal(baseline_removed, raw_spectrum)
