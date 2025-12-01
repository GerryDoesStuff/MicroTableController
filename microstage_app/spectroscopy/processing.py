from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.signal import convolve


def smooth_boxcar(spectrum: np.ndarray, window: int = 5) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return np.asarray(spectrum, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(spectrum, (window // 2, window - 1 - window // 2), mode="edge")
    smoothed = convolve(padded, kernel, mode="valid")
    return smoothed


def detect_saturation(spectrum: np.ndarray, max_value: float) -> np.ndarray:
    arr = np.asarray(spectrum, dtype=float)
    return arr >= float(max_value)


def subtract_dark(raw: np.ndarray, dark: np.ndarray) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=float)
    dark_arr = np.asarray(dark, dtype=float)
    if raw_arr.shape != dark_arr.shape:
        raise ValueError("raw and dark shapes must match")
    corrected = raw_arr - dark_arr
    return np.clip(corrected, 0.0, None)


def normalize_reference(signal: np.ndarray, reference: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    sig = np.asarray(signal, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if sig.shape != ref.shape:
        raise ValueError("signal and reference shapes must match")
    safe_ref = np.where(np.abs(ref) < epsilon, np.sign(ref) * epsilon + epsilon, ref)
    return sig / safe_ref


def compute_absorbance(sample: np.ndarray, reference: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    ratio = normalize_reference(reference, sample, epsilon=epsilon)
    return np.log10(np.clip(ratio, epsilon, None))


def compute_transmittance(sample: np.ndarray, reference: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    return normalize_reference(sample, reference, epsilon=epsilon)


def compute_reflectance(sample: np.ndarray, reference: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    return normalize_reference(sample, reference, epsilon=epsilon)


def apply_response_correction(signal: np.ndarray, response_curve: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    sig = np.asarray(signal, dtype=float)
    resp = np.asarray(response_curve, dtype=float)
    if sig.shape != resp.shape:
        raise ValueError("signal and response_curve shapes must match")
    safe_resp = np.where(np.abs(resp) < epsilon, np.sign(resp) * epsilon + epsilon, resp)
    return sig * safe_resp


def compute_irradiance(
    sample: np.ndarray,
    integration_time_ms: float,
    response_curve: np.ndarray | None = None,
    epsilon: float = 1e-9,
) -> np.ndarray:
    sample_arr = np.asarray(sample, dtype=float)
    if integration_time_ms <= 0:
        raise ValueError("integration_time_ms must be positive")
    irradiance = sample_arr / (integration_time_ms / 1000.0)
    if response_curve is not None:
        irradiance = apply_response_correction(irradiance, response_curve, epsilon=epsilon)
    return irradiance


def raman_shift_cm(wavelengths_nm: np.ndarray, excitation_nm: float, epsilon: float = 1e-9) -> np.ndarray:
    wl = np.asarray(wavelengths_nm, dtype=float)
    if excitation_nm <= 0:
        raise ValueError("excitation_nm must be positive")
    excitation_cm = 1e7 / excitation_nm
    return excitation_cm - np.where(wl == 0, epsilon, 1e7 / wl)


def apply_baseline(signal: np.ndarray, baseline_fn=None) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if baseline_fn is None:
        return arr
    baseline = np.asarray(baseline_fn(arr), dtype=float)
    if baseline.shape != arr.shape:
        raise ValueError("baseline shape must match signal")
    return arr - baseline


def median_baseline(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    return np.median(arr) * np.ones_like(arr)


def edge_baseline(signal: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    count = max(1, int(n * fraction))
    baseline_level = float(np.mean(np.concatenate([arr[:count], arr[-count:]])))
    return baseline_level * np.ones_like(arr)


def apply_mask_bands(
    x_axis: np.ndarray, signal: np.ndarray, bands: List[Tuple[float, float]]
) -> np.ndarray:
    if not bands:
        return signal
    arr = np.asarray(signal, dtype=float).copy()
    x_arr = np.asarray(x_axis, dtype=float)
    for start, end in bands:
        mask = (x_arr >= min(start, end)) & (x_arr <= max(start, end))
        arr[mask] = np.nan
    return arr


def beer_lambert_fit(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    if len(points) < 2:
        raise ValueError("At least two calibration points are required")
    conc = np.asarray([p[0] for p in points], dtype=float)
    absorb = np.asarray([p[1] for p in points], dtype=float)
    coeffs = np.polyfit(conc, absorb, 1)
    fit = np.poly1d(coeffs)
    residuals = absorb - fit(conc)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((absorb - np.mean(absorb)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coeffs[0]), float(coeffs[1]), r2


def cie_approx_metrics(wavelengths: np.ndarray, spectrum: np.ndarray) -> dict:
    wl = np.asarray(wavelengths, dtype=float)
    arr = np.asarray(spectrum, dtype=float)
    mask = (wl >= 380) & (wl <= 780)
    if not np.any(mask):
        return {}
    wl = wl[mask]
    arr = arr[mask]
    if np.all(arr <= 0):
        return {}
    normalized = arr / np.max(arr)
    centroid = float(np.average(wl, weights=normalized))
    weighted_mean = float(np.trapz(wl * normalized, wl) / max(np.trapz(normalized, wl), 1e-12))
    spread = float(np.sqrt(np.trapz(((wl - weighted_mean) ** 2) * normalized, wl) / max(np.trapz(normalized, wl), 1e-12)))
    return {"Centroid nm": centroid, "Dominant nm": weighted_mean, "Spectral spread": spread}
