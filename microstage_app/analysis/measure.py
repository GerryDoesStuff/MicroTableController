"""Utility functions for basic image-based measurements.

This module provides helper functions to convert pixel measurements to
real-world units using a known ``pixel_size``.  Additional convenience
functions leverage :mod:`cv2` for contour analysis.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def measure_distance(p1: Sequence[float], p2: Sequence[float], pixel_size: float) -> float:
    """Return the real-world distance between two pixel coordinates.

    Parameters
    ----------
    p1, p2:
        Two-element sequences representing ``(x, y)`` positions in pixels.
    pixel_size:
        The physical size of a single pixel in the desired units.

    Returns
    -------
    float
        The distance between ``p1`` and ``p2`` in real-world units.
    """
    p1_arr = np.asarray(p1, dtype=float)
    p2_arr = np.asarray(p2, dtype=float)
    if p1_arr.shape[-1] != 2 or p2_arr.shape[-1] != 2:
        raise ValueError("p1 and p2 must be 2D coordinates")
    dist_pixels = np.linalg.norm(p1_arr - p2_arr)
    return float(dist_pixels * pixel_size)


def measure_area(mask: np.ndarray, pixel_size: float) -> float:
    """Compute the real-world area of a binary mask.

    Parameters
    ----------
    mask:
        2-D binary image.  Non-zero values are treated as foreground.
    pixel_size:
        The physical size of a single pixel in the desired units.

    Returns
    -------
    float
        The area covered by ``mask`` in square units.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    pixel_area = float(pixel_size) ** 2
    num_pixels = int(np.count_nonzero(mask))
    return num_pixels * pixel_area


def find_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Find contours in a binary mask using :func:`cv2.findContours`.

    Parameters
    ----------
    mask:
        2-D binary image.

    Returns
    -------
    list of ``numpy.ndarray``
        Contours found in the mask.
    """
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return the centroid of a binary mask in pixel coordinates.

    The centroid is calculated from image moments.  ``None`` is returned if
    the mask contains no foreground pixels.
    """
    mask_u8 = mask.astype(np.uint8)
    m = cv2.moments(mask_u8)
    if m["m00"] == 0:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return float(cx), float(cy)


__all__ = [
    "measure_distance",
    "measure_area",
    "find_contours",
    "centroid",
]
