import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a minimal cv2 stub so the measure module can be imported
cv2_stub = types.SimpleNamespace(
    RETR_EXTERNAL=0,
    CHAIN_APPROX_SIMPLE=0,
    findContours=lambda *args, **kwargs: ([], None),
    moments=lambda *args, **kwargs: {"m00": 0, "m10": 0, "m01": 0},
)
sys.modules.setdefault("cv2", cv2_stub)

from microstage_app.analysis import measure_distance, measure_area


def test_measure_distance():
    """Distance between two points scales with pixel size."""
    assert measure_distance((0, 0), (3, 4), 0.5) == pytest.approx(2.5)


def test_measure_area_square():
    """Area of a filled rectangle should match pixel count."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 7:12] = 1  # 10x5 rectangle -> 50 pixels
    pixel_size = 0.2
    expected = 50 * pixel_size**2
    assert measure_area(mask, pixel_size) == pytest.approx(expected)


def test_measure_area_circle():
    """Area of a disk approximates analytic area."""
    size = 50
    radius = 10
    y, x = np.ogrid[:size, :size]
    mask = ((x - 25) ** 2 + (y - 25) ** 2 <= radius ** 2).astype(np.uint8)
    pixel_size = 0.2
    expected = np.pi * radius**2 * pixel_size**2
    assert measure_area(mask, pixel_size) == pytest.approx(expected, rel=0.02)
