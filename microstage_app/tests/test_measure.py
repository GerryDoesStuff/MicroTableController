import numpy as np

from microstage_app.analysis import centroid, find_contours, measure_area, measure_distance
import pytest


def test_measure_distance():
    assert measure_distance((0, 0), (3, 4), 0.5) == 2.5


def test_measure_area_and_centroid():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    assert measure_area(mask, 0.1) == pytest.approx(4 * 0.01)
    assert centroid(mask) == (1.5, 1.5)


def test_find_contours_single():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    contours = find_contours(mask)
    assert len(contours) == 1
