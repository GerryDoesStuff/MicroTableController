import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

# Ensure repository root on import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from microstage_app.control.autofocus import metric_value, FocusMetric


def test_metric_value_cpu(monkeypatch):
    monkeypatch.setattr(cv2.cuda, "getCudaEnabledDeviceCount", lambda: 0)
    img = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    expected_lap = cv2.Laplacian(img, cv2.CV_64F).var()
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    expected_ten = np.mean(gx * gx + gy * gy)
    assert metric_value(img, FocusMetric.LAPLACIAN) == pytest.approx(expected_lap)
    assert metric_value(img, FocusMetric.TENENGRAD) == pytest.approx(expected_ten)


def test_metric_value_gpu(monkeypatch):
    class FakeGpuMat:
        def __init__(self, mat=None):
            self.mat = mat
        def upload(self, arr):
            self.mat = arr.copy()
        def download(self):
            return self.mat
    def fake_create_laplacian(src_type, dst_type, ksize=1, scale=1, delta=0, borderType=None):
        class Filter:
            def apply(self, gm):
                return FakeGpuMat(cv2.Laplacian(gm.mat, dst_type, ksize=ksize))
        return Filter()
    def fake_create_sobel(src_type, dst_type, dx, dy, ksize=3, scale=1, delta=0, borderType=None):
        class Filter:
            def apply(self, gm):
                return FakeGpuMat(cv2.Sobel(gm.mat, dst_type, dx, dy, ksize=ksize))
        return Filter()
    monkeypatch.setattr(cv2, "cuda_GpuMat", FakeGpuMat)
    monkeypatch.setattr(cv2.cuda, "getCudaEnabledDeviceCount", lambda: 1)
    monkeypatch.setattr(cv2.cuda, "createLaplacianFilter", fake_create_laplacian, raising=False)
    monkeypatch.setattr(cv2.cuda, "createSobelFilter", fake_create_sobel, raising=False)

    img = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    expected_lap = cv2.Laplacian(img, cv2.CV_64F).var()
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    expected_ten = np.mean(gx * gx + gy * gy)
    assert metric_value(img, FocusMetric.LAPLACIAN) == pytest.approx(expected_lap)
    assert metric_value(img, FocusMetric.TENENGRAD) == pytest.approx(expected_ten)
