import sys
from pathlib import Path
import types

import numpy as np
import cv2
import pytest

# Ensure repository root on import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from microstage_app.ui import main_window


def _run_preview(monkeypatch, frame, gpu):
    mw = main_window.MainWindow.__new__(main_window.MainWindow)
    mw.camera = types.SimpleNamespace(get_latest_frame=lambda: frame)
    captured = {}
    mw.measure_view = types.SimpleNamespace(set_image=lambda img: captured.setdefault("img", img))
    mw.autoexp_chk = types.SimpleNamespace(isChecked=lambda: False)
    mw.exp_spin = None
    mw.gain_spin = None
    monkeypatch.setattr(main_window, "numpy_to_qimage", lambda arr: arr)
    if gpu:
        class FakeGpuMat:
            def __init__(self):
                self.mat = None
            def upload(self, arr):
                self.mat = arr
            def download(self):
                return self.mat
        def fake_cvtColor(gm, code):
            converted = cv2.cvtColor(gm.mat, code)
            out = FakeGpuMat()
            out.mat = converted
            return out
        monkeypatch.setattr(cv2, "cuda_GpuMat", FakeGpuMat)
        monkeypatch.setattr(cv2.cuda, "getCudaEnabledDeviceCount", lambda: 1)
        monkeypatch.setattr(cv2.cuda, "cvtColor", fake_cvtColor, raising=False)
    else:
        monkeypatch.setattr(cv2.cuda, "getCudaEnabledDeviceCount", lambda: 0)
    main_window.MainWindow._on_preview(mw)
    return captured["img"]


def test_preview_cpu(monkeypatch):
    frame = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    res = _run_preview(monkeypatch, frame, gpu=False)
    assert np.array_equal(res, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def test_preview_gpu(monkeypatch):
    frame = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    res = _run_preview(monkeypatch, frame, gpu=True)
    assert np.array_equal(res, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
