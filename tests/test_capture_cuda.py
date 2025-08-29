import sys
from pathlib import Path
import types

import numpy as np
import cv2
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from microstage_app.ui import main_window


def _run_capture(monkeypatch, frame, gpu):
    mw = main_window.MainWindow.__new__(main_window.MainWindow)
    mw.stage = types.SimpleNamespace(wait_for_moves=lambda: None, get_position=lambda: {})
    saved = {}
    def fake_save(img, **kwargs):
        saved['img'] = img
    mw.image_writer = types.SimpleNamespace(save_single=fake_save)
    mw.chk_scale_bar = types.SimpleNamespace(isChecked=lambda: True)
    mw.current_lens = types.SimpleNamespace(um_per_px=1.0, name='lens')
    mw.capture_dir = '/tmp'
    mw.capture_name = 'test'
    mw.auto_number = False
    mw.capture_format = 'png'

    if gpu:
        class FakeGpuMat:
            def __init__(self, mat=None):
                self.mat = mat
            def upload(self, arr):
                self.mat = arr.copy()
            def download(self):
                return self.mat
            def size(self):
                return (self.mat.shape[1], self.mat.shape[0])
            def rowRange(self, y1, y2):
                return FakeGpuMat(self.mat[y1:y2])
            def colRange(self, x1, x2):
                return FakeGpuMat(self.mat[:, x1:x2])
            def setTo(self, color):
                self.mat[...] = color
        def fake_cvtColor(gm, code):
            out = FakeGpuMat()
            out.mat = cv2.cvtColor(gm.mat, code)
            return out
        monkeypatch.setattr(cv2, 'cuda_GpuMat', FakeGpuMat)
        monkeypatch.setattr(cv2.cuda, 'getCudaEnabledDeviceCount', lambda: 1)
        monkeypatch.setattr(cv2.cuda, 'cvtColor', fake_cvtColor, raising=False)
    else:
        monkeypatch.setattr(cv2.cuda, 'getCudaEnabledDeviceCount', lambda: 0)

    def fake_snap(use_cuda=False):
        assert use_cuda == gpu
        if use_cuda:
            gm = cv2.cuda_GpuMat()
            gm.upload(frame)
            gm = cv2.cuda.cvtColor(gm, cv2.COLOR_BGR2RGB)
            return gm.download()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mw.camera = types.SimpleNamespace(name=lambda: 'cam', snap=fake_snap,
                                      get_exposure_ms=lambda: None, get_gain=lambda: None)

    class DummySignal:
        def connect(self, cb):
            cb(True, None)
    class DummyWorker:
        finished = DummySignal()
    def fake_run_async(func):
        func()
        return None, DummyWorker()
    monkeypatch.setattr(main_window, 'run_async', fake_run_async)

    mw._capture()
    return saved['img']


def test_capture_cpu(monkeypatch):
    frame = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    out = _run_capture(monkeypatch, frame, gpu=False)
    assert np.array_equal(out, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def test_capture_gpu(monkeypatch):
    frame = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    out = _run_capture(monkeypatch, frame, gpu=True)
    assert np.array_equal(out, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
