import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QThread

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub out the image utilities to avoid heavy Qt GUI imports during tests
import types


def _stub_draw_scale_bar(img, um_per_px):
    if img is None:
        return None

    arr = np.array(img, copy=True)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)

    h, w = arr.shape[:2]
    if um_per_px <= 0 or h == 0 or w == 0:
        return arr

    max_um = 0.2 * w * um_per_px
    exp = math.floor(math.log10(max_um)) if max_um > 0 else 0
    nice_um = 10 ** exp
    for m in (5, 2, 1):
        candidate = m * (10 ** exp)
        if candidate <= max_um:
            nice_um = candidate
            break

    length_px = int(round(nice_um / um_per_px)) if um_per_px > 0 else 0
    max_length = max(1, w - 40)
    length_px = max(1, min(length_px, max_length))

    margin = min(20, w - 1)
    x0 = max(0, int(round(w - margin - length_px)))
    y0 = max(0, min(h - 1, int(round(h - margin))))
    x1 = min(w - 1, x0 + length_px)
    arr[y0, x0 : x1 + 1] = 255
    return arr


img_stub = types.ModuleType("microstage_app.utils.img")
img_stub.draw_scale_bar = _stub_draw_scale_bar
img_stub.VERT_SCALE = 2
img_stub.TEXT_SCALE = 4
sys.modules["microstage_app.utils.img"] = img_stub

from microstage_app.control.raster import RasterRunner, RasterConfig

# Use offscreen platform to avoid GUI requirements
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class StageStub:
    def __init__(self):
        self.pos = [0.0, 0.0]

    def get_position(self):
        return tuple(self.pos)

    def move_absolute(self, x=None, y=None, **kwargs):
        if x is not None:
            self.pos[0] = x
        if y is not None:
            self.pos[1] = y

    def move_relative(self, dx=0.0, dy=0.0, **kwargs):
        self.pos[0] += dx
        self.pos[1] += dy

    def wait_for_moves(self):
        pass


class CameraStub:
    def snap(self):
        return None

    def name(self):
        return "CameraStub"


class WriterStub:
    def save_single(self, *args, **kwargs):
        pass


def test_raster_config_stack_defaults():
    cfg = RasterConfig()
    assert cfg.stack is False
    assert cfg.stack_range_mm == 0.5
    assert cfg.stack_step_mm == 0.01
    assert cfg.af_range_mm == 0.5
    assert cfg.af_coarse_step_mm == 0.01
    assert cfg.af_fine_step_mm == 0.002


def test_raster_thread_stop(capsys):
    stage = StageStub()
    cam = CameraStub()
    writer = WriterStub()
    cfg = RasterConfig(rows=3, cols=3, capture=False)
    runner = RasterRunner(stage, cam, writer, cfg)

    class RunnerThread(QThread):
        def run(self):
            runner.run()

    thread = RunnerThread()
    thread.start()
    time.sleep(0.05)
    runner.stop()
    assert thread.wait(1000)

    # Ensure thread is cleaned up and no QThread warnings are emitted
    del thread
    gc.collect()
    captured = capsys.readouterr()
    assert "QThread" not in captured.err
