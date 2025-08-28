import gc
import os
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QThread

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub out the image utilities to avoid heavy Qt GUI imports during tests
import types

img_stub = types.ModuleType("microstage_app.utils.img")
img_stub.draw_scale_bar = lambda img, um_per_px: img
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
