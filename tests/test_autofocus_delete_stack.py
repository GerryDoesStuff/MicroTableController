import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="OpenCV is required to test focus-stack file deletion",
    exc_type=ImportError,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from microstage_app.control.autofocus import AutoFocus
from microstage_app.io.storage import ImageWriter


class DummyStage:
    def __init__(self):
        self._z = 0.0

    def move_relative(self, dz=0.0, feed_mm_per_min=None):
        self._z += dz

    def wait_for_moves(self):
        pass

    def get_position(self):
        return {"x": 0.0, "y": 0.0, "z": self._z}


class DummyCamera:
    def __init__(self):
        self._count = 0

    def snap(self):
        self._count += 1
        return np.full((8, 8, 3), self._count, dtype=np.uint8)

    def name(self):
        return "DummyCam"


def test_focus_stack_deletes_frames(tmp_path, monkeypatch):
    stack_dir = tmp_path / "stack"
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))

    monkeypatch.setattr(
        "microstage_app.control.autofocus.time.sleep", lambda *args, **kwargs: None
    )

    def fake_fuse(images, use_cuda=True):
        assert len(images) == 3
        return images[0]

    monkeypatch.setattr("microstage_app.analysis.edf.fuse_stack", fake_fuse)

    autofocus = AutoFocus(DummyStage(), DummyCamera())
    autofocus.focus_stack(
        range_mm=0.1,
        step_mm=0.1,
        writer=writer,
        directory=str(stack_dir),
        base_name="stack",
        fmt="png",
        fuse_edf=True,
        delete_stack=True,
    )

    fused_path = stack_dir / "stack_edf.png"
    assert fused_path.exists()
    assert sorted(p.name for p in stack_dir.iterdir()) == ["stack_edf.png"]
