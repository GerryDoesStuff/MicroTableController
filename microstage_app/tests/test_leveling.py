import threading

import numpy as np
import pytest

from microstage_app.control.leveling import (
    three_point_level,
    grid_level,
    LevelingMode,
)


class DummyStage:
    def __init__(self, surface):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.surface = surface

    def move_absolute(self, x=None, y=None, z=None, feed_mm_per_min=0.0):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z

    def wait_for_moves(self):
        pass

    def get_position(self):
        self.z = self.surface(self.x, self.y)
        return (self.x, self.y, self.z)


class DummyCamera:
    def snap(self):  # pragma: no cover - not used
        return None


class ManualStage:
    def __init__(self, surface):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.surface = surface
        self.positions = []

    def move_absolute(self, x=None, y=None, z=None, feed_mm_per_min=0.0):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z

    def wait_for_moves(self):
        pass

    def get_position(self):
        pos = (self.x, self.y, self.z)
        self.positions.append(pos)
        return pos


def test_three_point_level_linear(monkeypatch):
    # Disable autofocus
    import microstage_app.control.leveling as leveling
    monkeypatch.setattr(leveling, "AutoFocus", None)

    def plane(x, y):
        return x + 2 * y

    stage = DummyStage(plane)
    cam = DummyCamera()
    pts = [(0, 0), (1, 0), (0, 1)]
    model = three_point_level(stage, cam, pts, LevelingMode.LINEAR)
    assert np.isclose(model.predict(2, 3), plane(2, 3))


def test_three_point_level_autofocus_called(monkeypatch):
    import microstage_app.control.leveling as leveling

    class DummyAF:
        calls = 0

        def __init__(self, stage, camera):
            DummyAF.calls += 1

        def coarse_to_fine(self, metric=None):
            pass

    monkeypatch.setattr(leveling, "AutoFocus", DummyAF)

    def plane(x, y):
        return x + 2 * y

    stage = DummyStage(plane)
    cam = DummyCamera()
    pts = [(0, 0), (1, 0), (0, 1)]
    model = three_point_level(stage, cam, pts, LevelingMode.LINEAR)
    assert DummyAF.calls == len(pts)
    assert np.isclose(model.predict(2, 3), plane(2, 3))


def test_three_point_level_insufficient_points(monkeypatch):
    import microstage_app.control.leveling as leveling
    monkeypatch.setattr(leveling, "AutoFocus", None)

    stage = DummyStage(lambda x, y: 0)
    cam = DummyCamera()

    for mode, required in [
        (LevelingMode.LINEAR, 3),
        (LevelingMode.QUADRATIC, 6),
        (LevelingMode.CUBIC, 10),
    ]:
        pts = [(0, 0)] * (required - 1)
        with pytest.raises(ValueError) as exc:
            three_point_level(stage, cam, pts, mode)
        assert str(required) in str(exc.value)


def test_grid_level_linear_autofocus(monkeypatch):
    import microstage_app.control.leveling as leveling

    class DummyAF:
        calls = 0

        def __init__(self, stage, camera):
            DummyAF.calls += 1

        def coarse_to_fine(self, metric=None):
            pass

    monkeypatch.setattr(leveling, "AutoFocus", DummyAF)

    def plane(x, y):
        return x + 2 * y

    stage = DummyStage(plane)
    cam = DummyCamera()
    rect = (0.0, 0.0, 1.0, 1.0)
    model = grid_level(stage, cam, rect, rows=2, cols=2, mode=LevelingMode.LINEAR)
    assert DummyAF.calls == 4
    assert np.isclose(model.predict(2, 3), plane(2, 3))


def test_grid_level_manual_records_z(monkeypatch):
    import microstage_app.control.leveling as leveling

    class DummyAF:
        calls = 0

        def __init__(self, stage, camera):
            DummyAF.calls += 1

        def coarse_to_fine(self, metric=None):
            pass

    monkeypatch.setattr(leveling, "AutoFocus", DummyAF)

    def plane(x, y):
        return x + 2 * y

    stage = ManualStage(plane)
    cam = DummyCamera()

    def focus_prompt(*args, **kwargs):
        stage.z = plane(stage.x, stage.y)
        return ""

    monkeypatch.setattr("builtins.input", focus_prompt)

    rect = (0.0, 0.0, 1.0, 1.0)
    model = grid_level(
        stage, cam, rect, rows=2, cols=2, mode=LevelingMode.LINEAR, autofocus=False
    )
    assert DummyAF.calls == 0
    expected = [
        (0.0, 0.0, plane(0, 0)),
        (1.0, 0.0, plane(1, 0)),
        (0.0, 1.0, plane(0, 1)),
        (1.0, 1.0, plane(1, 1)),
    ]
    assert np.allclose(stage.positions, expected)
    assert np.isclose(model.predict(2, 3), plane(2, 3))


def test_grid_level_event_cancel(monkeypatch):
    import microstage_app.control.leveling as leveling
    monkeypatch.setattr(leveling, "AutoFocus", None)

    cancel_event = threading.Event()

    def plane(x, y):
        return x + 2 * y

    class CancelStage(ManualStage):
        def __init__(self, surface, event):
            super().__init__(surface)
            self.event = event
            self.move_calls = 0

        def move_absolute(self, x=None, y=None, z=None, feed_mm_per_min=0.0):
            super().move_absolute(x=x, y=y, z=z, feed_mm_per_min=feed_mm_per_min)
            self.move_calls += 1
            if self.move_calls == 2:
                self.event.set()

        def wait_for_moves(self):
            if self.event.is_set():
                raise RuntimeError("cancelled")

        def get_position(self):
            if self.event.is_set():
                raise AssertionError("get_position called after cancel")
            return super().get_position()

    stage = CancelStage(plane, cancel_event)
    cam = DummyCamera()
    rect = (0.0, 0.0, 1.0, 1.0)

    with pytest.raises(RuntimeError):
        grid_level(stage, cam, rect, rows=2, cols=2, mode=LevelingMode.LINEAR)

    cancel_event.clear()
    assert not cancel_event.is_set()
    assert stage.move_calls == 2
    assert len(stage.positions) == 1
