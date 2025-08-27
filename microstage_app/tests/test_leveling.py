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
    monkeypatch.setattr(leveling, "AutoFocus", None)

    def plane(x, y):
        return x + 2 * y

    stage = DummyStage(plane)
    cam = DummyCamera()
    rect = (0.0, 0.0, 1.0, 1.0)
    model = grid_level(stage, cam, rect, rows=2, cols=2, mode=LevelingMode.LINEAR)
    assert np.isclose(model.predict(2, 3), plane(2, 3))


def test_grid_level_manual(monkeypatch):
    import microstage_app.control.leveling as leveling
    monkeypatch.setattr(leveling, "AutoFocus", None)
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    def plane(x, y):
        return x + 2 * y

    stage = DummyStage(plane)
    cam = DummyCamera()
    rect = (0.0, 0.0, 1.0, 1.0)
    model = grid_level(
        stage, cam, rect, rows=2, cols=2, mode=LevelingMode.LINEAR, autofocus=False
    )
    assert np.isclose(model.predict(2, 3), plane(2, 3))
