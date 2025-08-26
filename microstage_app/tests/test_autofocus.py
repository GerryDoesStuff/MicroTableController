import microstage_app.control.autofocus as af
from microstage_app.control.autofocus import AutoFocus, FocusMetric
import pytest

class StageMock:
    def __init__(self):
        self.z = 0.0
        self.moves = []
    def move_relative(self, dz=0.0, feed_mm_per_min=0.0):
        self.z += dz
        self.moves.append((dz, feed_mm_per_min))
    def wait_for_moves(self):
        pass

class CameraMock:
    def snap(self):
        return object()


def test_autofocus_converges(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    monkeypatch.setattr(af, 'metric_value', lambda img, metric: -abs(stage.z))
    autofocus = AutoFocus(stage, cam)
    best = autofocus.coarse_to_fine(
        FocusMetric.LAPLACIAN, z_range_mm=0.2, coarse_step_mm=0.1, fine_step_mm=0.05
    )
    assert abs(best) < 1e-6


def test_autofocus_zero_step_raises():
    stage = StageMock()
    cam = CameraMock()
    autofocus = AutoFocus(stage, cam)
    with pytest.raises(ValueError):
        autofocus.coarse_to_fine(FocusMetric.LAPLACIAN, coarse_step_mm=0.0)
    with pytest.raises(ValueError):
        autofocus.coarse_to_fine(FocusMetric.LAPLACIAN, fine_step_mm=0.0)


def test_fine_pass_window_and_step(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    positions = []

    def fake_metric(img, metric):
        positions.append(stage.z)
        return -abs(stage.z)

    monkeypatch.setattr(af, 'metric_value', fake_metric)
    autofocus = AutoFocus(stage, cam)

    z_range = 0.4
    coarse_step = 0.1
    fine_step = 0.02
    result = autofocus.coarse_to_fine(
        FocusMetric.LAPLACIAN,
        z_range_mm=z_range,
        coarse_step_mm=coarse_step,
        fine_step_mm=fine_step,
    )

    steps = int(max(1, round(z_range / coarse_step)))
    coarse_samples = 2 * steps + 1
    coarse_positions = positions[:coarse_samples]
    fine_positions = positions[coarse_samples:]

    fine_range = 0.1 * z_range
    coarse_best = max(coarse_positions, key=lambda z: -abs(z))

    assert fine_positions[0] == pytest.approx(coarse_best - fine_range)
    assert fine_positions[-1] == pytest.approx(coarse_best + fine_range)
    for a, b in zip(fine_positions, fine_positions[1:]):
        assert (b - a) == pytest.approx(fine_step)

    assert result == pytest.approx(coarse_best)
    assert stage.z == pytest.approx(coarse_best)


def test_feed_rate_passed_to_stage(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    def fake_metric(img, metric):
        return -abs(stage.z)

    monkeypatch.setattr(af, 'metric_value', fake_metric)
    af_inst = AutoFocus(stage, cam)
    af_inst.coarse_to_fine(
        FocusMetric.LAPLACIAN,
        z_range_mm=0.1,
        coarse_step_mm=0.05,
        fine_step_mm=0.02,
        feed_mm_per_min=55,
    )
    assert all(feed == 55 for _, feed in stage.moves)
