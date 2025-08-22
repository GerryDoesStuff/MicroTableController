import microstage_app.control.autofocus as af
from microstage_app.control.autofocus import AutoFocus, FocusMetric

class StageMock:
    def __init__(self):
        self.z = 0.0
        self.moves = []
    def move_relative(self, dz=0.0, feed_mm_per_min=0.0):
        self.z += dz
        self.moves.append(dz)
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
    best = autofocus.coarse_to_fine(FocusMetric.LAPLACIAN, z_range_mm=0.2, coarse_step_mm=0.1, fine_step_mm=0.05)
    assert abs(best) < 1e-6
