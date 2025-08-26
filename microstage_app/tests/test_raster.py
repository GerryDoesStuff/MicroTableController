import microstage_app.control.raster as raster
from microstage_app.control.raster import RasterRunner, RasterConfig

class StageMock:
    def __init__(self):
        self.moves = []
    def move_relative(self, dx=0.0, dy=0.0, dz=0.0, feed_mm_per_min=0.0):
        self.moves.append((dx, dy, dz))
    def wait_for_moves(self):
        pass

class CameraMock:
    def __init__(self):
        self.count = 0
    def snap(self):
        self.count += 1
        return self.count

class WriterMock:
    def __init__(self):
        self.saved = []
    def save_single(self, img, directory=None, filename="capture", auto_number=False, fmt="bmp"):
        self.saved.append((img, directory, filename, auto_number, fmt))


def test_raster_serpentine(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=2, cols=3, x1_mm=0.0, y1_mm=0.0, x2_mm=2.0, y2_mm=1.0, serpentine=True)
    runner = RasterRunner(stage, cam, writer, cfg, directory="out", base_name="foo", fmt="bmp")
    runner.run()
    assert writer.saved == [
        (1, "out", "foo_r0000_c0000", False, "bmp"),
        (2, "out", "foo_r0000_c0001", False, "bmp"),
        (3, "out", "foo_r0000_c0002", False, "bmp"),
        (4, "out", "foo_r0001_c0000", False, "bmp"),
        (5, "out", "foo_r0001_c0001", False, "bmp"),
        (6, "out", "foo_r0001_c0002", False, "bmp"),
    ]
    assert stage.moves == [
        (1.0,0.0,0.0),
        (1.0,0.0,0.0),
        (0.0,1.0,0.0),
        (-1.0,0.0,0.0),
        (-1.0,0.0,0.0),
    ]


def test_raster_no_serpentine(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=2, cols=3, x1_mm=0.0, y1_mm=0.0, x2_mm=2.0, y2_mm=1.0, serpentine=False)
    runner = RasterRunner(stage, cam, writer, cfg, directory="out", base_name="foo", fmt="bmp")
    runner.run()
    assert writer.saved == [
        (1, "out", "foo_r0000_c0000", False, "bmp"),
        (2, "out", "foo_r0000_c0001", False, "bmp"),
        (3, "out", "foo_r0000_c0002", False, "bmp"),
        (4, "out", "foo_r0001_c0000", False, "bmp"),
        (5, "out", "foo_r0001_c0001", False, "bmp"),
        (6, "out", "foo_r0001_c0002", False, "bmp"),
    ]
    assert stage.moves == [
        (1.0,0.0,0.0),
        (1.0,0.0,0.0),
        (-2.0,1.0,0.0),
        (1.0,0.0,0.0),
        (1.0,0.0,0.0),
    ]


def test_raster_capture_disabled():
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=1, cols=2, capture=False)
    runner = RasterRunner(stage, cam, writer, cfg)
    runner.run()
    assert writer.saved == []


def test_raster_autofocus(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=2, cols=2, autofocus=True, capture=False)
    called = []

    class DummyAF:
        def __init__(self, stage, camera):
            pass
        def coarse_to_fine(self, metric=None, **kwargs):
            called.append(metric)
            return 0.0

    monkeypatch.setattr(raster, "AutoFocus", DummyAF)
    runner = RasterRunner(stage, cam, writer, cfg)
    runner.run()
    assert len(called) == cfg.rows * cfg.cols
