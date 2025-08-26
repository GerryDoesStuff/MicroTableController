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
    def save_tile(self, img, r, c):
        self.saved.append((img, r, c))


def test_raster_serpentine(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=2, cols=3, x1_mm=0.0, y1_mm=0.0, x2_mm=2.0, y2_mm=1.0, serpentine=True)
    runner = RasterRunner(stage, cam, writer, cfg)
    runner.run()
    assert writer.saved == [(1,0,0),(2,0,1),(3,0,2),(4,1,0),(5,1,1),(6,1,2)]
    assert stage.moves == [
        (1.0,0.0,0.0),
        (1.0,0.0,0.0),
        (0.0,1.0,0.0),
        (-1.0,0.0,0.0),
        (-1.0,0.0,0.0),
    ]
