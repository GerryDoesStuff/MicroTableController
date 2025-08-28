import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ImageFont

import microstage_app.control.raster as raster
from microstage_app.control.raster import RasterRunner, RasterConfig

class StageMock:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.moves = []
        self.moves_abs = []
        self.pos = [x, y, z]
    def get_position(self):
        return self.pos[0], self.pos[1]
    def move_absolute(self, x=None, y=None, z=None, feed_mm_per_min=0.0):
        if x is not None:
            self.pos[0] = x
        if y is not None:
            self.pos[1] = y
        if z is not None:
            self.pos[2] = z
        self.moves_abs.append((self.pos[0], self.pos[1], self.pos[2]))
    def move_relative(self, dx=0.0, dy=0.0, dz=0.0, feed_mm_per_min=0.0):
        self.moves.append((dx, dy, dz))
        self.pos[0] += dx
        self.pos[1] += dy
        self.pos[2] += dz
    def wait_for_moves(self):
        pass

class CameraMock:
    def __init__(self):
        self.count = 0
    def snap(self):
        self.count += 1
        return self.count
    def name(self):
        return "CameraMock"

class WriterMock:
    def __init__(self):
        self.saved = []
    def save_single(
        self,
        img,
        directory=None,
        filename="capture",
        auto_number=False,
        fmt="bmp",
        metadata=None,
    ):
        self.saved.append((img, directory, filename, auto_number, fmt, metadata))


@pytest.mark.parametrize("mode", ["rectangle", "parallelogram", "trapezoid"])
@pytest.mark.parametrize("serpentine", [True, False])
def test_raster_traversal_modes(mode, serpentine):
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg_kwargs = dict(
        rows=2,
        cols=3,
        x1_mm=0.0,
        y1_mm=0.0,
        serpentine=serpentine,
        mode=mode,
    )
    if mode == "rectangle":
        cfg_kwargs.update(x2_mm=2.0, y2_mm=0.0, x3_mm=0.0, y3_mm=1.0, x4_mm=2.0, y4_mm=1.0)
    elif mode == "parallelogram":
        cfg_kwargs.update(x2_mm=2.0, y2_mm=0.0, x3_mm=2.5, y3_mm=1.0)
    elif mode == "trapezoid":
        cfg_kwargs.update(x2_mm=4.0, y2_mm=0.0, x3_mm=1.0, y3_mm=2.0, x4_mm=3.0, y4_mm=2.0)

    cfg = RasterConfig(**cfg_kwargs)
    runner = RasterRunner(stage, cam, writer, cfg, directory="out", base_name="foo", fmt="bmp")

    coord_matrix = runner._build_coord_matrix()
    runner.run()

    # expected moves and filenames
    expected_moves = []
    expected_files = []
    current_x, current_y = coord_matrix[0][0]
    for r in range(cfg.rows):
        forward = (r % 2 == 0) or (not serpentine)
        cols = range(cfg.cols) if forward else range(cfg.cols - 1, -1, -1)
        for c in cols:
            target_x, target_y = coord_matrix[r][c]
            dx = target_x - current_x
            dy = target_y - current_y
            if dx or dy:
                expected_moves.append((dx, dy, 0.0))
            current_x, current_y = target_x, target_y
            expected_files.append(f"foo_r{r:04d}_c{c:04d}")

    assert stage.moves == expected_moves
    assert [f[2] for f in writer.saved] == expected_files


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


def test_raster_initial_move():
    stage = StageMock(x=1.0, y=1.0)
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=1, cols=1, x1_mm=0.0, y1_mm=0.0, capture=False)
    runner = RasterRunner(stage, cam, writer, cfg)
    runner.run()
    assert stage.moves_abs == [(0.0, 0.0, 0.0)]


def test_raster_initial_move_cancelled():
    stage = StageMock(x=1.0, y=1.0)
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=1, cols=1, x1_mm=0.0, y1_mm=0.0, capture=False)
    runner = RasterRunner(stage, cam, writer, cfg)

    stop_event = threading.Event()
    stop_event.set()

    runner.run(stop_event=stop_event)

    assert stage.moves_abs == []
    assert stage.moves == []
    assert writer.saved == []


@pytest.mark.parametrize(
    "autofocus,capture,expected",
    [
        (True, True, ["autofocus", ("sleep", 1), "snap", "save", ("sleep", 1)]),
        (True, False, ["autofocus", ("sleep", 1)]),
        (False, True, ["snap", "save", ("sleep", 1)]),
    ],
)
def test_raster_operation_order(monkeypatch, autofocus, capture, expected):
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=1, cols=1, autofocus=autofocus, capture=capture)
    events = []

    class DummyAF:
        def __init__(self, stage, camera):
            pass

        def coarse_to_fine(self, metric=None, **kwargs):
            events.append("autofocus")

    monkeypatch.setattr(raster, "AutoFocus", DummyAF)

    def fake_snap():
        events.append("snap")
        return object()

    monkeypatch.setattr(cam, "snap", fake_snap)

    def fake_save(*args, **kwargs):
        events.append("save")

    monkeypatch.setattr(writer, "save_single", fake_save)

    def fake_sleep(delay):
        events.append(("sleep", delay))

    monkeypatch.setattr(raster.time, "sleep", fake_sleep)

    runner = RasterRunner(stage, cam, writer, cfg)
    runner.run()

    if events and events[0] == ("sleep", 0.03):
        events = events[1:]

    assert events == expected


def test_raster_cancels_with_event(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=2, cols=2)
    runner = RasterRunner(stage, cam, writer, cfg)

    cancel_event = threading.Event()

    def move_relative(dx=0.0, dy=0.0, dz=0.0, feed_mm_per_min=0.0):
        stage.moves.append((dx, dy, dz))
        stage.pos[0] += dx
        stage.pos[1] += dy
        stage.pos[2] += dz
        cancel_event.set()

    stage.move_relative = move_relative

    def wait_for_moves():
        if cancel_event.is_set():
            runner.stop()

    stage.wait_for_moves = wait_for_moves

    monkeypatch.setattr(raster.time, "sleep", lambda s: None)

    runner.run()

    cancel_event.clear()
    assert not cancel_event.is_set()
    assert len(stage.moves) == 1
    assert len(writer.saved) == 1


def test_raster_parallelogram_matrix():
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(
        rows=2,
        cols=3,
        x1_mm=0.0,
        y1_mm=0.0,
        x2_mm=2.0,
        y2_mm=0.0,
        x3_mm=2.5,
        y3_mm=1.0,
        mode="parallelogram",
        capture=False,
    )
    runner = RasterRunner(stage, cam, writer, cfg)
    assert runner._build_coord_matrix() == [
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        [(0.5, 1.0), (1.5, 1.0), (2.5, 1.0)],
    ]


def test_raster_trapezoid_matrix():
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(
        rows=3,
        cols=3,
        x1_mm=0.0,
        y1_mm=0.0,
        x2_mm=4.0,
        y2_mm=0.0,
        x3_mm=1.0,
        y3_mm=3.0,
        x4_mm=3.0,
        y4_mm=3.0,
        mode="trapezoid",
        capture=False,
    )
    runner = RasterRunner(stage, cam, writer, cfg)
    assert runner._build_coord_matrix() == [
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)],
        [(0.5, 1.5), (2.0, 1.5), (3.5, 1.5)],
        [(1.0, 3.0), (2.0, 3.0), (3.0, 3.0)],
    ]


def test_raster_scale_bar(monkeypatch):
    stage = StageMock()
    cam = CameraMock()
    writer = WriterMock()
    cfg = RasterConfig(rows=1, cols=1)
    called = []

    def fake_draw(img, um_per_px):
        called.append(um_per_px)
        return img

    monkeypatch.setattr(raster, "draw_scale_bar", fake_draw)

    runner = RasterRunner(stage, cam, writer, cfg, scale_bar_um_per_px=1.23)
    runner.run()

    assert called == [1.23]


def test_raster_capture_contains_scale_bar(monkeypatch):
    stage = StageMock()
    cam = SimpleNamespace(
        snap=lambda: np.zeros((100, 200, 3), dtype=np.uint8),
        name=lambda: "CameraMock",
    )
    saved = {}
    writer = SimpleNamespace(
        save_single=lambda img, **kw: saved.setdefault("img", img)
    )
    cfg = RasterConfig(rows=1, cols=1)

    monkeypatch.setattr(raster.time, "sleep", lambda s: None)

    orig_truetype = ImageFont.truetype

    def fake_truetype(font, size=10, *args, **kwargs):
        if isinstance(font, (str, bytes)):
            raise OSError("missing font")
        return orig_truetype(font, size, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", fake_truetype)

    runner = RasterRunner(stage, cam, writer, cfg, scale_bar_um_per_px=1.0)
    runner.run()

    out = saved["img"]
    bar_row = out[80]
    bar_pixels = np.where(np.all(bar_row == 255, axis=1))[0]
    assert bar_pixels[0] == 160
    assert bar_pixels[-1] - bar_pixels[0] == 20
    assert np.all(bar_row[:160] == 0)
    assert np.all(bar_row[181:] == 0)
