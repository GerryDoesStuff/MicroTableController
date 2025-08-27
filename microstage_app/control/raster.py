from dataclasses import dataclass
import time
from math import isclose

try:
    from .autofocus import AutoFocus, FocusMetric
except Exception:  # pragma: no cover - autofocus deps may be missing
    AutoFocus = None
    class FocusMetric:
        LAPLACIAN = None

@dataclass
class RasterConfig:
    rows: int = 5
    cols: int = 5
    x1_mm: float = 0.0
    y1_mm: float = 0.0
    x2_mm: float = 1.0
    y2_mm: float = 1.0
    serpentine: bool = True
    feed_x_mm_min: float = 50.0
    feed_y_mm_min: float = 50.0
    autofocus: bool = False
    capture: bool = True

class RasterRunner:
    def __init__(
        self,
        stage,
        camera,
        writer,
        cfg: RasterConfig,
        directory=None,
        base_name="tile",
        auto_number=False,
        fmt="tif",
        position_cb=None,
    ):
        self.stage = stage
        self.camera = camera
        self.writer = writer
        self.cfg = cfg
        self.directory = directory
        self.base_name = base_name
        self.auto_number = auto_number
        self.fmt = fmt
        self.position_cb = position_cb
        self.pitch_x_mm = (
            (cfg.x2_mm - cfg.x1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
        )
        self.pitch_y_mm = (
            (cfg.y2_mm - cfg.y1_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
        )

    def run(self):
        """Execute raster scan and capture images for each tile.

        A matrix of target coordinates is generated from the diagonal
        points and the requested row/column counts.  The stage is then
        stepped through these coordinates in a serpentine pattern (if
        enabled) ensuring that every tile is visited exactly once.
        """

        # Build matrix of absolute target coordinates
        xs = [self.cfg.x1_mm + self.pitch_x_mm * c for c in range(self.cfg.cols)]
        ys = [self.cfg.y1_mm + self.pitch_y_mm * r for r in range(self.cfg.rows)]
        coord_matrix = [[(x, y) for x in xs] for y in ys]

        start_x, start_y = coord_matrix[0][0]
        try:
            pos = self.stage.get_position()
        except Exception:
            pos = None
        if (
            pos is None
            or not (
                isclose(pos[0], start_x, abs_tol=1e-6)
                and isclose(pos[1], start_y, abs_tol=1e-6)
            )
        ):
            self.stage.move_absolute(x=start_x, y=start_y)
            self.stage.wait_for_moves()
        current_x, current_y = start_x, start_y

        for r in range(self.cfg.rows):
            forward = (r % 2 == 0) or (not self.cfg.serpentine)
            cols = range(self.cfg.cols) if forward else range(self.cfg.cols - 1, -1, -1)
            for c in cols:
                target_x, target_y = coord_matrix[r][c]
                dx = target_x - current_x
                dy = target_y - current_y
                if dx or dy:
                    self.stage.move_relative(dx=dx, dy=dy)
                    current_x, current_y = target_x, target_y

                self.stage.wait_for_moves()
                if self.position_cb:
                    try:
                        pos = self.stage.get_position()
                    except Exception:
                        pos = None
                    self.position_cb(pos)
                time.sleep(0.03)

                do_af = bool(self.cfg.autofocus and AutoFocus)
                do_capture = bool(self.cfg.capture)

                if do_af:
                    af = AutoFocus(self.stage, self.camera)
                    af.coarse_to_fine(metric=FocusMetric.LAPLACIAN)
                    time.sleep(1)

                if do_capture:
                    img = self.camera.snap()
                    if img is not None:
                        save_c = c
                        fname = f"{self.base_name}_r{r:04d}_c{save_c:04d}"
                        self.writer.save_single(
                            img,
                            directory=self.directory,
                            filename=fname,
                            auto_number=self.auto_number,
                            fmt=self.fmt,
                        )
                    time.sleep(1)

