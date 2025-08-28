from dataclasses import dataclass
import time
from math import isclose
from threading import Event
from typing import Optional

from ..utils.img import draw_scale_bar

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
    y2_mm: float = 0.0
    x3_mm: float = 0.0
    y3_mm: float = 1.0
    x4_mm: float = 1.0
    y4_mm: float = 1.0
    mode: str = "rectangle"  # rectangle, parallelogram, trapezoid
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
        lens_name=None,
        scale_bar_um_per_px: Optional[float] = None,
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
        self.lens_name = lens_name
        self.scale_bar_um_per_px = scale_bar_um_per_px

        self.coord_matrix = None
        self._stop = False

    def _build_coord_matrix(self):
        """Generate the coordinate matrix for the configured raster mode."""
        if self.coord_matrix is not None:
            return self.coord_matrix

        cfg = self.cfg
        matrix = []
        if cfg.mode == "rectangle":
            col_dx = (cfg.x2_mm - cfg.x1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
            col_dy = (cfg.y2_mm - cfg.y1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
            row_dx = (cfg.x3_mm - cfg.x1_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
            row_dy = (cfg.y3_mm - cfg.y1_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
            for r in range(cfg.rows):
                base_x = cfg.x1_mm + row_dx * r
                base_y = cfg.y1_mm + row_dy * r
                row = []
                for c in range(cfg.cols):
                    x = base_x + col_dx * c
                    y = base_y + col_dy * c
                    row.append((x, y))
                matrix.append(row)
        elif cfg.mode == "parallelogram":
            col_vec_x = (cfg.x2_mm - cfg.x1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
            col_vec_y = (cfg.y2_mm - cfg.y1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
            row_vec_x = (cfg.x3_mm - cfg.x1_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
            row_vec_y = (cfg.y3_mm - cfg.y1_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
            for r in range(cfg.rows):
                row = []
                for c in range(cfg.cols):
                    x = cfg.x1_mm + c * col_vec_x + r * row_vec_x
                    y = cfg.y1_mm + c * col_vec_y + r * row_vec_y
                    row.append((x, y))
                matrix.append(row)
        elif cfg.mode == "trapezoid":
            for r in range(cfg.rows):
                t_r = r / (cfg.rows - 1) if cfg.rows > 1 else 0.0
                start_x = cfg.x1_mm + (cfg.x3_mm - cfg.x1_mm) * t_r
                start_y = cfg.y1_mm + (cfg.y3_mm - cfg.y1_mm) * t_r
                end_x = cfg.x2_mm + (cfg.x4_mm - cfg.x2_mm) * t_r
                end_y = cfg.y2_mm + (cfg.y4_mm - cfg.y2_mm) * t_r
                row = []
                for c in range(cfg.cols):
                    t_c = c / (cfg.cols - 1) if cfg.cols > 1 else 0.0
                    x = start_x + (end_x - start_x) * t_c
                    y = start_y + (end_y - start_y) * t_c
                    row.append((x, y))
                matrix.append(row)
        else:  # pragma: no cover - validation
            raise ValueError(f"Unknown raster mode: {cfg.mode}")

        self.coord_matrix = matrix
        return matrix

    def stop(self):
        """Request that the raster scan stop after the current move."""
        self._stop = True

    def run(self, stop_event: Optional[Event] = None):
        """Execute raster scan and capture images for each tile.

        The coordinate matrix is generated based on :class:`RasterConfig.mode`
        and then traversed in either serpentine or raster order.
        """

        coord_matrix = self._build_coord_matrix()

        if stop_event and stop_event.is_set():
            return

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
            if stop_event and stop_event.is_set():
                return
            self.stage.move_absolute(x=start_x, y=start_y)
            self.stage.wait_for_moves()
            if self._stop:
                return
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
                if self._stop:
                    return
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
                        if self.scale_bar_um_per_px is not None:
                            img = draw_scale_bar(img, self.scale_bar_um_per_px)
                        save_c = c
                        fname = f"{self.base_name}_r{r:04d}_c{save_c:04d}"
                        pos = self.stage.get_position()
                        metadata = {
                            "Camera": self.camera.name(),
                            "Position": pos,
                            "Lens": self.lens_name,
                        }
                        self.writer.save_single(
                            img,
                            directory=self.directory,
                            filename=fname,
                            auto_number=self.auto_number,
                            fmt=self.fmt,
                            metadata=metadata,
                        )
                    time.sleep(1)

