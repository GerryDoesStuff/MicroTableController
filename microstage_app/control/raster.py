from dataclasses import dataclass
import time

@dataclass
class RasterConfig:
    rows: int = 5
    cols: int = 5
    x1_mm: float = 0.0
    y1_mm: float = 0.0
    x2_mm: float = 1.0
    y2_mm: float = 1.0
    serpentine: bool = True
    feed_x_mm_min: float = 20.0
    feed_y_mm_min: float = 20.0

class RasterRunner:
    def __init__(self, stage, camera, writer, cfg: RasterConfig):
        self.stage = stage
        self.camera = camera
        self.writer = writer
        self.cfg = cfg
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

        current_x, current_y = coord_matrix[0][0]
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
                time.sleep(0.03)
                img = self.camera.snap()
                if img is not None:
                    save_c = c if forward else (self.cfg.cols - 1 - c)
                    self.writer.save_tile(img, r, save_c)

