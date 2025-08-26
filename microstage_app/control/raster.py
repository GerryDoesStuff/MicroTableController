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
        for r in range(self.cfg.rows):
            forward = (r % 2 == 0) or (not self.cfg.serpentine)
            cols = range(self.cfg.cols) if forward else range(self.cfg.cols-1, -1, -1)
            last_c = self.cfg.cols - 1 if forward else 0
            for c in cols:
                self.stage.wait_for_moves()
                time.sleep(0.03)
                img = self.camera.snap()
                if img is not None:
                    self.writer.save_tile(img, r, c if forward else (self.cfg.cols-1 - c))
                if c != last_c:
                    dx = self.pitch_x_mm if forward else -self.pitch_x_mm
                    self.stage.move_relative(dx=dx)
            if r < self.cfg.rows - 1:
                self.stage.move_relative(dy=self.pitch_y_mm, feed_mm_per_min=self.cfg.feed_y_mm_min)
            if self.cfg.cols > 1 and not self.cfg.serpentine:
                dx = self.pitch_x_mm * (self.cfg.cols - 1)
                # Return to start of next row if needed
                self.stage.move_relative(dx=-dx, feed_mm_per_min=self.cfg.feed_x_mm_min)
