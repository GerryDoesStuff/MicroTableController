from dataclasses import dataclass
import time

@dataclass
class RasterConfig:
    rows: int = 5
    cols: int = 5
    pitch_x_mm: float = 1.0
    pitch_y_mm: float = 1.0
    serpentine: bool = True

class RasterRunner:
    def __init__(self, stage, camera, writer, cfg: RasterConfig):
        self.stage = stage; self.camera = camera; self.writer = writer; self.cfg = cfg

    def run(self):
        for r in range(self.cfg.rows):
            forward = (r % 2 == 0) or (not self.cfg.serpentine)
            cols = range(self.cfg.cols) if forward else range(self.cfg.cols-1, -1, -1)
            for c in cols:
                self.stage.wait_for_moves()
                time.sleep(0.03)
                img = self.camera.snap()
                if img is not None:
                    self.writer.save_tile(img, r, c if forward else (self.cfg.cols-1 - c))
            if r < self.cfg.rows - 1:
                self.stage.move_relative(dy=self.cfg.pitch_y_mm)
            if self.cfg.cols > 1:
                dx = self.cfg.pitch_x_mm * (self.cfg.cols - 1)
                # Return to start of next row if needed
                self.stage.move_relative(dx=-dx)
