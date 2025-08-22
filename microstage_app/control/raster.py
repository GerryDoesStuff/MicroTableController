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
            step_x = self.cfg.pitch_x_mm if forward else -self.cfg.pitch_x_mm
            for i in range(self.cfg.cols):
                col_index = i if forward else (self.cfg.cols - 1 - i)
                self.stage.wait_for_moves()
                time.sleep(0.03)
                img = self.camera.snap()
                if img is not None:
                    self.writer.save_tile(img, r, col_index)
                # move to next column except after last tile in the row
                if i < self.cfg.cols - 1:
                    self.stage.move_relative(dx=step_x)
            # move to next row
            if r < self.cfg.rows - 1:
                # if not serpentine, return X to starting edge before moving to next row
                if not self.cfg.serpentine and self.cfg.cols > 1:
                    self.stage.move_relative(dx=-step_x * (self.cfg.cols - 1))
                self.stage.move_relative(dy=self.cfg.pitch_y_mm)
