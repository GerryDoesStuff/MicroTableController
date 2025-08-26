from enum import Enum
import math
import numpy as np
import cv2
import time

class FocusMetric(str, Enum):
    LAPLACIAN = "LaplacianVar"
    TENENGRAD = "Tenengrad"

def metric_value(img_rgb, metric: FocusMetric):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if metric == FocusMetric.LAPLACIAN:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    elif metric == FocusMetric.TENENGRAD:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx*gx + gy*gy))
    else:
        raise ValueError(metric)

class AutoFocus:
    def __init__(self, stage, camera):
        self.stage = stage
        self.camera = camera

    def coarse_to_fine(self, metric: FocusMetric, z_range_mm=0.5, coarse_step_mm=0.01, fine_step_mm=0.002):
        if coarse_step_mm <= 0 or fine_step_mm <= 0:
            raise ValueError("coarse_step_mm and fine_step_mm must be > 0")
        samples = []
        steps = int(max(1, round(z_range_mm / coarse_step_mm)))
        zs = [(-steps + i) * coarse_step_mm for i in range(2*steps + 1)]
        cumulative = 0.0
        for dz in zs:
            move = dz - cumulative
            self.stage.move_relative(dz=move, feed_mm_per_min=240)
            cumulative = dz
            self.stage.wait_for_moves()
            time.sleep(0.03)
            img = self.camera.snap()
            if img is None: continue
            samples.append((dz, metric_value(img, metric)))
        if not samples: return 0.0
        best_dz, _ = max(samples, key=lambda t: t[1])
        # Go to coarse best position
        self.stage.move_relative(dz=(best_dz - cumulative), feed_mm_per_min=240)
        self.stage.wait_for_moves()

        # Fine sweep around coarse best
        fine_range = 0.1 * z_range_mm
        fine_steps = int(max(1, math.floor(fine_range / fine_step_mm)))
        offsets = [(-fine_steps + i) * fine_step_mm for i in range(2 * fine_steps + 1)]
        fine_samples = []
        cumulative = 0.0
        for offset in offsets:
            move = offset - cumulative
            self.stage.move_relative(dz=move, feed_mm_per_min=180)
            self.stage.wait_for_moves()
            time.sleep(0.02)
            img = self.camera.snap()
            if img is None:
                continue
            fine_samples.append((best_dz + offset, metric_value(img, metric)))
            cumulative = offset

        if not fine_samples:
            # Return to coarse best if no fine samples were collected
            self.stage.move_relative(dz=-cumulative, feed_mm_per_min=240)
            self.stage.wait_for_moves()
            return best_dz

        best_fine_dz, _ = max(fine_samples, key=lambda t: t[1])
        # Move to the best fine position
        self.stage.move_relative(dz=(best_fine_dz - (best_dz + cumulative)), feed_mm_per_min=240)
        self.stage.wait_for_moves()
        return best_fine_dz
