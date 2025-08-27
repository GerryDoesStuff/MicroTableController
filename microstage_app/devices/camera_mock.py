import numpy as np
import time

class MockCamera:
    def __init__(self):
        self._running = False
        self._latest = None
        self._t = 0.0
        self._resolution_idx = 0
        self._exposure_ms = 10.0
        self._auto = False
        self._gain = 100

    def name(self): return "MockCamera"
    def start_stream(self): self._running = True
    def stop_stream(self): self._running = False

    def get_latest_frame(self):
        """Generate a grayscale gradient image at the selected resolution."""
        try:
            _, w, h = self.list_resolutions()[self._resolution_idx]
        except Exception:
            # Fall back to the first resolution if index is invalid
            _, w, h = self.list_resolutions()[0]

        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        img = (x + y + 0.2 * np.sin(10 * (x + y + self._t))) % 1.0
        self._t += 0.05
        rgb = np.dstack([img, img, img])
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def snap(self):
        return self.get_latest_frame()

    # Mock resolution handling -------------------------------------------------

    def list_resolutions(self):
        """Return a small list of fake resolution tuples."""
        return [
            (0, 640, 480),
            (1, 320, 240),
        ]

    def set_resolution_index(self, idx):
        self._resolution_idx = int(idx)

    def get_resolution_index(self):
        return int(self._resolution_idx)

    # Mock exposure/gain -------------------------------------------------------

    def set_exposure_ms(self, ms, auto=False):
        self._exposure_ms = float(ms)
        self._auto = bool(auto)

    def get_exposure_ms(self):
        return float(self._exposure_ms)

    def set_gain(self, gain):
        self._gain = int(gain)

    def get_gain(self):
        return float(self._gain) / 100.0
