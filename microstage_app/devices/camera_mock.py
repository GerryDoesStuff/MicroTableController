import numpy as np
import time

class MockCamera:
    def __init__(self):
        self._running = False
        self._latest = None
        self._t = 0.0

    def name(self): return "MockCamera"
    def start_stream(self): self._running = True
    def stop_stream(self): self._running = False

    def get_latest_frame(self):
        h, w = 480, 640
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        img = (x + y + 0.2*np.sin(10*(x+y+self._t))) % 1.0
        self._t += 0.05
        rgb = np.dstack([img, img, img])
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def snap(self):
        return self.get_latest_frame()
