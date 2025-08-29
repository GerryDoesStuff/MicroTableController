try:
    import cv2
except Exception:  # pragma: no cover - handled gracefully when OpenCV missing
    cv2 = None


class WebcamCamera:
    """Minimal USB webcam wrapper using OpenCV's :class:`VideoCapture`.

    This class mirrors the subset of :class:`MockCamera`'s API that the
    application relies on. It supports starting/stopping a video stream,
    grabbing the latest frame, taking a snapshot, and choosing between a couple
    of common resolutions.
    """

    def __init__(self, index: int = 0):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for WebcamCamera")
        self._index = int(index)
        self._cap = None
        self._running = False
        self._latest = None
        self._resolution_idx = 0
        # A small set of typical webcam resolutions
        self._resolutions = [
            (0, 640, 480),
            (1, 320, 240),
        ]

    # ------------------------------------------------------------------
    # Basic stream control
    def name(self):
        return f"Webcam {self._index}"

    def start_stream(self):
        if self._running:
            return
        self._cap = cv2.VideoCapture(self._index)
        self._apply_resolution()
        self._running = True

    def stop_stream(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._running = False

    # ------------------------------------------------------------------
    def _apply_resolution(self):
        try:
            _, w, h = self.list_resolutions()[self._resolution_idx]
        except Exception:
            _, w, h = self.list_resolutions()[0]
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))

    def get_latest_frame(self):
        if not self._running:
            self.start_stream()
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam")
        # Convert BGR -> RGB for consistency with other drivers
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._latest = frame
        return frame

    def snap(self):
        return self.get_latest_frame()

    # ------------------------------------------------------------------
    # Resolution handling
    def list_resolutions(self):
        return list(self._resolutions)

    def set_resolution_index(self, idx):
        self._resolution_idx = int(idx)
        self._apply_resolution()

    def get_resolution_index(self):
        return int(self._resolution_idx)
