# microstage_app/devices/camera_toupcam.py
from __future__ import annotations
import importlib
import threading
import numpy as np
from ..utils.log import log

def _import_toupcam():
    try:
        return importlib.import_module("toupcam")
    except Exception as e1:
        try:
            return importlib.import_module("microstage_app.toupcam")
        except Exception as e2:
            log(f"Camera: toupcam import failed: {e1} / {e2}")
            raise

def create_camera():
    try:
        tp = _import_toupcam()
    except Exception:
        from .camera_mock import MockCamera
        return MockCamera()

    devs = tp.Toupcam.EnumV2() or []
    log(f"Camera: EnumV2 found {len(devs)} device(s).")
    if not devs:
        from .camera_mock import MockCamera
        return MockCamera()

    return ToupcamCamera(tp, devs[0].id, devs[0].displayname)

class ToupcamCamera:
    """RGB24 pull-mode capture using vendor wrapper expectations:
       - StartPullModeWithCallback(cb, ctx)
       - PullImageV2(buf: bytes, 24, None)  (buffer must be Python 'bytes')
    """
    def __init__(self, tp, dev_id, name):
        self._tp = tp
        self._id = dev_id
        self._name = name
        self._cam = None
        self._buf = None          # type: bytes
        self._w = self._h = 0
        self._stride = 0
        self._last = None
        self._lock = threading.Lock()
        self._first_logged = False
        self._open()

    def _force_rgb_mode(self):
        try:
            if hasattr(self._tp, "TOUPCAM_OPTION_RAW"):
                # 0 = RGB (not RAW/Bayer)
                self._cam.put_Option(self._tp.TOUPCAM_OPTION_RAW, 0)
        except Exception as e:
            log(f"Camera: couldn't force RGB mode (continuing): {e}")

    def _open(self):
        log(f"Camera: opening {self._name}")
        self._cam = self._tp.Toupcam.Open(self._id)
        if not self._cam:
            raise RuntimeError("Toupcam.Open returned null")
        self._force_rgb_mode()

        self._w, self._h = self._cam.get_Size()
        self._stride = ((self._w * 24 + 31) // 32) * 4
        bufsize = self._stride * self._h

        # IMPORTANT: vendor Python expects c_char_p -> Python 'bytes'
        self._buf = bytes(bufsize)
        log(f"Camera: size={self._w}x{self._h}, stride={self._stride}, buf={bufsize}B")

        def _on_event(evt, ctx=None):
            try:
                if evt != getattr(self._tp, "TOUPCAM_EVENT_IMAGE", 0x0001) or self._cam is None:
                    return
                # Pull one RGB24 frame; SDK fills rowPitch when None
                self._cam.PullImageV2(self._buf, 24, None)

                # Read the immutable bytes via a memoryview/bytearray, then NumPy
                # (pattern used in published Toupcam Python examples)
                mv = memoryview(self._buf)
                flat = np.frombuffer(mv, dtype=np.uint8)
                if flat.size < self._stride * self._h:
                    return  # safety
                arr = flat.reshape(self._h, self._stride)
                bgr = arr[:, : self._w * 3].reshape(self._h, self._w, 3)
                rgb = bgr[..., ::-1].copy()

                with self._lock:
                    self._last = rgb
                if not self._first_logged:
                    log(f"Camera: first frame {self._w}x{self._h}")
                    self._first_logged = True
            except Exception as e:
                log(f"Camera: PullImage error: {e}")

        self._on_event = _on_event
        try:
            self._cam.StartPullModeWithCallback(self._on_event, self)
        except TypeError:
            self._cam.StartPullModeWithCallback(self._on_event)
        log("Camera: pull mode started")

    def name(self): return f"Toupcam ({self._name})"
    def start_stream(self): pass

    def stop_stream(self):
        try:
            if self._cam:
                self._cam.Stop()
                log("Camera: stopped")
        except Exception as e:
            log(f"Camera: stop error: {e}")
        finally:
            try:
                if self._cam:
                    self._cam.Close()
            finally:
                self._cam = None
                self._buf = None

    def get_latest_frame(self):
        with self._lock:
            return None if self._last is None else self._last.copy()

    def snap(self):
        return self.get_latest_frame()
