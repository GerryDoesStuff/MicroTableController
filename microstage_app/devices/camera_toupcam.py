from __future__ import annotations
import importlib
import threading
import time
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
    """
    Pull-mode RGB24/RAW8 capture with adjustable controls:
      - exposure, gain, auto-exposure
      - RAW8 (fast mono) toggle
      - resolution list (put_eSize) + ROI presets (put_Roi if available)
      - USB 'Speed/Bandwidth' level (if supported)
      - display decimation (drop frames for UI)
    """

    def __init__(self, tp, dev_id, name):
        self._tp = tp
        self._id = dev_id
        self._name = name

        self._cam = None
        self._buf = None      # Python 'bytes' buffer for PullImageV2
        self._w = self._h = 0
        self._stride = 0
        self._bits = 24       # 24 = RGB24, 8 = RAW8/mono
        self._raw_mode = False

        self._last = None     # np.uint8 HxWx(3)
        self._lock = threading.Lock()
        self._first_logged = False

        # stats / throttling
        self._display_every = 1
        self._event_count = 0
        self._fps = 0.0
        self._fps_n = 0
        self._fps_t0 = time.time()

        self._is_streaming = False

        self._open()

    # ---------------- internal ----------------

    def _realloc_buffer(self):
        # bytes length is stride*height; the wrapper wants c_char_p
        rowbytes = ((self._w * self._bits + 31) // 32) * 4
        self._stride = rowbytes
        self._buf = bytes(self._stride * self._h)
        log(f"Camera: size={self._w}x{self._h}, bits={self._bits}, stride={self._stride}, buf={len(self._buf)}B")

    def _update_dimensions(self):
        """Refresh width/height using final output size after ROI/resolution."""
        try:
            if hasattr(self._cam, "get_FinalSize"):
                self._w, self._h = self._cam.get_FinalSize()
            else:
                self._w, self._h = self._cam.get_Size()
        except Exception:
            self._w, self._h = self._cam.get_Size()
        self._realloc_buffer()

    def _force_rgb_or_raw(self):
        # 0 = RGB, 1 = RAW (per SDK); not all models implement this option
        try:
            if hasattr(self._tp, "TOUPCAM_OPTION_RAW"):
                self._cam.put_Option(self._tp.TOUPCAM_OPTION_RAW, 1 if self._raw_mode else 0)
        except Exception as e:
            log(f"Camera: RAW toggle not supported: {e}")

    def _open(self):
        log(f"Camera: opening {self._name}")
        self._cam = self._tp.Toupcam.Open(self._id)
        if not self._cam:
            raise RuntimeError("Toupcam.Open returned null")

        # default: RGB24, query size, allocate
        self._raw_mode = False
        self._bits = 24
        self._force_rgb_or_raw()
        self._update_dimensions()

        def _on_event(evt, ctx=None):
            try:
                if evt != getattr(self._tp, "TOUPCAM_EVENT_IMAGE", 0x0001) or self._cam is None:
                    return

                self._event_count += 1
                if (self._event_count % max(1, self._display_every)) != 0:
                    # drain but skip UI update to reduce CPU load
                    self._cam.PullImageV2(self._buf, self._bits, None)
                    return

                # Pull one frame
                self._cam.PullImageV2(self._buf, self._bits, None)

                # Update FPS
                self._fps_n += 1
                now = time.time()
                if now - self._fps_t0 >= 0.5:
                    self._fps = self._fps_n / (now - self._fps_t0)
                    self._fps_n = 0
                    self._fps_t0 = now

                mv = memoryview(self._buf)
                arr = np.frombuffer(mv, dtype=np.uint8).reshape(self._h, self._stride)
                if self._bits == 24:
                    bgr = arr[:, : self._w * 3].reshape(self._h, self._w, 3)
                    rgb = bgr[..., ::-1].copy()
                else:  # 8-bit RAW/mono preview
                    mono = arr[:, : self._w].reshape(self._h, self._w)
                    rgb = np.repeat(mono[..., None], 3, axis=2).copy()

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

        # Sensible defaults
        try:
            self._cam.put_AutoExpoEnable(0)
        except Exception:
            pass

        self._is_streaming = True

    # ---------------- public API used by UI ----------------

    def name(self):
        return f"Toupcam ({self._name})"

    def start_stream(self):
        if self._cam is None:
            self._open()
            return

        if self._is_streaming:
            return

        try:
            if self._buf is None:
                self._update_dimensions()

            self._force_rgb_or_raw()
            try:
                self._cam.put_AutoExpoEnable(0)
            except Exception:
                pass

            try:
                self._cam.StartPullModeWithCallback(self._on_event, self)
            except TypeError:
                self._cam.StartPullModeWithCallback(self._on_event)
            log("Camera: pull mode started")
            self._is_streaming = True
        except Exception as e:
            log(f"Camera: start_stream error: {e}")

    def stop_stream(self):
        try:
            if self._cam:
                self._cam.Stop()
                log("Camera: stopped")
        except Exception as e:
            log(f"Camera: stop error: {e}")
        finally:
            self._is_streaming = False
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

    def get_fps(self) -> float:
        return float(self._fps)

    # ---- performance knobs ----

    def set_display_decimation(self, n: int):
        self._display_every = max(1, int(n))
        log(f"Camera: display every {self._display_every} frame(s)")

    def list_resolutions(self):
        """Return [(index, w, h), ...] for video sizes, if supported."""
        out = []
        try:
            n = self._cam.get_ResolutionNumber()
            for i in range(n):
                w, h = self._cam.get_Resolution(i)
                out.append((i, w, h))
        except Exception:
            # fallback: just report current
            out.append((0, self._w, self._h))
        return out

    def set_resolution_index(self, idx: int):
        try:
            self._cam.put_eSize(int(idx))
            self._update_dimensions()
            log(f"Camera: resolution index={idx} -> {self._w}x{self._h}")
        except Exception as e:
            log(f"Camera: set_resolution_index failed: {e}")

    def set_center_roi(self, w: int, h: int):
        """Center a ROI via put_Roi if supported; otherwise try put_Size."""
        try:
            if hasattr(self._cam, "put_Roi"):
                if w <= 0 or h <= 0:
                    # clear ROI
                    self._cam.put_Roi(0, 0, 0, 0)
                    log("Camera: ROI cleared")
                else:
                    w = max(16, int(w)); h = max(16, int(h))
                    # center in current sensor size
                    cw, ch = self._cam.get_Size()
                    x = max(0, (cw - w) // 2); y = max(0, (ch - h) // 2)
                    self._cam.put_Roi(x, y, w, h)
                    log(f"Camera: ROI {x},{y},{w},{h}")
                self._update_dimensions()
            else:
                # fall back to put_Size if exposed by wrapper
                if hasattr(self._cam, "put_Size"):
                    w = max(16, int(w)); h = max(16, int(h))
                    self._cam.put_Size(w, h)
                    self._update_dimensions()
                    log(f"Camera: Size {w}x{h}")
                else:
                    log("Camera: ROI/Size not supported by this wrapper")
        except Exception as e:
            log(f"Camera: set_center_roi failed: {e}")

    def set_raw_fast_mono(self, enable: bool):
        self._raw_mode = bool(enable)
        self._bits = 8 if self._raw_mode else 24
        try:
            self._force_rgb_or_raw()
        finally:
            # size unchanged, but stride may change with bits
            self._update_dimensions()
            log(f"Camera: RAW8 fast mono={'ON' if self._raw_mode else 'OFF'}")

    def set_speed_level(self, level: int):
        """Adjust USB bandwidth/speed if the SDK exposes it."""
        try:
            # Some SDKs expose put_Speed / get_Speed; others via put_Option
            if hasattr(self._cam, "put_Speed"):
                self._cam.put_Speed(int(level))
            elif hasattr(self._tp, "TOUPCAM_OPTION_SPEED"):
                self._cam.put_Option(self._tp.TOUPCAM_OPTION_SPEED, int(level))
            else:
                log("Camera: speed option not supported")
                return
            log(f"Camera: speed level set to {level}")
        except Exception as e:
            log(f"Camera: set_speed_level failed: {e}")

    def set_exposure_ms(self, ms: float, auto: bool = False):
        """Set exposure time in milliseconds to match the UI units."""
        us = int(ms * 1000.0)
        self.set_exposure_us(us, auto)

    def set_exposure_us(self, us: int, auto: bool = False):
        try:
            if hasattr(self._cam, "put_AutoExpoEnable"):
                self._cam.put_AutoExpoEnable(1 if auto else 0)
            if not auto:
                self._cam.put_ExpoTime(int(us))
            log(f"Camera: exposure {'auto' if auto else f'{us/1000.0:.3f} ms'}")
        except Exception as e:
            log(f"Camera: set_exposure_us failed: {e}")

    def set_gain(self, again: int):
        try:
            self._cam.put_ExpoAGain(int(again))
            log(f"Camera: gain {again}")
        except Exception as e:
            log(f"Camera: set_gain failed: {e}")
