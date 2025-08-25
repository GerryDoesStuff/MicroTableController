from __future__ import annotations
import importlib
import threading
import time
import ctypes
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
        # bytearray buffer for PullImageV2; reshaped view cached in _arr
        self._buf = None
        self._buf_ptr = None
        self._arr = None
        self._w = self._h = 0
        self._stride = 0
        self._bits = 24       # 24 = RGB24, 8 = RAW8/mono
        self._raw_mode = False
        # Original sensor dimensions (without ROI) used for ROI coordinates
        self._sensor_w = 0
        self._sensor_h = 0

        # cache for probed resolution list
        self._res_cache = None

        self._last = None     # np.uint8 HxWx(3)
        self._lock = threading.Lock()
        self._first_logged = False

        # stats / throttling
        self._display_every = 1
        self._event_count = 0
        self._fps = 0.0
        self._fps_n = 0
        self._fps_t0 = time.time()
        # frame timing diagnostics
        self._pull_acc = 0.0
        self._proc_acc = 0.0
        self._avg_pull_ms = 0.0
        self._avg_proc_ms = 0.0

        self._is_streaming = False

        self._open()

    # ---------------- internal ----------------

    def _realloc_buffer(self):
        # bytes length is stride*height; the wrapper wants c_char_p
        rowbytes = ((self._w * self._bits + 31) // 32) * 4
        self._stride = rowbytes
        # use mutable bytearray so the SDK can fill in-place and cache a NumPy view
        self._buf = bytearray(self._stride * self._h)
        self._buf_ptr = (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
        self._arr = np.frombuffer(self._buf, dtype=np.uint8).reshape(self._h, self._stride)
        log(
            f"Camera: size={self._w}x{self._h}, bits={self._bits}, stride={self._stride}, buf={len(self._buf)}B"
        )

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

    def _init_usb_and_speed(self):
        """Query USB type and maximize bandwidth if possible."""
        usb_code = None
        try:
            if hasattr(self._cam, "get_UsbType"):
                usb_code = self._cam.get_UsbType()
            elif hasattr(self._cam, "get_Option") and hasattr(self._tp, "TOUPCAM_OPTION_USB_TYPE"):
                usb_code = self._cam.get_Option(self._tp.TOUPCAM_OPTION_USB_TYPE)
        except Exception as e:
            log(f"Camera: USB type query failed: {e}")

        if usb_code is not None:
            usb_map = {
                0: "Unknown",
                1: "USB 1.x",
                2: "USB 2.0",
                3: "USB 3.0",
                4: "USB 3.1",
                5: "USB 3.2",
            }
            desc = usb_map.get(int(usb_code), f"code {usb_code}")
            log(f"Camera: USB type {desc}")
        else:
            # fall back to capability flags from enumeration
            try:
                devs = self._tp.Toupcam.EnumV2() or []
                for d in devs:
                    if d.id == self._id:
                        flags = getattr(d.model, "flag", 0)
                        if flags & getattr(self._tp, "TOUPCAM_FLAG_USB30_OVER_USB20", 0):
                            log("Camera: USB 3.x over USB 2.0 port")
                        elif flags & getattr(self._tp, "TOUPCAM_FLAG_USB30", 0):
                            log("Camera: USB 3.x")
                        elif flags & getattr(self._tp, "TOUPCAM_FLAG_USB32", 0):
                            log("Camera: USB 3.2")
                        else:
                            log("Camera: USB 2.0 or unknown")
                        break
            except Exception:
                log("Camera: USB type unknown")

        # ensure highest bandwidth mode if supported
        try:
            maxspeed = self._cam.get_MaxSpeed()
            cur = self._cam.get_Speed() if hasattr(self._cam, "get_Speed") else None
            log(f"Camera: speed levels 0-{maxspeed}, current {cur}")
            if cur is None or cur < maxspeed:
                self.set_speed_level(maxspeed)
        except Exception as e:
            log(f"Camera: speed query failed: {e}")

    def _open(self):
        log(f"Camera: opening {self._name}")
        self._cam = self._tp.Toupcam.Open(self._id)
        if not self._cam:
            raise RuntimeError("Toupcam.Open returned null")

        # default: RGB24, query size, allocate
        self._raw_mode = False
        self._bits = 24
        self._force_rgb_or_raw()
        # Record the full sensor dimensions for ROI calculations
        try:
            self._sensor_w, self._sensor_h = self._cam.get_Size()
        except Exception:
            self._sensor_w = self._sensor_h = 0
        self._update_dimensions()
        self._init_usb_and_speed()

        def _on_event(evt, ctx=None):
            try:
                if evt != getattr(self._tp, "TOUPCAM_EVENT_IMAGE", 0x0001) or self._cam is None:
                    return

                self._event_count += 1
                if (self._event_count % max(1, self._display_every)) != 0:
                    # drain but skip UI update to reduce CPU load
                    self._cam.PullImageV2(self._buf_ptr, self._bits, None)
                    return

                t0 = time.perf_counter()
                self._cam.PullImageV2(self._buf_ptr, self._bits, None)
                t1 = time.perf_counter()

                # Update FPS
                self._fps_n += 1
                now = time.time()
                if now - self._fps_t0 >= 0.5:
                    self._fps = self._fps_n / (now - self._fps_t0)
                    self._avg_pull_ms = (self._pull_acc / max(1, self._fps_n)) * 1000.0
                    self._avg_proc_ms = (self._proc_acc / max(1, self._fps_n)) * 1000.0
                    self._fps_n = 0
                    self._fps_t0 = now
                    self._pull_acc = 0.0
                    self._proc_acc = 0.0

                arr = self._arr
                if self._bits == 24:
                    bgr = arr[:, : self._w * 3].reshape(self._h, self._w, 3)
                    img = bgr[..., ::-1].copy()
                else:  # 8-bit RAW/mono preview
                    # Keep the grayscale frame instead of expanding to RGB.
                    # Converting to 3-channel was creating extra copies that
                    # slowed down the raw path and negated its bandwidth
                    # advantage.
                    img = arr[:, : self._w].reshape(self._h, self._w).copy()

                t2 = time.perf_counter()
                self._pull_acc += (t1 - t0)
                self._proc_acc += (t2 - t1)

                with self._lock:
                    self._last = img

                if not self._first_logged:
                    log(f"Camera: first frame {self._w}x{self._h}")
                    self._first_logged = True

            except Exception as e:
                log(f"Camera: PullImage error: {e}")

        self._on_event = _on_event
        # Stream startup is deferred to start_stream()

    # ---------------- public API used by UI ----------------

    def name(self):
        return f"Toupcam ({self._name})"

    def start_stream(self):
        if self._cam is None:
            self._open()

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
                self._buf_ptr = None

    def get_latest_frame(self):
        with self._lock:
            return None if self._last is None else self._last.copy()

    def snap(self):
        return self.get_latest_frame()

    def get_fps(self) -> float:
        return float(self._fps)

    def get_capture_stats(self):
        """Return recent FPS and average pull/processing times in milliseconds."""
        return {
            "fps": float(self._fps),
            "pull_ms": float(self._avg_pull_ms),
            "proc_ms": float(self._avg_proc_ms),
        }

    # ---- performance knobs ----

    def set_display_decimation(self, n: int):
        self._display_every = max(1, int(n))
        log(f"Camera: display every {self._display_every} frame(s)")

    def _probe_resolutions(self):
        """Attempt to discover valid resolutions when enumeration is unavailable."""
        if self._res_cache is not None:
            return self._res_cache

        # stop streaming temporarily as some cameras require this for size changes
        was_streaming = False
        try:
            if self._is_streaming:
                self._cam.Stop()
                was_streaming = True
        except Exception:
            was_streaming = False

        # remember current settings
        cur_idx = None
        cur_size = (self._w, self._h)
        try:
            cur_idx = int(self._cam.get_eSize())
        except Exception:
            cur_idx = None

        candidates = []
        try:
            n = self._cam.get_ResolutionNumber()
            for i in range(n):
                w, h = self._cam.get_Resolution(i)
                candidates.append((i, w, h))
        except Exception:
            candidates = []

        # Always try halving the sensor size to discover additional modes
        sw = self._sensor_w or self._w
        sh = self._sensor_h or self._h
        if sw and sh:
            candidates.append((None, sw, sh))
            for n in range(1, 5):
                w = sw // (2 ** n)
                h = sh // (2 ** n)
                if w <= 0 or h <= 0:
                    break
                candidates.append((None, w, h))

        found = []
        seen = set()
        for idx, w, h in candidates:
            try:
                if idx is not None and hasattr(self._cam, "put_eSize"):
                    self._cam.put_eSize(idx)
                elif hasattr(self._cam, "put_Size"):
                    self._cam.put_Size(w, h)
                else:
                    continue
                rw, rh = self._cam.get_Size()
                if rw == w and rh == h and (rw, rh) not in seen:
                    found.append((idx if idx is not None else len(found), rw, rh))
                    seen.add((rw, rh))
            except Exception:
                continue

        # restore previous resolution
        try:
            if cur_idx is not None and hasattr(self._cam, "put_eSize"):
                self._cam.put_eSize(cur_idx)
            elif hasattr(self._cam, "put_Size"):
                self._cam.put_Size(*cur_size)
        except Exception:
            pass
        self._update_dimensions()

        if was_streaming:
            try:
                self.start_stream()
            except Exception:
                pass

        self._res_cache = found if found else [(0, cur_size[0], cur_size[1])]
        return self._res_cache

    def list_resolutions(self):
        """Return [(index, w, h), ...] for video sizes, if supported."""
        out = []
        try:
            n = self._cam.get_ResolutionNumber()
            for i in range(n):
                w, h = self._cam.get_Resolution(i)
                out.append((i, w, h))
        except Exception:
            pass

        seen = set()
        uniq = []
        for idx, w, h in out:
            if (w, h) not in seen:
                seen.add((w, h))
                uniq.append((idx, w, h))

        if len(uniq) < 4:
            try:
                probes = self._probe_resolutions()
                for idx, w, h in probes:
                    if (w, h) not in seen:
                        seen.add((w, h))
                        uniq.append((idx, w, h))
                if not uniq:
                    return probes
            except Exception:
                if not uniq:
                    return self._probe_resolutions()

        return uniq

    def get_resolution_index(self) -> int:
        """Return current resolution index if available."""
        try:
            return int(self._cam.get_eSize())
        except Exception:
            try:
                for idx, w, h in self.list_resolutions():
                    if w == self._w and h == self._h:
                        return int(idx)
            except Exception:
                pass
            return 0

    def set_resolution_index(self, idx: int):
        try:
            self._cam.put_eSize(int(idx))
            # Resolution change resets the full sensor size
            try:
                self._sensor_w, self._sensor_h = self._cam.get_Size()
            except Exception:
                pass
            self._update_dimensions()
            log(f"Camera: resolution index={idx} -> {self._w}x{self._h}")
        except Exception as e:
            log(f"Camera: set_resolution_index failed: {e}")

    def set_center_roi(self, w: int, h: int):
        """Center a ROI via put_Roi if supported; otherwise try put_Size."""
        was_streaming = False
        try:
            if hasattr(self._cam, "put_Roi"):
                was_streaming = self._is_streaming
                if was_streaming:
                    try:
                        self._cam.Stop()
                    except Exception:
                        pass
                    self._is_streaming = False

                if w <= 0 or h <= 0:
                    # clear ROI and refresh full sensor size
                    self._cam.put_Roi(0, 0, 0, 0)
                    try:
                        self._sensor_w, self._sensor_h = self._cam.get_Size()
                    except Exception:
                        pass
                    log("Camera: ROI cleared")
                else:
                    w = max(16, min(int(w), self._sensor_w))
                    h = max(16, min(int(h), self._sensor_h))
                    # enforce even alignment to avoid sensor stride issues
                    w &= ~1
                    h &= ~1
                    # center in original sensor coordinates
                    x = max(0, (self._sensor_w - w) // 2)
                    y = max(0, (self._sensor_h - h) // 2)
                    x &= ~1
                    y &= ~1
                    self._cam.put_Roi(x, y, w, h)
                    log(f"Camera: ROI {x},{y},{w},{h}")

            else:
                # fall back to put_Size if exposed by wrapper
                if hasattr(self._cam, "put_Size"):
                    w = max(16, int(w))
                    h = max(16, int(h))
                    w &= ~1
                    h &= ~1
                    self._cam.put_Size(w, h)
                    try:
                        self._sensor_w, self._sensor_h = self._cam.get_Size()
                    except Exception:
                        pass
                    log(f"Camera: Size {w}x{h}")
                else:
                    log("Camera: ROI/Size not supported by this wrapper")
        except Exception as e:
            log(f"Camera: set_center_roi failed: {e}")
        finally:
            # Always refresh dimensions so buffer/stride match the new ROI
            try:
                self._update_dimensions()
            except Exception:
                pass
            if was_streaming and not self._is_streaming:
                try:
                    self._cam.StartPullModeWithCallback(self._on_event, self)
                except TypeError:
                    self._cam.StartPullModeWithCallback(self._on_event)
                self._is_streaming = True

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

    def get_exposure_ms(self) -> float:
        try:
            return float(self._cam.get_ExpoTime()) / 1000.0
        except Exception as e:
            log(f"Camera: get_exposure_ms failed: {e}")
            return 0.0

    def get_gain(self) -> int:
        try:
            return int(self._cam.get_ExpoAGain())
        except Exception as e:
            log(f"Camera: get_gain failed: {e}")
            return 0

    # ---- image controls ----

    def get_brightness(self) -> int:
        try:
            return int(self._cam.get_Brightness())
        except Exception as e:
            log(f"Camera: get_brightness failed: {e}")
            return 0

    def set_brightness(self, val: int):
        try:
            self._cam.put_Brightness(int(val))
            log(f"Camera: brightness {val}")
        except Exception as e:
            log(f"Camera: set_brightness failed: {e}")

    def get_contrast(self) -> int:
        try:
            return int(self._cam.get_Contrast())
        except Exception as e:
            log(f"Camera: get_contrast failed: {e}")
            return 0

    def set_contrast(self, val: int):
        try:
            self._cam.put_Contrast(int(val))
            log(f"Camera: contrast {val}")
        except Exception as e:
            log(f"Camera: set_contrast failed: {e}")

    def get_saturation(self) -> int:
        try:
            return int(self._cam.get_Saturation())
        except Exception as e:
            log(f"Camera: get_saturation failed: {e}")
            return 0

    def set_saturation(self, val: int):
        try:
            self._cam.put_Saturation(int(val))
            log(f"Camera: saturation {val}")
        except Exception as e:
            log(f"Camera: set_saturation failed: {e}")

    def get_hue(self) -> int:
        try:
            return int(self._cam.get_Hue())
        except Exception as e:
            log(f"Camera: get_hue failed: {e}")
            return 0

    def set_hue(self, val: int):
        try:
            self._cam.put_Hue(int(val))
            log(f"Camera: hue {val}")
        except Exception as e:
            log(f"Camera: set_hue failed: {e}")

    def get_gamma(self) -> int:
        try:
            return int(self._cam.get_Gamma())
        except Exception as e:
            log(f"Camera: get_gamma failed: {e}")
            return 0

    def set_gamma(self, val: int):
        try:
            self._cam.put_Gamma(int(val))
            log(f"Camera: gamma {val}")
        except Exception as e:
            log(f"Camera: set_gamma failed: {e}")
