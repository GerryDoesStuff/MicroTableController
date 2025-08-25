from PySide6 import QtWidgets, QtCore, QtGui

from ..devices.stage_marlin import StageMarlin, find_marlin_port
from ..devices.camera_toupcam import create_camera

from ..control.autofocus import FocusMetric, AutoFocus
from ..control.raster import RasterRunner, RasterConfig
from ..control.profiles import Profiles
from ..io.storage import ImageWriter

from ..utils.img import numpy_to_qimage
from ..utils.log import LOG, log
from ..utils.serial_worker import SerialWorker
from ..utils.workers import run_async

from pathlib import Path
import re
import time


def _load_stage_bounds():
    cfg = Path(__file__).resolve().parents[2] / "marlin/Marlin-2.1.3-b3/Marlin/Configuration.h"
    try:
        text = cfg.read_text()
    except Exception as e:
        LOG.warning("Failed to load stage bounds: %s", e)
        return None
    def _parse(pattern, name):
        m = re.search(pattern, text)
        if not m:
            LOG.warning("Stage bounds: failed to parse %s", name)
            return None
        return float(m.group(1))
    x = _parse(r"#define\s+X_BED_SIZE\s+(\d+)", "X_BED_SIZE")
    y = _parse(r"#define\s+Y_BED_SIZE\s+(\d+)", "Y_BED_SIZE")
    z = _parse(r"#define\s+Z_MAX_POS\s+(\d+)", "Z_MAX_POS")
    if x is None or y is None or z is None:
        return None
    return {"xmin": 0.0, "xmax": x, "ymin": 0.0, "ymax": y, "zmin": 0.0, "zmax": z}


def _load_feed_limits():
    cfg = Path(__file__).resolve().parents[2] / "marlin/Marlin-2.1.3-b3/Marlin/Configuration.h"
    try:
        text = cfg.read_text()
        m = re.search(r"DEFAULT_MAX_FEEDRATE\s*{([^}]+)}", text)
        if not m:
            return None
        vals = [float(v.strip()) for v in m.group(1).split(',')[:3]]
        return [v * 60.0 for v in vals]  # mm/min
    except Exception as e:
        LOG.warning("Failed to load feed limits: %s", e)
        return None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroStage App v0.1")
        self.resize(1400, 900)

        # device handles
        self.stage = None
        self.camera = None
        self.stage_bounds = _load_stage_bounds()
        self._stage_bounds_fallback = self.stage_bounds.copy() if self.stage_bounds else None

        # persistent serial worker
        self.stage_thread = None
        self.stage_worker = None

        # async connect helper refs
        self._conn_thread = None
        self._conn_worker = None

        # background op refs (prevent GC while running)
        self._last_thread = None
        self._last_worker = None

        # image writer (per-run folder)
        self.image_writer = ImageWriter()

        # profiles
        self.profiles = Profiles.load_or_create()

        # timers
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(33)          # ~30 FPS poll
        self.preview_timer.timeout.connect(self._on_preview)
        self.fps_timer = QtCore.QTimer(self)
        self.fps_timer.setInterval(500)             # update FPS label
        self.fps_timer.timeout.connect(self._update_fps)
        self.pos_timer = QtCore.QTimer(self)
        self.pos_timer.setInterval(250)
        self.pos_timer.timeout.connect(self._poll_stage_position)

        # jog hold / repeat
        self._jog_hold_timer = QtCore.QTimer(self)
        self._jog_hold_timer.setSingleShot(True)
        self._jog_hold_timer.timeout.connect(self._start_jog_repeat)
        self._jog_dir = None

        # UI
        self._build_ui()
        self._connect_signals()

        # mirror logs to the in-app log pane
        LOG.message.connect(self._append_log)

        # show window first, then connect devices asynchronously
        QtCore.QTimer.singleShot(0, self._auto_connect_async)

    # --------------------------- UI BUILD ---------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Left column: device + profiles
        leftw = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(leftw)
        self.stage_status = QtWidgets.QLabel("Stage: —")
        self.stage_pos = QtWidgets.QLabel("Pos: —")
        self.btn_stage_connect = QtWidgets.QPushButton("Connect Stage")
        self.btn_stage_disconnect = QtWidgets.QPushButton("Disconnect Stage")
        self.cam_status = QtWidgets.QLabel("Camera: —")
        self.btn_cam_connect = QtWidgets.QPushButton("Connect Camera")
        self.btn_cam_disconnect = QtWidgets.QPushButton("Disconnect Camera")
        self.profile_combo = QtWidgets.QComboBox()
        self.btn_reload_profiles = QtWidgets.QPushButton("Reload Profiles")
        left.addWidget(self.stage_status)
        left.addWidget(self.stage_pos)
        left.addWidget(self.btn_stage_connect)
        left.addWidget(self.btn_stage_disconnect)
        left.addSpacing(8)
        left.addWidget(self.cam_status)
        left.addWidget(self.btn_cam_connect)
        left.addWidget(self.btn_cam_disconnect)
        left.addSpacing(8)
        left.addWidget(QtWidgets.QLabel("Profile:"))
        left.addWidget(self.profile_combo)
        left.addWidget(self.btn_reload_profiles)
        left.addStretch(1)

        # Center: live preview + capture + FPS
        centerw = QtWidgets.QWidget()
        center = QtWidgets.QVBoxLayout(centerw)
        self.live_label = QtWidgets.QLabel()
        self.live_label.setMinimumSize(900, 650)
        self.live_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fps_label = QtWidgets.QLabel("FPS: —")
        self.btn_capture = QtWidgets.QPushButton("Capture")
        center.addWidget(self.live_label, 1)
        ctr2 = QtWidgets.QHBoxLayout()
        ctr2.addWidget(self.btn_capture)
        ctr2.addStretch(1)
        ctr2.addWidget(self.fps_label)
        center.addLayout(ctr2)

        # Right: tabs
        rightw = QtWidgets.QTabWidget()

        # ---- Jog tab
        jog = QtWidgets.QWidget()
        j = QtWidgets.QGridLayout(jog)
        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setDecimals(3)
        self.step_spin.setRange(0.001, 1000.0)  # allow big moves
        self.step_spin.setValue(0.100)
        self.feed_spin = QtWidgets.QDoubleSpinBox()
        limits = _load_feed_limits()
        max_feed = (min(limits) / 60.0) if limits else 1.0
        self.feed_spin.setRange(0.01, max_feed)
        self.feed_spin.setValue(20.0 / 60.0)
        self.feed_limits = QtWidgets.QLabel()
        if limits:
            self.feed_limits.setText(
                f"Limits: X {limits[0]:.0f} / Y {limits[1]:.0f} / Z {limits[2]:.0f} mm/min"
            )
        self.btn_home_all = QtWidgets.QPushButton("Home All")
        self.btn_home_x = QtWidgets.QPushButton("Home X")
        self.btn_home_y = QtWidgets.QPushButton("Home Y")
        self.btn_home_z = QtWidgets.QPushButton("Home Z")
        self.btn_xm = QtWidgets.QPushButton("X-")
        self.btn_xp = QtWidgets.QPushButton("X+")
        self.btn_ym = QtWidgets.QPushButton("Y-")
        self.btn_yp = QtWidgets.QPushButton("Y+")
        self.btn_zm = QtWidgets.QPushButton("Z-")
        self.btn_zp = QtWidgets.QPushButton("Z+")
        j.addWidget(QtWidgets.QLabel("Step (mm):"), 0, 0)
        j.addWidget(self.step_spin, 0, 1)
        j.addWidget(QtWidgets.QLabel("Feed (mm/s):"), 1, 0)
        j.addWidget(self.feed_spin, 1, 1)
        j.addWidget(self.feed_limits, 2, 0, 1, 2)
        j.addWidget(self.btn_home_all, 3, 0, 1, 2)
        j.addWidget(self.btn_home_x, 4, 0)
        j.addWidget(self.btn_home_y, 4, 1)
        j.addWidget(self.btn_home_z, 5, 0, 1, 2)
        j.addWidget(self.btn_xm, 6, 0)
        j.addWidget(self.btn_xp, 6, 1)
        j.addWidget(self.btn_ym, 7, 0)
        j.addWidget(self.btn_yp, 7, 1)
        j.addWidget(self.btn_zm, 8, 0)
        j.addWidget(self.btn_zp, 8, 1)
        rightw.addTab(jog, "Jog")

        # ---- Camera tab (performance controls)
        camtab = QtWidgets.QWidget()
        c = QtWidgets.QGridLayout(camtab)
        row = 0
        self.exp_spin = QtWidgets.QDoubleSpinBox(); self.exp_spin.setRange(0.01, 10000.0); self.exp_spin.setValue(10.0)
        self.exp_spin.setSuffix(" ms")
        self.autoexp_chk = QtWidgets.QCheckBox("Auto")
        c.addWidget(QtWidgets.QLabel("Exposure:"), row, 0); c.addWidget(self.exp_spin, row, 1); c.addWidget(self.autoexp_chk, row, 2); row += 1

        self.gain_spin = QtWidgets.QSpinBox(); self.gain_spin.setRange(1, 400); self.gain_spin.setValue(100)
        c.addWidget(QtWidgets.QLabel("Gain (AGain):"), row, 0); c.addWidget(self.gain_spin, row, 1); row += 1

        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.brightness_slider.setRange(-255, 255)
        self.brightness_spin = QtWidgets.QSpinBox(); self.brightness_spin.setRange(-255, 255); self.brightness_spin.setValue(0)
        c.addWidget(QtWidgets.QLabel("Brightness:"), row, 0); c.addWidget(self.brightness_slider, row, 1); c.addWidget(self.brightness_spin, row, 2); row += 1

        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.contrast_slider.setRange(-255, 255)
        self.contrast_spin = QtWidgets.QSpinBox(); self.contrast_spin.setRange(-255, 255); self.contrast_spin.setValue(0)
        c.addWidget(QtWidgets.QLabel("Contrast:"), row, 0); c.addWidget(self.contrast_slider, row, 1); c.addWidget(self.contrast_spin, row, 2); row += 1

        self.saturation_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.saturation_slider.setRange(0, 255)
        self.saturation_spin = QtWidgets.QSpinBox(); self.saturation_spin.setRange(0, 255); self.saturation_spin.setValue(128)
        c.addWidget(QtWidgets.QLabel("Saturation:"), row, 0); c.addWidget(self.saturation_slider, row, 1); c.addWidget(self.saturation_spin, row, 2); row += 1

        self.hue_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.hue_slider.setRange(-180, 180)
        self.hue_spin = QtWidgets.QSpinBox(); self.hue_spin.setRange(-180, 180); self.hue_spin.setValue(0)
        c.addWidget(QtWidgets.QLabel("Hue:"), row, 0); c.addWidget(self.hue_slider, row, 1); c.addWidget(self.hue_spin, row, 2); row += 1

        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.gamma_slider.setRange(20, 180)
        self.gamma_spin = QtWidgets.QSpinBox(); self.gamma_spin.setRange(20, 180); self.gamma_spin.setValue(100)
        c.addWidget(QtWidgets.QLabel("Gamma:"), row, 0); c.addWidget(self.gamma_slider, row, 1); c.addWidget(self.gamma_spin, row, 2); row += 1

        self.raw_chk = QtWidgets.QCheckBox("RAW8 fast mono (triples bandwidth efficiency)")
        c.addWidget(self.raw_chk, row, 0, 1, 3); row += 1

        self.res_combo = QtWidgets.QComboBox()
        self.btn_roi_full = QtWidgets.QPushButton("ROI: Full")
        self.btn_roi_2048 = QtWidgets.QPushButton("ROI: 2048²")
        self.btn_roi_1024 = QtWidgets.QPushButton("ROI: 1024²")
        self.btn_roi_512  = QtWidgets.QPushButton("ROI: 512²")
        c.addWidget(QtWidgets.QLabel("Resolution:"), row, 0); c.addWidget(self.res_combo, row, 1, 1, 2); row += 1
        c.addWidget(self.btn_roi_full, row, 0); c.addWidget(self.btn_roi_2048, row, 1); c.addWidget(self.btn_roi_1024, row, 2); row += 1
        c.addWidget(self.btn_roi_512, row, 0); row += 1

        self.speed_spin = QtWidgets.QSpinBox(); self.speed_spin.setRange(0, 5); self.speed_spin.setValue(0)
        c.addWidget(QtWidgets.QLabel("USB Speed/Bandwidth lvl:"), row, 0); c.addWidget(self.speed_spin, row, 1); row += 1

        self.decim_spin = QtWidgets.QSpinBox(); self.decim_spin.setRange(1, 8); self.decim_spin.setValue(1)
        c.addWidget(QtWidgets.QLabel("Display every Nth frame:"), row, 0); c.addWidget(self.decim_spin, row, 1); row += 1

        c.setRowStretch(row, 1)
        rightw.addTab(camtab, "Camera")

        # ---- Autofocus tab
        af = QtWidgets.QWidget()
        a = QtWidgets.QGridLayout(af)
        self.metric_combo = QtWidgets.QComboBox(); self.metric_combo.addItems([m.value for m in FocusMetric])
        self.af_range = QtWidgets.QDoubleSpinBox(); self.af_range.setRange(0.01, 5.0); self.af_range.setValue(0.5)
        self.af_coarse = QtWidgets.QDoubleSpinBox(); self.af_coarse.setRange(0.001, 1.0); self.af_coarse.setValue(0.05)
        self.af_fine = QtWidgets.QDoubleSpinBox(); self.af_fine.setRange(0.0005, 0.2); self.af_fine.setValue(0.01)
        self.btn_autofocus = QtWidgets.QPushButton("Run Autofocus")
        a.addWidget(QtWidgets.QLabel("Metric:"), 0, 0); a.addWidget(self.metric_combo, 0, 1)
        a.addWidget(QtWidgets.QLabel("Range (mm):"), 1, 0); a.addWidget(self.af_range, 1, 1)
        a.addWidget(QtWidgets.QLabel("Coarse step (mm):"), 2, 0); a.addWidget(self.af_coarse, 2, 1)
        a.addWidget(QtWidgets.QLabel("Fine step (mm):"), 3, 0); a.addWidget(self.af_fine, 3, 1)
        a.addWidget(self.btn_autofocus, 4, 0, 1, 2)
        rightw.addTab(af, "Autofocus")

        # ---- Raster tab
        rast = QtWidgets.QWidget()
        r = QtWidgets.QGridLayout(rast)
        self.rows_spin = QtWidgets.QSpinBox(); self.rows_spin.setRange(1, 1000); self.rows_spin.setValue(5)
        self.cols_spin = QtWidgets.QSpinBox(); self.cols_spin.setRange(1, 1000); self.cols_spin.setValue(5)
        self.pitchx_spin = QtWidgets.QDoubleSpinBox(); self.pitchx_spin.setRange(0.001, 50.0); self.pitchx_spin.setValue(1.0)
        self.pitchy_spin = QtWidgets.QDoubleSpinBox(); self.pitchy_spin.setRange(0.001, 50.0); self.pitchy_spin.setValue(1.0)
        self.btn_run_raster = QtWidgets.QPushButton("Run Raster")
        r.addWidget(QtWidgets.QLabel("Rows:"), 0, 0); r.addWidget(self.rows_spin, 0, 1)
        r.addWidget(QtWidgets.QLabel("Cols:"), 1, 0); r.addWidget(self.cols_spin, 1, 1)
        r.addWidget(QtWidgets.QLabel("Pitch X (mm):"), 2, 0); r.addWidget(self.pitchx_spin, 2, 1)
        r.addWidget(QtWidgets.QLabel("Pitch Y (mm):"), 3, 0); r.addWidget(self.pitchy_spin, 3, 1)
        r.addWidget(self.btn_run_raster, 4, 0, 1, 2)
        rightw.addTab(rast, "Raster")

        # ---- Scripts tab (restored)
        scripts = QtWidgets.QWidget()
        s = QtWidgets.QVBoxLayout(scripts)
        self.btn_run_example_script = QtWidgets.QPushButton("Run Example Script (Z stack)")
        s.addWidget(self.btn_run_example_script)
        s.addStretch(1)
        rightw.addTab(scripts, "Scripts")

        # log pane
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(8000)

        left_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left_right.addWidget(leftw)
        left_right.addWidget(centerw)
        left_right.addWidget(rightw)
        left_right.setStretchFactor(0, 0)
        left_right.setStretchFactor(1, 1)
        left_right.setStretchFactor(2, 0)

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vsplit.addWidget(left_right)
        vsplit.addWidget(self.log_view)
        vsplit.setStretchFactor(0, 1)
        vsplit.setStretchFactor(1, 0)

        root = QtWidgets.QVBoxLayout(central)
        root.addWidget(vsplit)

        self._reload_profiles()
        self._update_stage_buttons()
        self._update_cam_buttons()

    def _connect_signals(self):
        self.btn_stage_connect.clicked.connect(self._connect_stage_async)
        self.btn_stage_disconnect.clicked.connect(self._disconnect_stage)
        self.btn_cam_connect.clicked.connect(self._connect_camera)
        self.btn_cam_disconnect.clicked.connect(self._disconnect_camera)
        self.btn_capture.clicked.connect(self._capture)
        self.btn_home_all.clicked.connect(self._home_all)
        self.btn_home_x.clicked.connect(lambda: self._home_axis('x'))
        self.btn_home_y.clicked.connect(lambda: self._home_axis('y'))
        self.btn_home_z.clicked.connect(lambda: self._home_axis('z'))
        self._setup_jog_button(self.btn_xm, sx=-1)
        self._setup_jog_button(self.btn_xp, sx=1)
        self._setup_jog_button(self.btn_ym, sy=-1)
        self._setup_jog_button(self.btn_yp, sy=1)
        self._setup_jog_button(self.btn_zm, sz=-1)
        self._setup_jog_button(self.btn_zp, sz=1)
        self.btn_autofocus.clicked.connect(self._run_autofocus)
        self.btn_run_raster.clicked.connect(self._run_raster)
        self.btn_reload_profiles.clicked.connect(self._reload_profiles)

        # camera controls
        self.exp_spin.valueChanged.connect(self._apply_exposure)
        self.autoexp_chk.toggled.connect(self._apply_exposure)
        self.gain_spin.valueChanged.connect(self._apply_gain)
        self.brightness_slider.valueChanged.connect(self.brightness_spin.setValue)
        self.brightness_spin.valueChanged.connect(self.brightness_slider.setValue)
        self.brightness_spin.valueChanged.connect(self._apply_brightness)
        self.contrast_slider.valueChanged.connect(self.contrast_spin.setValue)
        self.contrast_spin.valueChanged.connect(self.contrast_slider.setValue)
        self.contrast_spin.valueChanged.connect(self._apply_contrast)
        self.saturation_slider.valueChanged.connect(self.saturation_spin.setValue)
        self.saturation_spin.valueChanged.connect(self.saturation_slider.setValue)
        self.saturation_spin.valueChanged.connect(self._apply_saturation)
        self.hue_slider.valueChanged.connect(self.hue_spin.setValue)
        self.hue_spin.valueChanged.connect(self.hue_slider.setValue)
        self.hue_spin.valueChanged.connect(self._apply_hue)
        self.gamma_slider.valueChanged.connect(self.gamma_spin.setValue)
        self.gamma_spin.valueChanged.connect(self.gamma_slider.setValue)
        self.gamma_spin.valueChanged.connect(self._apply_gamma)
        self.raw_chk.toggled.connect(self._apply_raw)
        self.res_combo.currentIndexChanged.connect(self._apply_resolution)
        self.btn_roi_full.clicked.connect(lambda: self._apply_roi('full'))
        self.btn_roi_2048.clicked.connect(lambda: self._apply_roi(2048))
        self.btn_roi_1024.clicked.connect(lambda: self._apply_roi(1024))
        self.btn_roi_512.clicked.connect(lambda: self._apply_roi(512))
        self.speed_spin.valueChanged.connect(self._apply_speed)
        self.decim_spin.valueChanged.connect(self._apply_decimation)

        # scripts
        self.btn_run_example_script.clicked.connect(self._run_example_script)

    def _setup_jog_button(self, btn, sx=0, sy=0, sz=0):
        def _pressed():
            step = self.step_spin.value()
            self._jog(step * sx, step * sy, step * sz, wait_ok=True)
            self._jog_dir = (sx, sy, sz)
            self._jog_hold_timer.start(1000)

        def _released():
            self._jog_hold_timer.stop()
            self._jog_dir = None

        btn.pressed.connect(_pressed)
        btn.released.connect(_released)

    def _start_jog_repeat(self):
        if self._jog_dir:
            self._repeat_jog()

    def _repeat_jog(self, _res=None):
        if not self._jog_dir:
            return
        sx, sy, sz = self._jog_dir
        step = self.step_spin.value()
        self._jog(step * sx, step * sy, step * sz, wait_ok=True, callback=self._repeat_jog)

    @QtCore.Slot(str)
    def _append_log(self, line: str):
        self.log_view.appendPlainText(line)

    # --------------------------- CONNECT ---------------------------

    def _auto_connect_async(self):
        self._connect_camera()
        self._connect_stage_async()

    def _attach_stage_worker(self):
        if not self.stage or self.stage_thread:
            return
        self.stage_thread = QtCore.QThread(self)
        self.stage_worker = SerialWorker(self.stage)
        self.stage_worker.moveToThread(self.stage_thread)
        self.stage_worker.result.connect(self._dispatch_stage_result)
        self.stage_thread.started.connect(self.stage_worker.loop)
        self.stage_thread.start()
        self.pos_timer.start()

    def _dispatch_stage_result(self, cb, res):
        if cb:
            cb(res)

    # --------------------------- CONNECT/DISCONNECT ---------------------------

    def _connect_camera(self):
        if self.camera is not None:
            log("UI: camera already connected; skip re-open")
            return
        try:
            cam = create_camera()
            self.camera = cam
            self.cam_status.setText(f"Camera: {self.camera.name()}")
            self.camera.start_stream()
            self._populate_resolutions()
            self._sync_cam_controls()
            self.preview_timer.start()
            self.fps_timer.start()
            log("UI: camera connected")
        except Exception as e:
            log(f"UI: camera connect failed: {e}")
        self._update_cam_buttons()

    def _disconnect_camera(self):
        if not self.camera:
            return
        try:
            self.camera.stop_stream()
        except Exception:
            pass
        self.camera = None
        self.cam_status.setText("Camera: —")
        self.preview_timer.stop()
        self.fps_timer.stop()
        self.live_label.clear()
        self._update_cam_buttons()

    def _connect_stage_async(self):
        if self.stage is not None:
            log("UI: stage already connected; skip re-probe")
            return

        def connect_stage():
            port = find_marlin_port()
            if not port:
                return None
            return StageMarlin(port)

        self._conn_thread, self._conn_worker = run_async(connect_stage)
        self._conn_worker.finished.connect(self._on_stage_connect)

    @QtCore.Slot(object, object)
    def _on_stage_connect(self, stage, err):
        if err or not stage:
            if err:
                log(f"UI: stage connect failed: {err}")
            else:
                log("UI: stage not found")
            self.stage_status.setText("Stage: not found")
            self._update_stage_buttons()
        else:
            self.stage = stage
            info = self.stage.get_info()
            name = info.get("name") or "connected"
            uuid = info.get("uuid")
            text = f"Stage: {name}"
            if uuid:
                text += f" ({uuid})"
            self.stage_status.setText(text)
            try:
                self.stage_bounds = self.stage.get_bounds()
            except Exception as e:
                log(f"Stage: failed to get bounds: {e}")
                self.stage_bounds = None
            log("UI: stage connected (async)")
            self._attach_stage_worker()
            self._update_stage_buttons()
        thread = self._conn_thread
        self._conn_thread = self._conn_worker = None
        if thread and thread != QtCore.QThread.currentThread():
            thread.wait()

    def _disconnect_stage(self):
        if self._conn_thread:
            self._conn_thread.quit()
            self._conn_thread.wait()
            self._conn_thread = None
            self._conn_worker = None
        if self.stage_worker:
            self.stage_worker.stop()
        if self.stage_thread:
            self.stage_thread.quit()
            self.stage_thread.wait(2000)
            self.stage_thread = None
            self.stage_worker = None
        if self.stage:
            try:
                self.stage.ser.close()
            except Exception:
                pass
            self.stage = None
        self.stage_status.setText("Stage: —")
        self.stage_pos.setText("Pos: —")
        self.stage_bounds = None
        self.pos_timer.stop()
        self._update_stage_buttons()

    def _update_stage_buttons(self):
        connected = self.stage is not None
        self.btn_stage_connect.setEnabled(not connected)
        self.btn_stage_disconnect.setEnabled(connected)

    def _update_cam_buttons(self):
        connected = self.camera is not None
        self.btn_cam_connect.setEnabled(not connected)
        self.btn_cam_disconnect.setEnabled(connected)

    def _poll_stage_position(self):
        if not self.stage_worker:
            return
        self.stage_worker.enqueue(self.stage.get_position, callback=self._on_stage_position)

    def _on_stage_position(self, pos):
        if not pos:
            return
        x, y, z = pos
        if x is None or y is None or z is None:
            return
        # merge hardware-reported bounds with fallback from config
        b = self.stage_bounds or {}
        fb = getattr(self, "_stage_bounds_fallback", None)
        if fb:
            if not b:
                b = fb.copy()
            else:
                for k, v in fb.items():
                    if b.get(k) is None:
                        b[k] = v
        if b:
            self.stage_bounds = b
        # always build a deterministic two-line label: first the coordinates,
        # then the limits. This ensures the limits line never precedes the
        # coordinates, even if the stage bounds are unavailable.
        coords_line = f"Pos: X{x:.3f} Y{y:.3f} Z{z:.3f}"
        if self.stage_bounds:
            b = self.stage_bounds
            limits_line = (
                f"Limits: X[{b['xmin']:.3f},{b['xmax']:.3f}] "
                f"Y[{b['ymin']:.3f},{b['ymax']:.3f}] "
                f"Z[{b['zmin']:.3f},{b['zmax']:.3f}]"
            )
        else:
            limits_line = "Limits: —"

        self.stage_pos.setText(f"{coords_line}\n{limits_line}")

    # --------------------------- PREVIEW ---------------------------

    def _on_preview(self):
        if not self.camera:
            return
        frame = self.camera.get_latest_frame()
        if frame is None:
            return
        qimg = numpy_to_qimage(frame)
        self.live_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.live_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _update_fps(self):
        if self.camera:
            try:
                self.fps_label.setText(f"FPS: {self.camera.get_fps():.1f}")
            except Exception:
                self.fps_label.setText("FPS: —")

    # --------------------------- CAMERA APPLY ---------------------------

    def _populate_resolutions(self):
        if not self.camera:
            return
        self.res_combo.blockSignals(True)
        self.res_combo.clear()
        for idx, w, h in self.camera.list_resolutions():
            self.res_combo.addItem(f"{w}×{h}", idx)
        self.res_combo.blockSignals(False)

    def _sync_cam_controls(self):
        if not self.camera:
            return
        try:
            if hasattr(self.camera, "get_brightness"):
                self.brightness_spin.setValue(int(self.camera.get_brightness()))
        except Exception:
            pass
        try:
            if hasattr(self.camera, "get_contrast"):
                self.contrast_spin.setValue(int(self.camera.get_contrast()))
        except Exception:
            pass
        try:
            if hasattr(self.camera, "get_saturation"):
                self.saturation_spin.setValue(int(self.camera.get_saturation()))
        except Exception:
            pass
        try:
            if hasattr(self.camera, "get_hue"):
                self.hue_spin.setValue(int(self.camera.get_hue()))
        except Exception:
            pass
        try:
            if hasattr(self.camera, "get_gamma"):
                self.gamma_spin.setValue(int(self.camera.get_gamma()))
        except Exception:
            pass

    def _apply_exposure(self):
        if not self.camera: return
        auto = self.autoexp_chk.isChecked()
        ms = self.exp_spin.value()
        self.camera.set_exposure_ms(ms, auto)

    def _apply_gain(self):
        if not self.camera: return
        self.camera.set_gain(int(self.gain_spin.value()))

    def _apply_brightness(self):
        if not self.camera: return
        if hasattr(self.camera, "set_brightness"):
            self.camera.set_brightness(int(self.brightness_spin.value()))

    def _apply_contrast(self):
        if not self.camera: return
        if hasattr(self.camera, "set_contrast"):
            self.camera.set_contrast(int(self.contrast_spin.value()))

    def _apply_saturation(self):
        if not self.camera: return
        if hasattr(self.camera, "set_saturation"):
            self.camera.set_saturation(int(self.saturation_spin.value()))

    def _apply_hue(self):
        if not self.camera: return
        if hasattr(self.camera, "set_hue"):
            self.camera.set_hue(int(self.hue_spin.value()))

    def _apply_gamma(self):
        if not self.camera: return
        if hasattr(self.camera, "set_gamma"):
            self.camera.set_gamma(int(self.gamma_spin.value()))

    def _apply_raw(self, on: bool):
        if not self.camera: return
        self.camera.set_raw_fast_mono(bool(on))

    def _apply_resolution(self, i: int):
        if not self.camera: return
        idx = self.res_combo.currentData()
        if idx is None: return
        self.camera.set_resolution_index(int(idx))

    def _apply_roi(self, mode):
        if not self.camera: return
        if mode == 'full':
            # reset ROI to full frame
            self.camera.set_center_roi(0, 0)
        else:
            side = int(mode)
            self.camera.set_center_roi(side, side)

    def _apply_speed(self):
        if not self.camera: return
        self.camera.set_speed_level(int(self.speed_spin.value()))

    def _apply_decimation(self):
        if not self.camera: return
        self.camera.set_display_decimation(int(self.decim_spin.value()))

    # --------------------------- STAGE OPS ---------------------------

    def _home_all(self):
        if not self.stage_worker:
            log("Home ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return
        log("Home: Z then X/Y")
        self.stage_worker.enqueue(self.stage.home_all)
        self.stage_worker.enqueue(
            self.stage.get_position, callback=self._on_stage_position
        )

    def _home_axis(self, axis: str):
        if not self.stage_worker:
            log("Home ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return
        if axis == 'x':
            fn = self.stage.home_x
        elif axis == 'y':
            fn = self.stage.home_y
        else:
            fn = self.stage.home_z
        log(f"Home axis: {axis.upper()}")
        self.stage_worker.enqueue(fn)
        self.stage_worker.enqueue(
            self.stage.get_position, callback=self._on_stage_position
        )

    def _jog(self, dx=0, dy=0, dz=0, *, wait_ok=True, callback=None):
        if not self.stage_worker:
            log("Jog ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return
        f = max(1.0, float(self.feed_spin.value()) * 60.0)  # mm/s -> mm/min
        log(f"Jog: dx={dx} dy={dy} dz={dz} F={f}")
        self.stage_worker.enqueue(
            self.stage.move_relative,
            dx,
            dy,
            dz,
            f,
            wait_ok,
            callback=callback,
        )
        self.stage_worker.enqueue(
            self.stage.get_position, callback=self._on_stage_position
        )

    # --------------------------- CAPTURE / MODES ---------------------------

    def _capture(self):
        if not (self.stage and self.camera):
            log("Capture ignored: stage or camera not connected")
            return

        def do_capture():
            self.stage.wait_for_moves()
            time.sleep(0.03)
            img = self.camera.snap()
            if img is not None:
                self.image_writer.save_single(img)
            return True

        log("Capture: starting")
        t, w = run_async(do_capture)
        self._last_thread, self._last_worker = t, w
        w.finished.connect(lambda res, err: log("Capture: done" if not err else f"Capture error: {err}"))

    def _run_autofocus(self):
        if not (self.stage and self.camera):
            log("Autofocus ignored: stage or camera not connected")
            return

        metric = FocusMetric(self.metric_combo.currentText())

        def do_af():
            af = AutoFocus(self.stage, self.camera)
            best_z = af.coarse_to_fine(
                metric=metric,
                z_range_mm=float(self.af_range.value()),
                coarse_step_mm=float(self.af_coarse.value()),
                fine_step_mm=float(self.af_fine.value()),
            )
            return best_z

        log(f"Autofocus: metric={metric.value}")
        t, w = run_async(do_af)
        self._last_thread, self._last_worker = t, w

        def _done(best, err):
            if err:
                log(f"Autofocus error: {err}")
                QtWidgets.QMessageBox.critical(self, "Autofocus", str(err))
            else:
                log(f"Autofocus: best ΔZ={best:.4f} mm")
                QtWidgets.QMessageBox.information(self, "Autofocus", f"Best Z offset (relative): {best:.4f} mm")

        w.finished.connect(_done)

    def _run_raster(self):
        if not (self.stage and self.camera):
            log("Raster ignored: stage or camera not connected")
            return
        cfg = RasterConfig(
            rows=self.rows_spin.value(),
            cols=self.cols_spin.value(),
            pitch_x_mm=self.pitchx_spin.value(),
            pitch_y_mm=self.pitchy_spin.value(),
        )

        def do_raster():
            runner = RasterRunner(self.stage, self.camera, self.image_writer, cfg)
            runner.run()
            return True

        log("Raster: starting")
        t, w = run_async(do_raster)
        self._last_thread, self._last_worker = t, w
        w.finished.connect(lambda res, err: log("Raster: done" if not err else f"Raster error: {err}"))

    def _run_example_script(self):
        if not (self.stage and self.camera):
            log("Script ignored: stage or camera not connected")
            return
        # Import inside to avoid import costs if unused
        from ..scripts.zstack_example import run as zrun

        def do_script():
            zrun(self.stage, self.camera, self.image_writer)
            return True

        log("Script: Z-stack example")
        t, w = run_async(do_script)
        self._last_thread, self._last_worker = t, w
        w.finished.connect(lambda res, err: log("Script: done" if not err else f"Script error: {err}"))

    # --------------------------- PROFILES ---------------------------

    def _reload_profiles(self):
        self.profiles = Profiles.load_or_create()
        self.profile_combo.clear()
        self.profile_combo.addItems(self.profiles.list_profile_names())

    # --------------------------- CLOSE ---------------------------

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        try:
            if self.stage_worker:
                self.stage_worker.stop()
            if self.stage_thread:
                self.stage_thread.quit()
                self.stage_thread.wait(2000)
            if self.camera:
                try:
                    self.camera.stop_stream()
                except Exception:
                    pass
        finally:
            return super().closeEvent(e)
