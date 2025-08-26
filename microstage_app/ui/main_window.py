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
import os
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
        self._last_pos = {"x": None, "y": None, "z": None}

        # persistent serial worker
        self.stage_thread = None
        self.stage_worker = None

        # async connect helper refs
        self._conn_thread = None
        self._conn_worker = None

        # background op refs (prevent GC while running)
        self._last_thread = None
        self._last_worker = None

        # autofocus state
        self._autofocusing = False
        self._af_thread = None
        self._af_worker = None

        # image writer (per-run folder)
        self.image_writer = ImageWriter()

        # profiles
        self.profiles = Profiles.load_or_create()

        # capture settings
        dir_profile = self.profiles.get('capture.dir')
        self.capture_dir = dir_profile if dir_profile else self.image_writer.run_dir
        self.capture_name = self.profiles.get('capture.name', "capture")
        self.auto_number = self.profiles.get('capture.auto_number', False)
        self.capture_format = self.profiles.get('capture.format', 'bmf')

        # timers
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(33)          # ~30 FPS poll
        self.preview_timer.timeout.connect(self._on_preview)
        self.fps_timer = QtCore.QTimer(self)
        self.fps_timer.setInterval(500)             # update FPS label
        self.fps_timer.timeout.connect(self._update_fps)

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
        self.stage_status.setTextFormat(QtCore.Qt.PlainText)
        self.stage_pos = QtWidgets.QLabel("Pos: —")
        self.btn_stage_connect = QtWidgets.QPushButton("Connect Stage")
        self.btn_stage_disconnect = QtWidgets.QPushButton("Disconnect Stage")
        self.cam_status = QtWidgets.QLabel("Camera: —")
        self.btn_cam_connect = QtWidgets.QPushButton("Connect Camera")
        self.btn_cam_disconnect = QtWidgets.QPushButton("Disconnect Camera")
        self.profile_combo = QtWidgets.QComboBox()
        self.btn_reload_profiles = QtWidgets.QPushButton("Reload Profiles")
        left.addWidget(self.stage_status)
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
        # Homing controls
        home_box = QtWidgets.QGroupBox("Homing")
        hl = QtWidgets.QVBoxLayout(home_box)
        self.btn_home_all = QtWidgets.QPushButton("Home All")
        self.btn_home_x = QtWidgets.QPushButton("Home X")
        self.btn_home_y = QtWidgets.QPushButton("Home Y")
        self.btn_home_z = QtWidgets.QPushButton("Home Z")
        hl.addWidget(self.btn_home_all)
        hrow = QtWidgets.QHBoxLayout()
        hrow.addWidget(self.btn_home_x)
        hrow.addWidget(self.btn_home_y)
        hl.addLayout(hrow)
        hl.addWidget(self.btn_home_z)
        left.addWidget(home_box)

        # Jog controls
        jog_box = QtWidgets.QGroupBox("Jog")
        j = QtWidgets.QGridLayout(jog_box)
        limits = _load_feed_limits()
        self.stepx_spin = QtWidgets.QDoubleSpinBox(); self.stepx_spin.setDecimals(3); self.stepx_spin.setRange(0.001, 1000.0); self.stepx_spin.setValue(0.100)
        self.stepy_spin = QtWidgets.QDoubleSpinBox(); self.stepy_spin.setDecimals(3); self.stepy_spin.setRange(0.001, 1000.0); self.stepy_spin.setValue(0.100)
        self.stepz_spin = QtWidgets.QDoubleSpinBox(); self.stepz_spin.setDecimals(3); self.stepz_spin.setRange(0.001, 1000.0); self.stepz_spin.setValue(0.100)
        self.feedx_spin = QtWidgets.QDoubleSpinBox(); self.feedx_spin.setRange(0.01, limits[0] if limits else 1000.0); self.feedx_spin.setValue(20.0)
        self.feedy_spin = QtWidgets.QDoubleSpinBox(); self.feedy_spin.setRange(0.01, limits[1] if limits else 1000.0); self.feedy_spin.setValue(20.0)
        self.feedz_spin = QtWidgets.QDoubleSpinBox(); self.feedz_spin.setRange(0.01, limits[2] if limits else 1000.0); self.feedz_spin.setValue(20.0)
        self.feed_limit_x = QtWidgets.QLabel(f"\u2264 {limits[0]:.0f} mm/min" if limits else "")
        self.feed_limit_y = QtWidgets.QLabel(f"\u2264 {limits[1]:.0f} mm/min" if limits else "")
        self.feed_limit_z = QtWidgets.QLabel(f"\u2264 {limits[2]:.0f} mm/min" if limits else "")
        self.btn_xm = QtWidgets.QPushButton("X-")
        self.btn_xp = QtWidgets.QPushButton("X+")
        self.btn_ym = QtWidgets.QPushButton("Y-")
        self.btn_yp = QtWidgets.QPushButton("Y+")
        self.btn_zm = QtWidgets.QPushButton("Z-")
        self.btn_zp = QtWidgets.QPushButton("Z+")
        # per-axis rows
        j.addWidget(QtWidgets.QLabel("X"), 0, 0)
        j.addWidget(self.stepx_spin, 0, 1)
        j.addWidget(self.feedx_spin, 0, 2)
        j.addWidget(self.feed_limit_x, 0, 3)
        j.addWidget(self.btn_xm, 0, 4)
        j.addWidget(self.btn_xp, 0, 5)
        j.addWidget(QtWidgets.QLabel("Y"), 1, 0)
        j.addWidget(self.stepy_spin, 1, 1)
        j.addWidget(self.feedy_spin, 1, 2)
        j.addWidget(self.feed_limit_y, 1, 3)
        j.addWidget(self.btn_ym, 1, 4)
        j.addWidget(self.btn_yp, 1, 5)
        j.addWidget(QtWidgets.QLabel("Z"), 2, 0)
        j.addWidget(self.stepz_spin, 2, 1)
        j.addWidget(self.feedz_spin, 2, 2)
        j.addWidget(self.feed_limit_z, 2, 3)
        j.addWidget(self.btn_zm, 2, 4)
        j.addWidget(self.btn_zp, 2, 5)
        # absolute move controls
        self.absx_spin = QtWidgets.QDoubleSpinBox(); self.absx_spin.setDecimals(3)
        self.absy_spin = QtWidgets.QDoubleSpinBox(); self.absy_spin.setDecimals(3)
        self.absz_spin = QtWidgets.QDoubleSpinBox(); self.absz_spin.setDecimals(3)
        if self.stage_bounds:
            self.absx_spin.setRange(self.stage_bounds["xmin"], self.stage_bounds["xmax"])
            self.absy_spin.setRange(self.stage_bounds["ymin"], self.stage_bounds["ymax"])
            self.absz_spin.setRange(self.stage_bounds["zmin"], self.stage_bounds["zmax"])
        else:
            for sb in (self.absx_spin, self.absy_spin, self.absz_spin):
                sb.setRange(-1000.0, 1000.0)
        self.btn_move_to_coords = QtWidgets.QPushButton("Move to coordinates")
        j.addWidget(QtWidgets.QLabel("Abs"), 3, 0)
        j.addWidget(self.absx_spin, 3, 1)
        j.addWidget(self.absy_spin, 3, 2)
        j.addWidget(self.absz_spin, 3, 3)
        j.addWidget(self.btn_move_to_coords, 3, 4, 1, 2)
        left.addWidget(jog_box)

        # Autofocus controls moved from right-hand tab to left column
        af_box = QtWidgets.QGroupBox("Autofocus")
        a = QtWidgets.QGridLayout(af_box)
        self.metric_combo = QtWidgets.QComboBox(); self.metric_combo.addItems([m.value for m in FocusMetric])
        self.af_range = QtWidgets.QDoubleSpinBox(); self.af_range.setRange(0.01, 5.0); self.af_range.setValue(0.5)
        self.af_coarse = QtWidgets.QDoubleSpinBox(); self.af_coarse.setDecimals(3); self.af_coarse.setRange(0.001, 1.0); self.af_coarse.setValue(0.01)
        self.af_fine = QtWidgets.QDoubleSpinBox(); self.af_fine.setDecimals(3); self.af_fine.setRange(0.0005, 0.2); self.af_fine.setValue(0.002)
        self.btn_autofocus = QtWidgets.QPushButton("Run Autofocus")
        a.addWidget(QtWidgets.QLabel("Metric:"), 0, 0); a.addWidget(self.metric_combo, 0, 1)
        a.addWidget(QtWidgets.QLabel("Range (mm):"), 1, 0); a.addWidget(self.af_range, 1, 1)
        a.addWidget(QtWidgets.QLabel("Coarse step (mm):"), 2, 0); a.addWidget(self.af_coarse, 2, 1)
        a.addWidget(QtWidgets.QLabel("Fine step (mm):"), 3, 0); a.addWidget(self.af_fine, 3, 1)
        a.addWidget(self.btn_autofocus, 4, 0, 1, 2)
        left.addWidget(af_box)

        left.addStretch(1)
        left.addWidget(self.stage_pos)

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

        ctr3 = QtWidgets.QHBoxLayout()
        ctr3.addWidget(QtWidgets.QLabel("Dir:"))
        self.capture_dir_edit = QtWidgets.QLineEdit(self.capture_dir)
        self.capture_dir_edit.setPlaceholderText("Folder for captures")
        self.capture_dir_edit.setToolTip(
            "Destination folder for captured images. Created if missing and "
            "remembered between sessions."
        )
        ctr3.addWidget(self.capture_dir_edit, 1)
        self.btn_browse_dir = QtWidgets.QPushButton("Browse...")
        ctr3.addWidget(self.btn_browse_dir)
        center.addLayout(ctr3)

        ctr4 = QtWidgets.QHBoxLayout()
        ctr4.addWidget(QtWidgets.QLabel("Base name:"))
        self.capture_name_edit = QtWidgets.QLineEdit(self.capture_name)
        self.capture_name_edit.setPlaceholderText("Base filename")
        self.capture_name_edit.setToolTip(
            "Base filename without extension. Must not be empty or contain "
            "\\ / : * ? \" < > |. Saved between sessions."
        )
        ctr4.addWidget(self.capture_name_edit)
        self.autonumber_chk = QtWidgets.QCheckBox("Auto-number (_n)")
        self.autonumber_chk.setChecked(self.auto_number)
        self.autonumber_chk.setToolTip(
            "If enabled, append an incrementing _n suffix when a file with "
            "the same name exists to avoid overwriting. Setting persists."
        )
        ctr4.addWidget(self.autonumber_chk)
        ctr4.addWidget(QtWidgets.QLabel("Format:"))
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["BMF", "TIF", "PNG", "JPG"])
        self.format_combo.setCurrentText(self.capture_format.upper())
        self.format_combo.setToolTip("Image file format for captures")
        ctr4.addWidget(self.format_combo)
        ctr4.addStretch(1)
        center.addLayout(ctr4)

        # Right: tabs
        rightw = QtWidgets.QTabWidget()

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
        self.brightness_slider.setValue(self.brightness_spin.value())
        c.addWidget(QtWidgets.QLabel("Brightness:"), row, 0); c.addWidget(self.brightness_slider, row, 1); c.addWidget(self.brightness_spin, row, 2); row += 1

        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.contrast_slider.setRange(-255, 255)
        self.contrast_spin = QtWidgets.QSpinBox(); self.contrast_spin.setRange(-255, 255); self.contrast_spin.setValue(0)
        self.contrast_slider.setValue(self.contrast_spin.value())
        c.addWidget(QtWidgets.QLabel("Contrast:"), row, 0); c.addWidget(self.contrast_slider, row, 1); c.addWidget(self.contrast_spin, row, 2); row += 1

        self.saturation_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.saturation_slider.setRange(0, 255)
        self.saturation_spin = QtWidgets.QSpinBox(); self.saturation_spin.setRange(0, 255); self.saturation_spin.setValue(128)
        self.saturation_slider.setValue(self.saturation_spin.value())
        c.addWidget(QtWidgets.QLabel("Saturation:"), row, 0); c.addWidget(self.saturation_slider, row, 1); c.addWidget(self.saturation_spin, row, 2); row += 1

        self.hue_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.hue_slider.setRange(-180, 180)
        self.hue_spin = QtWidgets.QSpinBox(); self.hue_spin.setRange(-180, 180); self.hue_spin.setValue(0)
        self.hue_slider.setValue(self.hue_spin.value())
        c.addWidget(QtWidgets.QLabel("Hue:"), row, 0); c.addWidget(self.hue_slider, row, 1); c.addWidget(self.hue_spin, row, 2); row += 1

        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.gamma_slider.setRange(20, 180)
        self.gamma_spin = QtWidgets.QSpinBox(); self.gamma_spin.setRange(20, 180); self.gamma_spin.setValue(100)
        self.gamma_slider.setValue(self.gamma_spin.value())
        c.addWidget(QtWidgets.QLabel("Gamma:"), row, 0); c.addWidget(self.gamma_slider, row, 1); c.addWidget(self.gamma_spin, row, 2); row += 1

        self.raw_chk = QtWidgets.QCheckBox("RAW8 fast mono (triples bandwidth efficiency)")
        c.addWidget(self.raw_chk, row, 0, 1, 3); row += 1

        self.bin_combo = QtWidgets.QComboBox()
        self.res_combo = QtWidgets.QComboBox()
        self.btn_roi_full = QtWidgets.QPushButton("ROI: Full")
        self.btn_roi_2048 = QtWidgets.QPushButton("ROI: 2048²")
        self.btn_roi_1024 = QtWidgets.QPushButton("ROI: 1024²")
        self.btn_roi_512  = QtWidgets.QPushButton("ROI: 512²")
        c.addWidget(QtWidgets.QLabel("Binning:"), row, 0); c.addWidget(self.bin_combo, row, 1, 1, 2); row += 1
        c.addWidget(QtWidgets.QLabel("Resolution:"), row, 0); c.addWidget(self.res_combo, row, 1, 1, 2); row += 1
        c.addWidget(self.btn_roi_full, row, 0); c.addWidget(self.btn_roi_2048, row, 1); c.addWidget(self.btn_roi_1024, row, 2); row += 1
        c.addWidget(self.btn_roi_512, row, 0); row += 1

        self.speed_spin = QtWidgets.QSpinBox(); self.speed_spin.setRange(0, 5); self.speed_spin.setValue(0)
        c.addWidget(QtWidgets.QLabel("USB Speed/Bandwidth lvl:"), row, 0); c.addWidget(self.speed_spin, row, 1); row += 1

        self.decim_spin = QtWidgets.QSpinBox(); self.decim_spin.setRange(1, 8); self.decim_spin.setValue(1)
        c.addWidget(QtWidgets.QLabel("Display every Nth frame:"), row, 0); c.addWidget(self.decim_spin, row, 1); row += 1

        c.setRowStretch(row, 1)
        rightw.addTab(camtab, "Camera")

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
        self._setup_jog_button(self.btn_xm, self.stepx_spin, self.feedx_spin, sx=-1)
        self._setup_jog_button(self.btn_xp, self.stepx_spin, self.feedx_spin, sx=1)
        self._setup_jog_button(self.btn_ym, self.stepy_spin, self.feedy_spin, sy=-1)
        self._setup_jog_button(self.btn_yp, self.stepy_spin, self.feedy_spin, sy=1)
        self._setup_jog_button(self.btn_zm, self.stepz_spin, self.feedz_spin, sz=-1)
        self._setup_jog_button(self.btn_zp, self.stepz_spin, self.feedz_spin, sz=1)
        self.btn_move_to_coords.clicked.connect(self._move_to_coords)
        self.btn_autofocus.clicked.connect(self._run_autofocus)
        self.btn_run_raster.clicked.connect(self._run_raster)
        self.btn_reload_profiles.clicked.connect(self._reload_profiles)
        self.capture_dir_edit.textChanged.connect(self._on_capture_dir_changed)
        self.capture_name_edit.textChanged.connect(self._on_capture_name_changed)
        self.autonumber_chk.toggled.connect(self._on_autonumber_toggled)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.btn_browse_dir.clicked.connect(self._browse_capture_dir)

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
        self.bin_combo.currentIndexChanged.connect(self._apply_binning)
        self.res_combo.currentIndexChanged.connect(self._apply_resolution)
        self.btn_roi_full.clicked.connect(lambda: self._apply_roi('full'))
        self.btn_roi_2048.clicked.connect(lambda: self._apply_roi(2048))
        self.btn_roi_1024.clicked.connect(lambda: self._apply_roi(1024))
        self.btn_roi_512.clicked.connect(lambda: self._apply_roi(512))
        self.speed_spin.valueChanged.connect(self._apply_speed)
        self.decim_spin.valueChanged.connect(self._apply_decimation)

        # scripts
        self.btn_run_example_script.clicked.connect(self._run_example_script)

    def _on_capture_dir_changed(self, text: str):
        self.capture_dir = text
        self.profiles.set('capture.dir', text)
        self.profiles.save()

    def _on_capture_name_changed(self, text: str):
        self.capture_name = text
        self.profiles.set('capture.name', text)
        self.profiles.save()

    def _on_autonumber_toggled(self, checked: bool):
        self.auto_number = checked
        self.profiles.set('capture.auto_number', checked)
        self.profiles.save()

    def _on_format_changed(self, text: str):
        self.capture_format = text.lower()
        self.profiles.set('capture.format', self.capture_format)
        self.profiles.save()

    def _browse_capture_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Capture Directory", self.capture_dir
        )
        if d:
            self.capture_dir_edit.setText(d)

    def _setup_jog_button(self, btn, step_spin, feed_spin, sx=0, sy=0, sz=0):
        def _pressed():
            step = step_spin.value()
            feed = feed_spin.value()
            self._jog(step * sx, step * sy, step * sz, feed, wait_ok=True)
            self._jog_dir = (sx, sy, sz, step_spin, feed_spin)
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
        sx, sy, sz, step_spin, feed_spin = self._jog_dir
        step = step_spin.value()
        feed = feed_spin.value()
        self._jog(step * sx, step * sy, step * sz, feed, wait_ok=True, callback=self._repeat_jog)

    def _move_to_coords(self):
        if not self.stage_worker:
            log("Move ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return
        x = self.absx_spin.value()
        y = self.absy_spin.value()
        z = self.absz_spin.value()
        feed = max(self.feedx_spin.value(), self.feedy_spin.value(), self.feedz_spin.value())
        log(f"Move to: x={x} y={y} z={z} F={feed}")
        self.stage_worker.enqueue(self.stage.move_absolute, x, y, z, feed, True)
        self.stage_worker.enqueue(self.stage.wait_for_moves)
        self.stage_worker.enqueue(self.stage.get_position, callback=self._on_stage_position)

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

    def _dispatch_stage_result(self, cb, res):
        if cb:
            QtCore.QTimer.singleShot(0, lambda r=res: cb(r))

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
            # populate after stream start so all options are available
            self._populate_binning()
            self._populate_resolutions()
            QtCore.QTimer.singleShot(0, self._populate_resolutions)
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
        self.res_combo.clear()
        self.bin_combo.clear()
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
            name = info.get("machine_type") or info.get("name") or "connected"
            uuid = info.get("uuid")
            text = f"Stage: {name}"
            if uuid:
                text += f"\n{uuid}"
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
        self._update_stage_buttons()

    def _update_stage_buttons(self):
        connected = self.stage is not None
        self.btn_stage_connect.setEnabled(not connected)
        self.btn_stage_disconnect.setEnabled(connected)

    def _update_cam_buttons(self):
        connected = self.camera is not None
        self.btn_cam_connect.setEnabled(not connected)
        self.btn_cam_disconnect.setEnabled(connected)

    def _on_stage_position(self, pos):
        if not pos:
            return
        # update cached coordinates; ``pos`` may omit axes via ``None``
        try:
            x, y, z = pos
        except Exception:
            x = y = z = None
        if x is not None:
            self._last_pos["x"] = x
        if y is not None:
            self._last_pos["y"] = y
        if z is not None:
            self._last_pos["z"] = z
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
        def _fmt(v):
            return f"{v:.3f}" if v is not None else "—"
        coords_line = (
            f"Pos: X{_fmt(self._last_pos['x'])} "
            f"Y{_fmt(self._last_pos['y'])} "
            f"Z{_fmt(self._last_pos['z'])}"
        )
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
        if frame is not None:
            qimg = numpy_to_qimage(frame)
            self.live_label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
                self.live_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        if self.autoexp_chk.isChecked():
            try:
                self.exp_spin.blockSignals(True)
                self.gain_spin.blockSignals(True)
                ms = float(self.camera.get_exposure_ms())
                gain = int(self.camera.get_gain())
                self.exp_spin.setValue(ms)
                self.gain_spin.setValue(gain)
            except Exception:
                pass
            finally:
                self.exp_spin.blockSignals(False)
                self.gain_spin.blockSignals(False)

    def _update_fps(self):
        if self.camera:
            try:
                self.fps_label.setText(f"FPS: {self.camera.get_fps():.1f}")
            except Exception:
                self.fps_label.setText("FPS: —")

    # --------------------------- CAMERA APPLY ---------------------------

    def _populate_binning(self):
        if not self.camera or not hasattr(self.camera, "list_binning_factors"):
            self.bin_combo.clear()
            self.bin_combo.setEnabled(False)
            return
        self.bin_combo.blockSignals(True)
        self.bin_combo.clear()
        try:
            factors = self.camera.list_binning_factors()
            for f in factors:
                self.bin_combo.addItem(f"{f}×", f)
            cur = 1
            if hasattr(self.camera, "get_binning"):
                cur = int(self.camera.get_binning())
            pos = self.bin_combo.findData(cur)
            if pos >= 0:
                self.bin_combo.setCurrentIndex(pos)
        except Exception:
            pass
        self.bin_combo.setEnabled(self.bin_combo.count() > 1)
        self.bin_combo.blockSignals(False)

    def _populate_resolutions(self):
        if not self.camera:
            return
        self.res_combo.blockSignals(True)
        res = list(self.camera.list_resolutions())
        self.res_combo.clear()
        for idx, w, h in res:
            self.res_combo.addItem(f"{w}×{h}", idx)
        current = 0
        try:
            if hasattr(self.camera, "get_resolution_index"):
                current = int(self.camera.get_resolution_index())
        except Exception:
            current = 0
        try:
            self.camera.resolutions = res
        except Exception:
            pass
        pos = self.res_combo.findData(current)
        if pos >= 0:
            self.res_combo.setCurrentIndex(pos)
        self.res_combo.blockSignals(False)

    def _sync_cam_controls(self):
        if not self.camera:
            return
        try:
            if hasattr(self.camera, "get_brightness"):
                val = int(self.camera.get_brightness())
                self.brightness_spin.setValue(val)
                self.brightness_slider.setValue(val)
        except Exception:
            pass
        try:
            if hasattr(self.camera, "get_contrast"):
                val = int(self.camera.get_contrast())
                self.contrast_spin.setValue(val)
                self.contrast_slider.setValue(val)
        except Exception:
            pass
        try:
            if hasattr(self.camera, "get_saturation"):
                val = int(self.camera.get_saturation())
                self.saturation_spin.setValue(val)
                self.saturation_slider.setValue(val)
        except Exception:
            pass
        try:
            if hasattr(self.camera, "get_hue"):
                val = int(self.camera.get_hue())
                self.hue_spin.setValue(val)
                self.hue_slider.setValue(val)
        except Exception:
            pass
        try:
            if hasattr(self.camera, "get_gamma"):
                val = int(self.camera.get_gamma())
                self.gamma_spin.setValue(val)
                self.gamma_slider.setValue(val)
        except Exception:
            pass

    def _apply_exposure(self):
        if not self.camera: return
        auto = self.autoexp_chk.isChecked()
        ms = self.exp_spin.value()
        self.camera.set_exposure_ms(ms, auto)
        self.exp_spin.setEnabled(not auto)
        self.gain_spin.setEnabled(not auto)
        if auto:
            try:
                self.exp_spin.blockSignals(True)
                self.gain_spin.blockSignals(True)
                ms = float(self.camera.get_exposure_ms())
                gain = int(self.camera.get_gain())
                self.exp_spin.setValue(ms)
                self.gain_spin.setValue(gain)
            except Exception:
                pass
            finally:
                self.exp_spin.blockSignals(False)
                self.gain_spin.blockSignals(False)

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

    def _apply_binning(self, i: int):
        if not self.camera: return
        factor = self.bin_combo.currentData()
        if factor is None:
            return
        try:
            self.camera.set_binning(int(factor))
        except Exception:
            pass
        self._populate_resolutions()

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
        self.stage_worker.enqueue(self.stage.wait_for_moves)
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
        self.stage_worker.enqueue(self.stage.wait_for_moves)
        self.stage_worker.enqueue(
            self.stage.get_position, callback=self._on_stage_position
        )

    def _jog(self, dx=0, dy=0, dz=0, feed=0, *, wait_ok=True, callback=None):
        if not self.stage_worker:
            log("Jog ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return
        f = max(1.0, float(feed))
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
        self.stage_worker.enqueue(self.stage.wait_for_moves)
        self.stage_worker.enqueue(
            self.stage.get_position, callback=self._on_stage_position
        )

    # --------------------------- CAPTURE / MODES ---------------------------

    def _capture(self):
        if not (self.stage and self.camera):
            log("Capture ignored: stage or camera not connected")
            return

        directory = self.capture_dir
        name = self.capture_name
        auto_num = self.auto_number

        # validate directory
        if not directory:
            log("Capture aborted: directory not specified")
            QtWidgets.QMessageBox.critical(
                self, "Capture", "Capture directory is not set."
            )
            return
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            log(f"Capture aborted: cannot create directory {directory}: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Capture",
                f"Unable to create directory:\n{directory}\n{e}",
            )
            return

        # validate filename
        if not name:
            log("Capture aborted: filename empty")
            QtWidgets.QMessageBox.critical(
                self, "Capture", "Filename cannot be empty."
            )
            return
        if re.search(r"[\\/:*?\"<>|]", name):
            log("Capture aborted: illegal characters in filename")
            QtWidgets.QMessageBox.critical(
                self,
                "Capture",
                "Filename contains illegal characters (\\ / : * ? \" < > |).",
            )
            return

        def do_capture():
            self.stage.wait_for_moves()
            time.sleep(0.03)
            img = self.camera.snap()
            if img is not None:
                self.image_writer.save_single(
                    img,
                    directory=directory,
                    filename=name,
                    auto_number=auto_num,
                    fmt=self.capture_format,
                )
            return True

        log("Capture: starting")
        t, w = run_async(do_capture)
        self._last_thread, self._last_worker = t, w
        w.finished.connect(lambda res, err: log("Capture: done" if not err else f"Capture error: {err}"))

    @QtCore.Slot(object, object)
    def _on_autofocus_done(self, best, err):
        self.btn_autofocus.setEnabled(True)
        self._autofocusing = False
        self._af_thread = None
        self._af_worker = None
        if err:
            log(f"Autofocus error: {err}")
            QtWidgets.QMessageBox.critical(self, "Autofocus", str(err))
        else:
            log(f"Autofocus: best ΔZ={best:.4f} mm")
            QtWidgets.QMessageBox.information(self, "Autofocus", f"Best Z offset (relative): {best:.4f} mm")

    def _run_autofocus(self):
        if self._autofocusing:
            log("Autofocus ignored: already running")
            return
        if not (self.stage and self.camera):
            log("Autofocus ignored: stage or camera not connected")
            return

        coarse = float(self.af_coarse.value())
        fine = float(self.af_fine.value())
        if coarse <= 0 or fine <= 0:
            QtWidgets.QMessageBox.warning(self, "Autofocus", "Coarse and fine steps must be > 0")
            return

        metric = FocusMetric(self.metric_combo.currentText())
        self._autofocusing = True
        self.btn_autofocus.setEnabled(False)

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
        self._af_thread, self._af_worker = t, w

        w.finished.connect(self._on_autofocus_done)

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
