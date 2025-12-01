from PySide6 import QtWidgets, QtCore, QtGui

from typing import Optional

from .system_monitor_tab import SystemMonitorTab
from .tooltips import apply_tooltip_context
from .spectroscopy_window import SpectroscopyWindow

import numpy as np
import cv2

from ..devices.stage_marlin import StageMarlin, find_marlin_port, list_marlin_ports
from ..devices.camera_toupcam import create_camera, list_cameras
from ..devices.shelly_dimmer import ShellyDimmer, ShellyState
from ..spectroscopy.devices import SpectrometerDescriptor, SpectrometerManager

from ..control.autofocus import FocusMetric, AutoFocus
from ..control.raster import RasterRunner, RasterConfig
from ..control.profiles import Profiles
from ..io.storage import ImageWriter
from ..analysis import Lens
from ..control.focus_planes import (
    FocusPlaneManager,
    SurfaceModel,
    SurfaceKind,
    Area,
)

from ..utils.img import numpy_to_qimage, draw_scale_bar, VERT_SCALE, TEXT_SCALE
from ..utils.log import LOG, log
from ..utils.serial_worker import SerialWorker
from ..utils.workers import run_async

from pathlib import Path
import os
import re
import time
import math
import datetime
import threading


# Preferred lens display order
PRESET_LENS_ORDER = ["5x", "10x", "20x", "50x"]


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


class MeasureView(QtWidgets.QGraphicsView):
    calibration_measured = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap)
        self._mode = None
        self._reticle_enabled = False
        self._scale_bar_enabled = False
        self._scale_um_per_px = 1.0

        # ruler state
        self._anchor = None
        self._anchor_item = None
        self._live_line = None
        self._live_ticks = []
        self._live_text = None
        self._lines = []
        self._um_per_px = 1.0

        # calibration state
        self._points = []
        self._item = None

    def set_reticle(self, enabled: bool):
        self._reticle_enabled = enabled
        self.viewport().update()

    def set_scale_bar(self, enabled: bool, um_per_px: float):
        """Enable/disable the scale bar and set the current scale."""
        self._scale_bar_enabled = enabled
        self._scale_um_per_px = um_per_px
        self.viewport().update()

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        super().drawForeground(painter, rect)
        pix = self._pixmap.pixmap()
        if pix.isNull():
            return
        br = self._pixmap.boundingRect()
        painter.save()
        if self._reticle_enabled:
            painter.save()
            cx = br.center().x()
            cy = br.center().y()
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Difference)
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 3 * VERT_SCALE))
            painter.drawLine(QtCore.QLineF(br.left(), cy, br.right(), cy))
            painter.drawLine(QtCore.QLineF(cx, br.top(), cx, br.bottom()))
            painter.restore()

        if self._scale_bar_enabled and self._scale_um_per_px > 0:
            # compute a "nice" length that fits within ~20% of the image width
            max_um = 0.2 * br.width() * self._scale_um_per_px
            exp = math.floor(math.log10(max_um)) if max_um > 0 else 0
            nice_um = 10 ** exp
            for m in (5, 2, 1):
                candidate = m * (10 ** exp)
                if candidate <= max_um:
                    nice_um = candidate
                    break
            length_px = nice_um / self._scale_um_per_px
            margin = 20
            x0 = br.right() - margin - length_px
            y0 = br.bottom() - margin
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 2 * VERT_SCALE))
            painter.drawLine(x0, y0, x0 + length_px, y0)
            label = (
                f"{nice_um/1000:.2f} mm" if nice_um >= 1000 else f"{nice_um:.0f} µm"
            )
            font = painter.font()
            ps = font.pointSizeF()
            if ps > 0:
                font.setPointSizeF(ps * TEXT_SCALE)
            else:
                font.setPixelSize(font.pixelSize() * TEXT_SCALE)
            painter.setFont(font)
            fm = painter.fontMetrics()
            painter.drawText(x0, y0 - (7 * TEXT_SCALE) - fm.descent(), label)
        painter.restore()

    def set_image(self, qimg: QtGui.QImage):
        self._pixmap.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.setSceneRect(self._pixmap.boundingRect())
        self.fitInView(self._pixmap, QtCore.Qt.KeepAspectRatio)
        self.viewport().update()

    def clear_image(self):
        self._pixmap.setPixmap(QtGui.QPixmap())
        self._clear_temp()
        self.viewport().update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.fitInView(self._pixmap, QtCore.Qt.KeepAspectRatio)
        self.viewport().update()

    def start_ruler(self, um_per_px: float):
        self._clear_temp()
        self._mode = "ruler"
        self._um_per_px = um_per_px

    def start_calibration(self):
        self.clear_overlays()
        self._mode = "calibration"

    def _add_square(self, coord):
        size = 6
        rect = QtCore.QRectF(coord[0] - size / 2, coord[1] - size / 2, size, size)
        return self.scene().addRect(
            rect, QtGui.QPen(QtCore.Qt.red), QtGui.QBrush(QtCore.Qt.red)
        )

    def _clear_temp(self):
        # ruler temp items
        if self._anchor_item:
            self.scene().removeItem(self._anchor_item)
            self._anchor_item = None
        if self._live_line:
            self.scene().removeItem(self._live_line)
            self._live_line = None
        for t in self._live_ticks:
            self.scene().removeItem(t)
        self._live_ticks = []
        if self._live_text:
            self.scene().removeItem(self._live_text)
            self._live_text = None
        self._anchor = None

        # calibration temp items
        self._points = []
        if self._item:
            self.scene().removeItem(self._item)
            self._item = None

    def clear_overlays(self):
        self._mode = None
        for ln in self._lines:
            self.scene().removeItem(ln["start"])
            self.scene().removeItem(ln["end"])
            self.scene().removeItem(ln["line"])
            for t in ln["ticks"]:
                self.scene().removeItem(t)
            self.scene().removeItem(ln["text"])
        self._lines = []
        self._clear_temp()

    def _update_live_line(self, end):
        if not self._live_line:
            return
        line = QtCore.QLineF(
            QtCore.QPointF(*self._anchor), QtCore.QPointF(end[0], end[1])
        )
        self._live_line.setLine(line)

        # remove old ticks
        for t in self._live_ticks:
            self.scene().removeItem(t)
        self._live_ticks = []

        length = line.length()

        pixels = length
        microns = pixels * self._um_per_px
        self._live_text.setPlainText(f"{pixels:.1f} px / {microns:.1f} µm")

        if length > 0:
            unit_x = line.dx() / length
            unit_y = line.dy() / length
            norm_x = -unit_y
            norm_y = unit_x
            spacing = 50
            for d in range(spacing, int(length), spacing):
                px = self._anchor[0] + unit_x * d
                py = self._anchor[1] + unit_y * d
                tick = self.scene().addLine(
                    px + norm_x * 5,
                    py + norm_y * 5,
                    px - norm_x * 5,
                    py - norm_y * 5,
                    QtGui.QPen(QtCore.Qt.red, 1),
                )
                self._live_ticks.append(tick)

            midx = self._anchor[0] + unit_x * length / 2
            midy = self._anchor[1] + unit_y * length / 2
            self._live_text.setPos(midx + norm_x * 10, midy + norm_y * 10)
        else:
            self._live_text.setPos(*self._anchor)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._mode == "ruler" and self._anchor is not None:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            pt = self.mapToScene(pos)
            self._update_live_line((pt.x(), pt.y()))
        return super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._mode:
            return super().mousePressEvent(event)
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        pt = self.mapToScene(pos)
        coord = (pt.x(), pt.y())
        if self._mode == "ruler":
            if event.button() == QtCore.Qt.LeftButton:
                if self._anchor is None:
                    self._anchor = coord
                    self._anchor_item = self._add_square(coord)
                    self._live_line = self.scene().addLine(
                        QtCore.QLineF(pt, pt), QtGui.QPen(QtCore.Qt.red, 2)
                    )
                    self._live_text = self.scene().addText("")
                    self._live_text.setDefaultTextColor(QtCore.Qt.red)
                    font = self._live_text.font()
                    font.setPointSizeF(font.pointSizeF() * 4)
                    self._live_text.setFont(font)
                    self._update_live_line(coord)
            elif event.button() == QtCore.Qt.RightButton:
                if self._anchor is not None:
                    # Cancel the live segment in progress
                    self._clear_temp()
                elif self._lines:
                    # Remove the most recently completed line
                    last = self._lines.pop()
                    self.scene().removeItem(last["start"])
                    self.scene().removeItem(last["end"])
                    self.scene().removeItem(last["line"])
                    for t in last["ticks"]:
                        self.scene().removeItem(t)
                    self.scene().removeItem(last["text"])
                else:
                    # No lines to remove; ignore the click
                    pass
        elif self._mode == "calibration":
            self._points.append(coord)
            if len(self._points) == 2:
                line = QtCore.QLineF(
                    QtCore.QPointF(*self._points[0]), QtCore.QPointF(*self._points[1])
                )
                self._item = self.scene().addLine(line, QtGui.QPen(QtCore.Qt.blue, 2))
                p1 = np.asarray(self._points[0])
                p2 = np.asarray(self._points[1])
                pixels = float(np.linalg.norm(p1 - p2))
                self.calibration_measured.emit(pixels)
                self._mode = None
                self._points = []
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._mode == "ruler"
            and event.button() == QtCore.Qt.LeftButton
            and self._anchor is not None
        ):
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            pt = self.mapToScene(pos)
            coord = (pt.x(), pt.y())
            self._update_live_line(coord)
            term_item = self._add_square(coord)
            line_item = self._live_line
            text_item = self._live_text
            ticks = list(self._live_ticks)
            self._lines.append(
                {
                    "start": self._anchor_item,
                    "end": term_item,
                    "line": line_item,
                    "ticks": ticks,
                    "text": text_item,
                }
            )
            self._anchor = None
            self._anchor_item = None
            self._live_line = None
            self._live_ticks = []
            self._live_text = None
            self._mode = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._mode == "ruler":
            self._mode = None
            self._clear_temp()
        super().mouseDoubleClickEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroStage App v0.1")
        self.resize(1400, 900)
        self._default_style = QtWidgets.QApplication.instance().style().objectName()

        # device handles
        self.stage = None
        self.camera = None
        self.dimmer: ShellyDimmer | None = None
        self.spectrometer_manager = SpectrometerManager()
        self.spectroscopy_window: SpectroscopyWindow | None = None
        self.stage_bounds = _load_stage_bounds()
        self._stage_bounds_fallback = self.stage_bounds.copy() if self.stage_bounds else None
        self._last_pos = {"x": None, "y": None, "z": None}

        # persistent serial worker
        self.stage_thread = None
        self.stage_worker = None

        # async connect helper refs
        self._conn_thread = None
        self._conn_worker = None

        # shelly dimmer async helpers
        self._dimmer_conn_thread = None
        self._dimmer_conn_worker = None
        self._dimmer_op_thread = None
        self._dimmer_op_worker = None
        self._updating_dimmer_ui = False

        # background op refs (prevent GC while running)
        self._last_thread = None
        self._last_worker = None

        # raster state
        self._raster_runner = None
        self._raster_thread = None
        self._raster_worker = None

        # autofocus state
        self._autofocusing = False
        self._af_thread = None
        self._af_worker = None

        # focus stack state
        self._stack_thread = None
        self._stack_worker = None
        self._last_stack_dir = None

        # leveling state
        self._level_thread = None
        self._level_worker = None
        self._leveling = False
        self._level_continue_event = threading.Event()

        # image writer (per-run folder)
        self.image_writer = ImageWriter()

        # focus plane manager
        self.focus_mgr = FocusPlaneManager()
        # flag indicating whether leveling corrections are applied
        self.leveling_enabled = False
        # track the most recent absolute Z target requested by the user
        self._last_requested_z: Optional[float] = None

        # illumination rows shown in the Camera tab
        self.max_illumination_rows = 5
        self.illumination_rows: list[dict[str, object]] = []
        self._updating_illumination_ui = False

        # profiles
        self.profiles = Profiles.load_or_create()
        self.dark_mode = self.profiles.get('ui.dark_mode', False, expected_type=bool)
        lenses_cfg = self.profiles.get('measurement.lenses', {}, expected_type=dict)
        self.lenses: dict[str, Lens] = {}
        for name, cfg in lenses_cfg.items():
            if isinstance(cfg, dict):
                um = cfg.get('um_per_px', 1.0)
                cal = cfg.get('calibrations', {}) if isinstance(cfg.get('calibrations'), dict) else {}
                extras = {
                    k: v for k, v in cfg.items() if k not in ('um_per_px', 'calibrations')
                    and isinstance(v, (int, float))
                }
                if extras:
                    cal = dict(cal)
                    cal.update({k: float(v) for k, v in extras.items()})
                lens = Lens(name, um, cal)
            else:
                # legacy flat value
                lens = Lens(name, float(cfg))
            self.lenses[name] = lens
        cur_name = self.profiles.get('measurement.current_lens', '10x', expected_type=str)
        self.current_lens = self.lenses.get(cur_name)
        if not self.current_lens:
            # fall back to a known lens or create a default
            self.current_lens = self.lenses.get('10x') or Lens(cur_name, 1.0)
            self.lenses[self.current_lens.name] = self.current_lens

        # capture settings
        dir_profile = self.profiles.get('capture.dir', self.image_writer.run_dir,
                                        expected_type=str)
        self.capture_dir = dir_profile if dir_profile else self.image_writer.run_dir
        self.capture_name = self.profiles.get('capture.name', "capture", expected_type=str)
        self.auto_prefix = self.profiles.get('capture.auto_prefix', False, expected_type=bool)
        self.auto_number = self.profiles.get('capture.auto_number', False, expected_type=bool)
        fmt = self.profiles.get('capture.format', 'png', expected_type=str)
        if fmt.lower() not in {"bmp", "tif", "png", "jpg"}:
            log(f"WARNING: profile 'capture.format' has invalid value {fmt!r}; using default 'png'")
            fmt = 'png'
        self.capture_format = fmt

        # illumination
        self.dimmer_host_default = self.profiles.get(
            "illumination.dimmer.host", "", expected_type=str
        )
        self.dimmer_on_default = self.profiles.get(
            "illumination.dimmer.on", False, expected_type=bool
        )
        self.dimmer_brightness_default = int(
            self.profiles.get(
                "illumination.dimmer.brightness",
                0,
                expected_type=(int, float),
                min_value=0,
                max_value=100,
            )
        )

        # placeholders for legacy connect/disconnect buttons moved to the menu
        self.btn_stage_connect = None
        self.btn_stage_disconnect = None
        self.btn_cam_connect = None
        self.btn_cam_disconnect = None

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
        self._apply_dark_palette(self.act_dark_mode.isChecked())

        # load persisted values; extend _persistent_widgets() to add more fields
        for w, path in self._persistent_widgets():
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                val = self.profiles.get(
                    path,
                    w.value(),
                    expected_type=(int, float),
                    min_value=w.minimum(),
                    max_value=w.maximum(),
                )
                w.setValue(round(float(val), w.decimals()))
            elif isinstance(w, QtWidgets.QAbstractSpinBox):
                val = self.profiles.get(
                    path,
                    w.value(),
                    expected_type=(int, float),
                    min_value=w.minimum(),
                    max_value=w.maximum(),
                )
                w.setValue(val)
            elif isinstance(w, QtWidgets.QCheckBox):
                val = self.profiles.get(path, w.isChecked(), expected_type=bool)
                w.setChecked(val)
            elif isinstance(w, QtWidgets.QComboBox):
                data = w.currentData()
                default = data if data is not None else w.currentText()
                val = self.profiles.get(path, default, expected_type=(int, float, str))
                pos = w.findData(val)
                if pos >= 0:
                    w.setCurrentIndex(pos)
                elif isinstance(val, str) and val in [w.itemText(i) for i in range(w.count())]:
                    w.setCurrentText(val)
                else:
                    log(f"WARNING: profile '{path}' option {val!r} not valid; using default {default!r}")
            elif isinstance(w, QtWidgets.QLineEdit):
                val = self.profiles.get(path, w.text(), expected_type=str)
                w.setText(val)

        # ensure sliders match spins after loading persisted values
        self.brightness_slider.setValue(self.brightness_spin.value())
        self.contrast_slider.setValue(self.contrast_spin.value())
        self.saturation_slider.setValue(self.saturation_spin.value())
        self.hue_slider.setValue(self.hue_spin.value())
        self.gamma_slider.setValue(self.gamma_spin.value())
        self.dimmer_brightness_slider.setValue(self.dimmer_brightness_spin.value())

        self._load_illumination_rows_from_profiles()

        self.measure_view.set_scale_bar(
            self.chk_scale_bar.isChecked(), self.current_lens.um_per_px
        )

        self._connect_signals()
        self._init_persistent_fields()
        self._update_leveling_method()
        # ensure raster UI reflects current mode after loading profiles
        self._update_raster_mode()

        # install right-click tooltip handlers for all inputs/buttons
        apply_tooltip_context(self)

        # mirror logs to the in-app log pane
        LOG.message.connect(self._append_log)

        # show window first, then connect devices asynchronously
        QtCore.QTimer.singleShot(0, self._auto_connect_async)
        QtCore.QTimer.singleShot(0, self._refresh_spectrometers)

    # --------------------------- UI BUILD ---------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # removed empty toolbar previously used for measurement actions

        # Menu for device selection
        devices_menu = self.menuBar().addMenu("Devices")
        self.act_show_cameras = devices_menu.addAction("Cameras")
        self.act_show_stages = devices_menu.addAction("Stages")
        self.act_show_cameras.triggered.connect(self._show_camera_dialog)
        self.act_show_stages.triggered.connect(self._show_stage_dialog)

        modules_menu = self.menuBar().addMenu("Modules")
        self.act_show_spectroscopy = modules_menu.addAction("Vis Spectroscopy…")
        self.act_show_spectroscopy.triggered.connect(self._open_spectroscopy)

        view_menu = self.menuBar().addMenu("View")
        self.act_dark_mode = view_menu.addAction("Dark Theme")
        self.act_dark_mode.setCheckable(True)
        self.act_dark_mode.setChecked(self.dark_mode)
        self.act_dark_mode.triggered.connect(self._on_toggle_dark_mode)

        # Left column: device + profiles
        leftw = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(leftw)
        self.stage_status = QtWidgets.QLabel("Stage: —")
        self.stage_status.setTextFormat(QtCore.Qt.PlainText)
        self.stage_pos = QtWidgets.QLabel("Pos: —")
        self.stage_pos.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
        )
        self.cam_status = QtWidgets.QLabel("Camera: —")
        self.dimmer_status = QtWidgets.QLabel("Illumination: —")
        self.dimmer_status.setTextFormat(QtCore.Qt.PlainText)
        self.spectrometer_status = QtWidgets.QLabel("Spectrometer: —")
        self.profile_combo = QtWidgets.QComboBox()
        self.btn_reload_profiles = QtWidgets.QPushButton("Reload Profiles")
        self.profile_label = QtWidgets.QLabel("Profile:")
        left.addWidget(self.profile_label)
        left.addWidget(self.profile_combo)
        left.addWidget(self.btn_reload_profiles)
        self.profile_label.hide()
        self.profile_combo.hide()
        self.btn_reload_profiles.hide()
        left.addWidget(self.stage_status)
        left.addWidget(self.cam_status)
        left.addWidget(self.dimmer_status)
        left.addWidget(self.spectrometer_status)

        # Illumination controls
        illum_box = QtWidgets.QGroupBox("Illumination")
        il = QtWidgets.QGridLayout(illum_box)
        il.setColumnStretch(1, 1)
        il.setColumnStretch(2, 0)
        row = 0
        self.dimmer_host_edit = QtWidgets.QLineEdit(self.dimmer_host_default)
        self.btn_dimmer_connect = QtWidgets.QPushButton("Connect")
        self.btn_dimmer_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_dimmer_disconnect.setEnabled(False)
        il.addWidget(QtWidgets.QLabel("Host:"), row, 0)
        il.addWidget(self.dimmer_host_edit, row, 1)
        row += 1
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        btn_row.addWidget(self.btn_dimmer_connect)
        btn_row.addWidget(self.btn_dimmer_disconnect)
        il.addLayout(btn_row, row, 0, 1, 3)
        row += 1
        self.dimmer_toggle = QtWidgets.QCheckBox("On")
        self.dimmer_toggle.setChecked(self.dimmer_on_default)
        self.dimmer_toggle.setEnabled(False)
        il.addWidget(self.dimmer_toggle, row, 0)
        row += 1
        self.dimmer_brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dimmer_brightness_slider.setRange(0, 100)
        self.dimmer_brightness_spin = QtWidgets.QSpinBox()
        self.dimmer_brightness_spin.setRange(0, 100)
        self.dimmer_brightness_spin.setValue(self.dimmer_brightness_default)
        self.dimmer_brightness_slider.setValue(self.dimmer_brightness_spin.value())
        self.dimmer_brightness_slider.setEnabled(False)
        self.dimmer_brightness_spin.setEnabled(False)
        il.addWidget(QtWidgets.QLabel("Brightness:"), row, 0)
        il.addWidget(self.dimmer_brightness_slider, row, 1)
        il.addWidget(self.dimmer_brightness_spin, row, 2)
        row += 1
        il.setRowStretch(row, 1)
        left.addWidget(illum_box)
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
        self.stepx_spin = QtWidgets.QDoubleSpinBox(); self.stepx_spin.setDecimals(6); self.stepx_spin.setRange(0.000001, 1000.0); self.stepx_spin.setSingleStep(0.000001); self.stepx_spin.setValue(0.100)
        self.stepy_spin = QtWidgets.QDoubleSpinBox(); self.stepy_spin.setDecimals(6); self.stepy_spin.setRange(0.000001, 1000.0); self.stepy_spin.setSingleStep(0.000001); self.stepy_spin.setValue(0.100)
        self.stepz_spin = QtWidgets.QDoubleSpinBox(); self.stepz_spin.setDecimals(6); self.stepz_spin.setRange(0.000001, 1000.0); self.stepz_spin.setSingleStep(0.000001); self.stepz_spin.setValue(0.100)
        self.feedx_spin = QtWidgets.QDoubleSpinBox(); self.feedx_spin.setRange(0.01, limits[0] if limits else 1000.0); self.feedx_spin.setValue(50.0)
        self.feedy_spin = QtWidgets.QDoubleSpinBox(); self.feedy_spin.setRange(0.01, limits[1] if limits else 1000.0); self.feedy_spin.setValue(50.0)
        self.feedz_spin = QtWidgets.QDoubleSpinBox(); self.feedz_spin.setRange(0.01, limits[2] if limits else 1000.0); self.feedz_spin.setValue(50.0)
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
        self.absx_spin = QtWidgets.QDoubleSpinBox(); self.absx_spin.setDecimals(6)
        self.absy_spin = QtWidgets.QDoubleSpinBox(); self.absy_spin.setDecimals(6)
        self.absz_spin = QtWidgets.QDoubleSpinBox(); self.absz_spin.setDecimals(6)
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
        self.af_range = QtWidgets.QDoubleSpinBox(); self.af_range.setRange(0.01, 5.0); self.af_range.setDecimals(4); self.af_range.setValue(0.5)
        self.af_range.setToolTip(
            "Half of the autofocus sweep distance; the stage moves from -value to +value"
            " around the starting Z position."
        )
        self.af_coarse = QtWidgets.QDoubleSpinBox(); self.af_coarse.setDecimals(6); self.af_coarse.setRange(0.000001, 1.0); self.af_coarse.setSingleStep(0.000001); self.af_coarse.setValue(0.01)
        self.af_fine = QtWidgets.QDoubleSpinBox(); self.af_fine.setDecimals(6); self.af_fine.setRange(0.0005, 0.2); self.af_fine.setValue(0.002)
        self.btn_autofocus = QtWidgets.QPushButton("Run Autofocus")
        a.addWidget(QtWidgets.QLabel("Metric:"), 0, 0); a.addWidget(self.metric_combo, 0, 1)
        a.addWidget(QtWidgets.QLabel("Half-range (±mm):"), 1, 0); a.addWidget(self.af_range, 1, 1)
        a.addWidget(QtWidgets.QLabel("Coarse step (mm):"), 2, 0); a.addWidget(self.af_coarse, 2, 1)
        a.addWidget(QtWidgets.QLabel("Fine step (mm):"), 3, 0); a.addWidget(self.af_fine, 3, 1)
        a.addWidget(self.btn_autofocus, 4, 0, 1, 2)
        stack_box = QtWidgets.QGroupBox("Focus Stack")
        s = QtWidgets.QGridLayout(stack_box)
        self.stack_range = QtWidgets.QDoubleSpinBox(); self.stack_range.setRange(0.01, 5.0); self.stack_range.setDecimals(4); self.stack_range.setValue(0.5)
        self.stack_range.setToolTip(
            "Half of the focus-stack sweep; captures span from -value to +value"
            " relative to the start position."
        )
        self.stack_step = QtWidgets.QDoubleSpinBox(); self.stack_step.setDecimals(6); self.stack_step.setRange(0.000001, 1.0); self.stack_step.setSingleStep(0.000001); self.stack_step.setValue(0.01)
        self.chk_edf = QtWidgets.QCheckBox("EDF Fusion")
        self.chk_delete_stack = QtWidgets.QCheckBox("Delete stack images")
        self.chk_delete_stack.setEnabled(False)
        self.btn_focus_stack = QtWidgets.QPushButton("Run Focus Stack")
        s.addWidget(QtWidgets.QLabel("Half-range (±mm):"), 0, 0); s.addWidget(self.stack_range, 0, 1)
        s.addWidget(QtWidgets.QLabel("Step (mm):"), 1, 0); s.addWidget(self.stack_step, 1, 1)
        s.addWidget(self.chk_edf, 2, 0, 1, 2)
        s.addWidget(self.chk_delete_stack, 3, 0, 1, 2)
        s.addWidget(self.btn_focus_stack, 4, 0, 1, 2)

        af_box.setMaximumWidth(240)
        stack_box.setMaximumWidth(240)
        af_stack_row = QtWidgets.QHBoxLayout()
        af_stack_row.addWidget(af_box)
        af_stack_row.addWidget(stack_box)
        left.addLayout(af_stack_row)

        # Leveling controls
        lvl_box = QtWidgets.QGroupBox("Leveling")
        l = QtWidgets.QGridLayout(lvl_box)
        l.setContentsMargins(8, 6, 8, 6)
        l.setHorizontalSpacing(6)
        l.setVerticalSpacing(4)
        l.setColumnStretch(1, 1)
        l.setColumnStretch(3, 1)
        l.setColumnStretch(4, 0)
        self.level_method = QtWidgets.QComboBox(); self.level_method.addItems(["Three-point", "Grid"])
        self.level_poly = QtWidgets.QComboBox(); self.level_poly.addItems(["Linear", "Quadratic", "Cubic"])
        self.level_rows = QtWidgets.QSpinBox(); self.level_rows.setRange(2, 10); self.level_rows.setValue(3)
        self.level_cols = QtWidgets.QSpinBox(); self.level_cols.setRange(2, 10); self.level_cols.setValue(3)
        self.level_mode = QtWidgets.QComboBox(); self.level_mode.addItems(["Auto", "Manual"])
        # Coordinate fields for leveling points
        self.level_x1_spin = QtWidgets.QDoubleSpinBox(); self.level_x1_spin.setDecimals(6); self.level_x1_spin.setRange(-1000.0, 1000.0); self.level_x1_spin.setValue(0.0)
        self.level_y1_spin = QtWidgets.QDoubleSpinBox(); self.level_y1_spin.setDecimals(6); self.level_y1_spin.setRange(-1000.0, 1000.0); self.level_y1_spin.setValue(0.0)
        self.btn_level_p1 = QtWidgets.QPushButton("Use pos")
        self.level_x2_spin = QtWidgets.QDoubleSpinBox(); self.level_x2_spin.setDecimals(6); self.level_x2_spin.setRange(-1000.0, 1000.0); self.level_x2_spin.setValue(0.0)
        self.level_y2_spin = QtWidgets.QDoubleSpinBox(); self.level_y2_spin.setDecimals(6); self.level_y2_spin.setRange(-1000.0, 1000.0); self.level_y2_spin.setValue(0.0)
        self.btn_level_p2 = QtWidgets.QPushButton("Use pos")
        self.level_x3_spin = QtWidgets.QDoubleSpinBox(); self.level_x3_spin.setDecimals(6); self.level_x3_spin.setRange(-1000.0, 1000.0); self.level_x3_spin.setValue(0.0)
        self.level_y3_spin = QtWidgets.QDoubleSpinBox(); self.level_y3_spin.setDecimals(6); self.level_y3_spin.setRange(-1000.0, 1000.0); self.level_y3_spin.setValue(0.0)
        self.btn_level_p3 = QtWidgets.QPushButton("Use pos")
        self.btn_start_level = QtWidgets.QPushButton("Start Leveling")
        self.btn_apply_level = QtWidgets.QPushButton("Apply Leveling")
        self.btn_disable_level = QtWidgets.QPushButton("Disable Leveling")
        self.level_status = QtWidgets.QLabel("Disabled")
        row = 0
        l.addWidget(QtWidgets.QLabel("Method:"), row, 0); l.addWidget(self.level_method, row, 1); row += 1
        l.addWidget(QtWidgets.QLabel("Polynomial:"), row, 0); l.addWidget(self.level_poly, row, 1); row += 1
        l.addWidget(QtWidgets.QLabel("Rows:"), row, 0); l.addWidget(self.level_rows, row, 1); row += 1
        l.addWidget(QtWidgets.QLabel("Cols:"), row, 0); l.addWidget(self.level_cols, row, 1); row += 1
        l.addWidget(QtWidgets.QLabel("Mode:"), row, 0); l.addWidget(self.level_mode, row, 1); row += 1
        l.addWidget(QtWidgets.QLabel("P1 X:"), row, 0); l.addWidget(self.level_x1_spin, row, 1); l.addWidget(QtWidgets.QLabel("Y:"), row, 2); l.addWidget(self.level_y1_spin, row, 3); l.addWidget(self.btn_level_p1, row, 4); row += 1
        l.addWidget(QtWidgets.QLabel("P2 X:"), row, 0); l.addWidget(self.level_x2_spin, row, 1); l.addWidget(QtWidgets.QLabel("Y:"), row, 2); l.addWidget(self.level_y2_spin, row, 3); l.addWidget(self.btn_level_p2, row, 4); row += 1
        l.addWidget(QtWidgets.QLabel("P3 X:"), row, 0); l.addWidget(self.level_x3_spin, row, 1); l.addWidget(QtWidgets.QLabel("Y:"), row, 2); l.addWidget(self.level_y3_spin, row, 3); l.addWidget(self.btn_level_p3, row, 4); row += 1
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        btn_row.addWidget(self.btn_start_level)
        btn_row.addWidget(self.btn_apply_level)
        btn_row.addWidget(self.btn_disable_level)
        l.addLayout(btn_row, row, 0, 1, 5)
        row += 1
        self.level_equation = QtWidgets.QLabel("")
        self.level_equation.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.level_equation.setWordWrap(True)
        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)
        status_row.addWidget(self.level_status)
        status_row.addSpacing(12)
        status_row.addWidget(self.level_equation, 1)
        l.addLayout(status_row, row, 0, 1, 5)
        row += 1
        self.level_prompt = QtWidgets.QLabel("")
        self.level_prompt.setVisible(False)
        self.btn_level_continue = QtWidgets.QPushButton("Next")
        self.btn_level_continue.setVisible(False)
        l.addWidget(self.level_prompt, row, 0, 1, 5); row += 1
        l.addWidget(self.btn_level_continue, row, 0, 1, 5); row += 1
        left.addWidget(lvl_box)

        left.addStretch(1)
        left.addWidget(self.stage_pos)

        # Center: live preview + capture + FPS
        centerw = QtWidgets.QWidget()
        center = QtWidgets.QVBoxLayout(centerw)
        self.measure_view = MeasureView()
        self.measure_view.setMinimumSize(900, 650)
        self.fps_label = QtWidgets.QLabel("FPS: —")
        self.btn_capture = QtWidgets.QPushButton("Capture")
        center.addWidget(self.measure_view, 1)
        ctr2 = QtWidgets.QHBoxLayout()
        ctr2.addWidget(self.btn_capture)
        self.chk_reticle = QtWidgets.QCheckBox("Reticle")
        ctr2.addWidget(self.chk_reticle)
        self.chk_scale_bar = QtWidgets.QCheckBox("Scale bar")
        ctr2.addWidget(self.chk_scale_bar)
        self.lens_combo = QtWidgets.QComboBox()
        self._refresh_lens_combo()
        ctr2.addWidget(self.lens_combo)
        self.btn_add_lens = QtWidgets.QToolButton()
        self.btn_add_lens.setText("Add Lens...")
        ctr2.addWidget(self.btn_add_lens)
        self.measure_button = QtWidgets.QToolButton()
        self.measure_button.setText("Measure")
        _mnu = QtWidgets.QMenu(self.measure_button)
        self.act_calibrate = _mnu.addAction("Calibrate")
        self.act_ruler = _mnu.addAction("Ruler")
        self.measure_button.setMenu(_mnu)
        self.measure_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        ctr2.addWidget(self.measure_button)
        self.btn_clear_screen = QtWidgets.QPushButton("Clear screen")
        ctr2.addWidget(self.btn_clear_screen)
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
        self.autoprefix_chk = QtWidgets.QCheckBox("Auto-prefix (yyyymmddhhmmss_)")
        self.autoprefix_chk.setChecked(self.auto_prefix)
        self.autoprefix_chk.setToolTip(
            "When enabled, prepend a timestamp such as 20240101010101_ to the "
            "base filename for every capture. Auto-numbering still applies "
            "after the prefix when enabled."
        )
        ctr4.addWidget(self.autoprefix_chk)
        ctr4.addWidget(QtWidgets.QLabel("Format:"))
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["BMP", "TIF", "PNG", "JPG"])
        self.format_combo.setCurrentText(self.capture_format.upper())
        self.format_combo.setToolTip("Image file format for captures")
        ctr4.addWidget(self.format_combo)
        ctr4.addStretch(1)
        center.addLayout(ctr4)

        # Right: tabs
        rightw = QtWidgets.QTabWidget()
        rightw.setMaximumWidth(350)  # tweak as needed
        rightw.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred
        )

        # ---- Camera tab (performance controls)
        camtab = QtWidgets.QWidget()
        c = QtWidgets.QGridLayout(camtab)
        c.setColumnStretch(1, 1)
        c.setColumnStretch(2, 0)
        row = 0
        self.exp_spin = QtWidgets.QDoubleSpinBox(); self.exp_spin.setRange(0.01, 10000.0); self.exp_spin.setValue(10.0)
        self.exp_spin.setSuffix(" ms")
        self.autoexp_chk = QtWidgets.QCheckBox("Auto")
        c.addWidget(QtWidgets.QLabel("Exposure:"), row, 0); c.addWidget(self.exp_spin, row, 1); c.addWidget(self.autoexp_chk, row, 2); row += 1

        self.gain_spin = QtWidgets.QDoubleSpinBox(); self.gain_spin.setRange(1.0, 4.0); self.gain_spin.setSingleStep(0.01); self.gain_spin.setValue(1.0)
        self.gain_spin.setSuffix("x")
        self.gain_spin.setToolTip("Analog gain (1.0–4.0x). Internally scaled ×100 for the SDK.")
        c.addWidget(QtWidgets.QLabel("Gain (AGain):"), row, 0); c.addWidget(self.gain_spin, row, 1); row += 1

        illum_box = self._build_camera_illumination_group()
        c.addWidget(illum_box, row, 0, 1, 3)
        row += 1

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

        self.depth_combo = QtWidgets.QComboBox()
        c.addWidget(QtWidgets.QLabel("Color depth:"), row, 0); c.addWidget(self.depth_combo, row, 1, 1, 2); row += 1

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

        self.speed_spin = QtWidgets.QSpinBox(); self.speed_spin.setRange(0, 5); self.speed_spin.setValue(5)
        c.addWidget(QtWidgets.QLabel("USB speed level:"), row, 0); c.addWidget(self.speed_spin, row, 1); row += 1

        c.setRowStretch(row, 1)
        rightw.addTab(camtab, "Camera")

        # ---- Area tab
        area = QtWidgets.QWidget()
        a = QtWidgets.QVBoxLayout(area)

        # Raster controls
        rast_box = QtWidgets.QGroupBox("Raster")
        r = QtWidgets.QGridLayout(rast_box)
        r.setColumnStretch(1, 1)
        r.setColumnStretch(3, 1)
        r.setColumnStretch(4, 0)
        r.setColumnStretch(5, 0)
        self.rows_spin = QtWidgets.QSpinBox(); self.rows_spin.setRange(1, 1000); self.rows_spin.setValue(5)
        self.cols_spin = QtWidgets.QSpinBox(); self.cols_spin.setRange(1, 1000); self.cols_spin.setValue(5)

        # Raster point spin boxes
        self.rast_x1_spin = QtWidgets.QDoubleSpinBox(); self.rast_x1_spin.setDecimals(6); self.rast_x1_spin.setRange(-1000.0, 1000.0); self.rast_x1_spin.setValue(0.0)
        self.rast_y1_spin = QtWidgets.QDoubleSpinBox(); self.rast_y1_spin.setDecimals(6); self.rast_y1_spin.setRange(-1000.0, 1000.0); self.rast_y1_spin.setValue(0.0)
        self.rast_x2_spin = QtWidgets.QDoubleSpinBox(); self.rast_x2_spin.setDecimals(6); self.rast_x2_spin.setRange(-1000.0, 1000.0); self.rast_x2_spin.setValue(4.0)
        self.rast_y2_spin = QtWidgets.QDoubleSpinBox(); self.rast_y2_spin.setDecimals(6); self.rast_y2_spin.setRange(-1000.0, 1000.0); self.rast_y2_spin.setValue(4.0)
        self.rast_x3_spin = QtWidgets.QDoubleSpinBox(); self.rast_x3_spin.setDecimals(6); self.rast_x3_spin.setRange(-1000.0, 1000.0); self.rast_x3_spin.setValue(0.0)
        self.rast_y3_spin = QtWidgets.QDoubleSpinBox(); self.rast_y3_spin.setDecimals(6); self.rast_y3_spin.setRange(-1000.0, 1000.0); self.rast_y3_spin.setValue(0.0)
        self.rast_x4_spin = QtWidgets.QDoubleSpinBox(); self.rast_x4_spin.setDecimals(6); self.rast_x4_spin.setRange(-1000.0, 1000.0); self.rast_x4_spin.setValue(0.0)
        self.rast_y4_spin = QtWidgets.QDoubleSpinBox(); self.rast_y4_spin.setDecimals(6); self.rast_y4_spin.setRange(-1000.0, 1000.0); self.rast_y4_spin.setValue(0.0)

        # Raster point buttons
        self.btn_raster_p1 = QtWidgets.QPushButton("Raster Point 1")
        self.btn_raster_p2 = QtWidgets.QPushButton("Raster Point 2")
        self.btn_raster_p3 = QtWidgets.QPushButton("Raster Point 3")
        self.btn_raster_p4 = QtWidgets.QPushButton("Raster Point 4")

        # Raster mode selection
        self.raster_mode_combo = QtWidgets.QComboBox()
        self.raster_mode_combo.addItems(["2-point", "3-point", "4-point"])

        self.chk_raster_capture = QtWidgets.QCheckBox("Capture images")
        self.chk_raster_capture.setChecked(True)
        self.chk_raster_af = QtWidgets.QCheckBox("Autofocus before capture")
        self.chk_raster_stack = QtWidgets.QCheckBox("Focus stack after capture")
        self.btn_run_raster = QtWidgets.QPushButton("Run Raster")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setToolTip(
            "Stop active raster, leveling, or focus stack operations"
        )
        self.btn_stop.setEnabled(False)

        r.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        r.addWidget(self.raster_mode_combo, 0, 1)
        r.addWidget(QtWidgets.QLabel("Rows:"), 0, 2)
        r.addWidget(self.rows_spin, 0, 3)
        r.addWidget(QtWidgets.QLabel("Cols:"), 0, 4)
        r.addWidget(self.cols_spin, 0, 5)

        r.addWidget(QtWidgets.QLabel("P1 X:"), 1, 0)
        r.addWidget(self.rast_x1_spin, 1, 1)
        r.addWidget(QtWidgets.QLabel("Y:"), 1, 2)
        r.addWidget(self.rast_y1_spin, 1, 3)
        r.addWidget(self.btn_raster_p1, 1, 4, 1, 2)

        r.addWidget(QtWidgets.QLabel("P2 X:"), 2, 0)
        r.addWidget(self.rast_x2_spin, 2, 1)
        r.addWidget(QtWidgets.QLabel("Y:"), 2, 2)
        r.addWidget(self.rast_y2_spin, 2, 3)
        r.addWidget(self.btn_raster_p2, 2, 4, 1, 2)

        r.addWidget(QtWidgets.QLabel("P3 X:"), 3, 0)
        r.addWidget(self.rast_x3_spin, 3, 1)
        r.addWidget(QtWidgets.QLabel("Y:"), 3, 2)
        r.addWidget(self.rast_y3_spin, 3, 3)
        r.addWidget(self.btn_raster_p3, 3, 4, 1, 2)

        r.addWidget(QtWidgets.QLabel("P4 X:"), 4, 0)
        r.addWidget(self.rast_x4_spin, 4, 1)
        r.addWidget(QtWidgets.QLabel("Y:"), 4, 2)
        r.addWidget(self.rast_y4_spin, 4, 3)
        r.addWidget(self.btn_raster_p4, 4, 4, 1, 2)

        r.addWidget(self.chk_raster_capture, 5, 0, 1, 2)
        r.addWidget(self.chk_raster_af, 5, 2, 1, 2)
        r.addWidget(self.chk_raster_stack, 5, 4, 1, 2)
        r.addWidget(self.btn_run_raster, 6, 0, 1, 3)
        r.addWidget(self.btn_stop, 6, 3, 1, 3)
        r.setRowStretch(7, 1)
        rast_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        a.addWidget(rast_box)
        a.addStretch(1)
        rightw.addTab(area, "Area")

        # ---- Scripts tab (restored)
        scripts = QtWidgets.QWidget()
        s = QtWidgets.QVBoxLayout(scripts)
        self.btn_run_example_script = QtWidgets.QPushButton("Run Example Script (Z stack)")
        s.addWidget(self.btn_run_example_script)
        s.addStretch(1)
        rightw.addTab(scripts, "Scripts")

        # ---- Devices tab (spectrometers)
        devices_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(devices_tab)
        spec_box = QtWidgets.QGroupBox("Spectrometers")
        spec_layout = QtWidgets.QVBoxLayout(spec_box)
        self.spectrometer_list = QtWidgets.QListWidget()
        self.spectrometer_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        spec_layout.addWidget(self.spectrometer_list)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_spec_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_spec_connect = QtWidgets.QPushButton("Connect")
        self.btn_spec_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_spec_disconnect.setEnabled(False)
        btn_row.addWidget(self.btn_spec_refresh)
        btn_row.addWidget(self.btn_spec_connect)
        btn_row.addWidget(self.btn_spec_disconnect)
        spec_layout.addLayout(btn_row)
        dv.addWidget(spec_box)
        dv.addStretch(1)
        rightw.addTab(devices_tab, "Devices")

        # ---- System monitor tab
        self.system_tab = SystemMonitorTab()
        self.system_tab.add_device_status_widgets(
            self.stage_status, self.cam_status, self.dimmer_status, self.spectrometer_status
        )
        rightw.addTab(self.system_tab, "System")
        self.system_tab.start()

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
        self._update_raster_mode()

    def _apply_dark_palette(self, enabled: bool):
        app = QtWidgets.QApplication.instance()
        if enabled:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
            app.setPalette(palette)
            app.setStyle("Fusion")
        else:
            app.setStyle(self._default_style)
            app.setPalette(app.style().standardPalette())

    def _on_toggle_dark_mode(self, checked: bool):
        self._apply_dark_palette(checked)
        self.profiles.set('ui.dark_mode', checked)
        self.profiles.save()

    def _refresh_lens_combo(self):
        self.lens_combo.blockSignals(True)
        self.lens_combo.clear()

        # First add lenses in the preferred order if they exist
        preset = [
            self.lenses[name] for name in PRESET_LENS_ORDER if name in self.lenses
        ]
        # Then append any remaining lenses alphabetically
        remaining = sorted(
            [lens for name, lens in self.lenses.items() if name not in PRESET_LENS_ORDER],
            key=lambda l: l.name,
        )
        for lens in preset + remaining:
            self.lens_combo.addItem(
                f"{lens.name} ({lens.um_per_px:.3f} µm/px)", lens.name
            )
        idx = self.lens_combo.findData(self.current_lens.name)
        if idx >= 0:
            self.lens_combo.setCurrentIndex(idx)
        self.lens_combo.blockSignals(False)

    def _update_leveling_method(self):
        grid = self.level_method.currentText() == "Grid"
        self.level_rows.setEnabled(grid)
        self.level_cols.setEnabled(grid)

    def _update_raster_mode(self):
        mode = self.raster_mode_combo.currentText()
        p3 = mode in ("3-point", "4-point")
        p4 = mode == "4-point"
        for w in (self.rast_x3_spin, self.rast_y3_spin, self.btn_raster_p3):
            w.setEnabled(p3)
        for w in (self.rast_x4_spin, self.rast_y4_spin, self.btn_raster_p4):
            w.setEnabled(p4)

    def _build_camera_illumination_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Illumination")
        layout = QtWidgets.QGridLayout(box)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 0)
        layout.setHorizontalSpacing(6)

        layout.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("IP / Host"), 0, 1)
        layout.addWidget(QtWidgets.QLabel("Power"), 0, 2)
        layout.addWidget(QtWidgets.QLabel("Intensity"), 0, 3, 1, 2)

        self.illumination_rows = []
        for idx in range(self.max_illumination_rows):
            name_edit = QtWidgets.QLineEdit(f"Light {idx + 1}")
            name_edit.setPlaceholderText("Light label")
            name_edit.setToolTip("Custom label for this light source. Right-click to re-show.")

            host_edit = QtWidgets.QLineEdit()
            host_edit.setPlaceholderText("192.168.1.10")
            host_edit.setToolTip("Device IP or hostname for this light. Right-click to re-show.")

            toggle = QtWidgets.QCheckBox("On")
            toggle.setToolTip("Turn this light on or off. Right-click to re-show.")

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setToolTip("Adjust illumination intensity (0–100%). Right-click to re-show.")

            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix(" %")
            spin.setValue(slider.value())
            spin.setToolTip("Numeric intensity entry for this light. Right-click to re-show.")

            layout.addWidget(name_edit, idx + 1, 0)
            layout.addWidget(host_edit, idx + 1, 1)
            layout.addWidget(toggle, idx + 1, 2)
            layout.addWidget(slider, idx + 1, 3)
            layout.addWidget(spin, idx + 1, 4)

            row_data = {
                "widgets": [name_edit, host_edit, toggle, slider, spin],
                "name": name_edit,
                "host": host_edit,
                "toggle": toggle,
                "slider": slider,
                "spin": spin,
                "active": False,
            }
            self.illumination_rows.append(row_data)
            self._set_illumination_row_active(row_data, idx == 0)

        controls_row = self.max_illumination_rows + 1
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        self.btn_add_illumination = QtWidgets.QPushButton("Add Light")
        self.btn_add_illumination.setToolTip("Show another illumination device slot. Right-click to re-show.")
        self.btn_remove_illumination = QtWidgets.QPushButton("Remove Light")
        self.btn_remove_illumination.setToolTip("Hide the last visible illumination slot. Right-click to re-show.")
        btn_row.addWidget(self.btn_add_illumination)
        btn_row.addWidget(self.btn_remove_illumination)
        layout.addLayout(btn_row, controls_row, 0, 1, 5)
        layout.setRowStretch(controls_row + 1, 1)

        self._update_illumination_buttons()
        return box

    def _set_illumination_row_active(self, row: dict[str, object], active: bool, *, clear: bool = True) -> None:
        self._updating_illumination_ui = True
        try:
            widgets = row.get("widgets", [])
            for w in widgets:
                if isinstance(w, QtWidgets.QWidget):
                    w.setVisible(active)
                    w.setEnabled(active)

            row["active"] = active

            if not active and clear:
                toggle = row.get("toggle")
                slider = row.get("slider")
                spin = row.get("spin")
                if isinstance(toggle, QtWidgets.QCheckBox):
                    toggle.setChecked(False)
                if isinstance(slider, QtWidgets.QSlider):
                    slider.setValue(0)
                if isinstance(spin, QtWidgets.QSpinBox):
                    spin.setValue(0)
        finally:
            self._updating_illumination_ui = False

    def _active_illumination_rows(self) -> list[dict[str, object]]:
        return [row for row in self.illumination_rows if row.get("active")]

    def _update_illumination_buttons(self) -> None:
        visible_rows = self._active_illumination_rows()
        self.btn_add_illumination.setEnabled(len(visible_rows) < self.max_illumination_rows)
        self.btn_remove_illumination.setEnabled(len(visible_rows) > 1)

    def _show_next_illumination_row(self):
        for row in self.illumination_rows:
            if not row.get("active"):
                self._set_illumination_row_active(row, True)
                break
        self._update_illumination_buttons()
        self._persist_illumination_rows()

    def _hide_last_illumination_row(self):
        visible_rows = self._active_illumination_rows()
        if len(visible_rows) <= 1:
            return

        row = visible_rows[-1]
        name_edit = row.get("name")
        host_edit = row.get("host")
        if isinstance(name_edit, QtWidgets.QLineEdit):
            name_edit.clear()
        if isinstance(host_edit, QtWidgets.QLineEdit):
            host_edit.clear()

        self._set_illumination_row_active(row, False)
        self._update_illumination_buttons()
        self._persist_illumination_rows()

    def _load_illumination_rows_from_profiles(self):
        defaults = Profiles.illumination_light_defaults()
        lights_raw = self.profiles.get(
            "illumination.lights", defaults, expected_type=list
        )
        lights, changed = Profiles.sanitize_illumination_lights(lights_raw)

        if changed:
            self.profiles.set("illumination.lights", lights)
            self.profiles.save()

        self._updating_illumination_ui = True
        try:
            for idx, row in enumerate(self.illumination_rows):
                cfg = lights[idx] if idx < len(lights) else defaults[idx % len(defaults)]
                name_edit = row.get("name")
                host_edit = row.get("host")
                toggle = row.get("toggle")
                slider = row.get("slider")
                spin = row.get("spin")

                if isinstance(name_edit, QtWidgets.QLineEdit):
                    name_edit.setText(cfg.get("name", defaults[idx]["name"]))
                if isinstance(host_edit, QtWidgets.QLineEdit):
                    host_edit.setText(cfg.get("host", defaults[idx]["host"]))
                if isinstance(toggle, QtWidgets.QCheckBox):
                    toggle.setChecked(bool(cfg.get("enabled", False)))
                brightness = int(cfg.get("brightness", 0))
                if isinstance(spin, QtWidgets.QSpinBox):
                    spin.setValue(brightness)
                if isinstance(slider, QtWidgets.QSlider):
                    slider.setValue(brightness)

                should_activate = (
                    idx == 0
                    or bool(
                        cfg.get("host")
                        or cfg.get("enabled")
                        or cfg.get("brightness")
                        or cfg.get("name")
                    )
                )
                self._set_illumination_row_active(row, should_activate, clear=False)
            self._update_illumination_buttons()
        finally:
            self._updating_illumination_ui = False

    def _collect_illumination_profile_rows(self) -> list[dict[str, object]]:
        defaults = Profiles.illumination_light_defaults()
        lights: list[dict[str, object]] = []
        for idx, row in enumerate(self.illumination_rows):
            default_cfg = defaults[idx] if idx < len(defaults) else defaults[0]
            name_edit = row.get("name")
            host_edit = row.get("host")
            toggle = row.get("toggle")
            spin = row.get("spin")

            name = name_edit.text().strip() if isinstance(name_edit, QtWidgets.QLineEdit) else default_cfg["name"]
            host = host_edit.text().strip() if isinstance(host_edit, QtWidgets.QLineEdit) else default_cfg["host"]
            enabled = toggle.isChecked() if isinstance(toggle, QtWidgets.QCheckBox) else default_cfg["enabled"]
            brightness = (
                int(spin.value()) if isinstance(spin, QtWidgets.QSpinBox) else int(default_cfg["brightness"])
            )
            brightness = max(0, min(100, brightness))

            lights.append({
                "name": name or default_cfg["name"],
                "host": host,
                "enabled": bool(enabled),
                "brightness": brightness,
            })
        return lights

    def _persist_illumination_rows(self, *, save: bool = True):
        lights = self._collect_illumination_profile_rows()
        lights, _ = Profiles.sanitize_illumination_lights(lights)
        self.profiles.set("illumination.lights", lights)
        if save:
            self.profiles.save()

    def _on_illumination_row_changed(self, row: dict[str, object]):
        if self._updating_illumination_ui:
            return
        # activate the row if it currently holds user data
        has_data = False
        name_edit = row.get("name")
        host_edit = row.get("host")
        toggle = row.get("toggle")
        spin = row.get("spin")
        if isinstance(name_edit, QtWidgets.QLineEdit) and name_edit.text().strip():
            has_data = True
        if isinstance(host_edit, QtWidgets.QLineEdit) and host_edit.text().strip():
            has_data = True
        if isinstance(toggle, QtWidgets.QCheckBox) and toggle.isChecked():
            has_data = True
        if isinstance(spin, QtWidgets.QSpinBox) and spin.value() > 0:
            has_data = True
        if has_data and not row.get("active"):
            self._set_illumination_row_active(row, True, clear=False)
            self._update_illumination_buttons()
        self._persist_illumination_rows()

    def _connect_signals(self):
        self.btn_capture.clicked.connect(self._capture)
        self.chk_reticle.toggled.connect(self.measure_view.set_reticle)
        self.chk_scale_bar.toggled.connect(self._on_scale_bar_toggled)
        self.chk_edf.toggled.connect(self._on_edf_toggled)
        self.btn_add_lens.clicked.connect(self._add_lens)
        # ensure index is emitted as int for lens change
        self.lens_combo.currentIndexChanged[int].connect(self._on_lens_changed)
        self.btn_clear_screen.clicked.connect(self.measure_view.clear_overlays)
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
        self.btn_level_p1.clicked.connect(lambda: self._set_level_point(1))
        self.btn_level_p2.clicked.connect(lambda: self._set_level_point(2))
        self.btn_level_p3.clicked.connect(lambda: self._set_level_point(3))
        self.btn_start_level.clicked.connect(self._run_leveling)
        self.btn_apply_level.clicked.connect(self._apply_leveling)
        self.btn_disable_level.clicked.connect(self._disable_leveling)
        self.btn_level_continue.clicked.connect(self._on_level_continue)
        self.level_method.currentTextChanged.connect(self._update_leveling_method)
        self.raster_mode_combo.currentTextChanged.connect(self._update_raster_mode)
        self.btn_focus_stack.clicked.connect(self._run_focus_stack)
        self.btn_raster_p1.clicked.connect(lambda: self._set_raster_point(1))
        self.btn_raster_p2.clicked.connect(lambda: self._set_raster_point(2))
        self.btn_raster_p3.clicked.connect(lambda: self._set_raster_point(3))
        self.btn_raster_p4.clicked.connect(lambda: self._set_raster_point(4))
        self.btn_run_raster.clicked.connect(self._run_raster)
        self.btn_stop.clicked.connect(self._stop_all)
        self.btn_reload_profiles.clicked.connect(self._reload_profiles)
        self.capture_dir_edit.textChanged.connect(self._on_capture_dir_changed)
        self.capture_name_edit.textChanged.connect(self._on_capture_name_changed)
        self.autonumber_chk.toggled.connect(self._on_autonumber_toggled)
        self.autoprefix_chk.toggled.connect(self._on_autoprefix_toggled)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.btn_browse_dir.clicked.connect(self._browse_capture_dir)
        self.btn_dimmer_connect.clicked.connect(self._connect_dimmer_async)
        self.btn_dimmer_disconnect.clicked.connect(self._disconnect_dimmer)
        self.dimmer_toggle.toggled.connect(self._on_dimmer_toggle)
        self.dimmer_brightness_slider.valueChanged.connect(self.dimmer_brightness_spin.setValue)
        self.dimmer_brightness_spin.valueChanged.connect(self.dimmer_brightness_slider.setValue)
        self.dimmer_brightness_slider.sliderReleased.connect(self._on_dimmer_brightness_changed)
        self.dimmer_brightness_spin.editingFinished.connect(self._on_dimmer_brightness_changed)
        self.btn_add_illumination.clicked.connect(self._show_next_illumination_row)
        self.btn_remove_illumination.clicked.connect(self._hide_last_illumination_row)
        for row in self.illumination_rows:
            slider = row.get("slider")
            spin = row.get("spin")
            name_edit = row.get("name")
            host_edit = row.get("host")
            toggle = row.get("toggle")
            if isinstance(slider, QtWidgets.QSlider) and isinstance(spin, QtWidgets.QSpinBox):
                slider.valueChanged.connect(spin.setValue)
                spin.valueChanged.connect(slider.setValue)
                spin.valueChanged.connect(lambda _=None, r=row: self._on_illumination_row_changed(r))
            if isinstance(name_edit, QtWidgets.QLineEdit):
                name_edit.textChanged.connect(lambda _=None, r=row: self._on_illumination_row_changed(r))
            if isinstance(host_edit, QtWidgets.QLineEdit):
                host_edit.textChanged.connect(lambda _=None, r=row: self._on_illumination_row_changed(r))
            if isinstance(toggle, QtWidgets.QCheckBox):
                toggle.toggled.connect(lambda _=None, r=row: self._on_illumination_row_changed(r))

        self._on_edf_toggled(self.chk_edf.isChecked())

        self.act_calibrate.triggered.connect(self._calibrate)
        self.act_ruler.triggered.connect(self._start_ruler)
        self.measure_view.calibration_measured.connect(self._on_calibration_done)

        # camera controls
        self.exp_spin.valueChanged.connect(self._apply_exposure)
        self.autoexp_chk.toggled.connect(self._apply_exposure)
        self.gain_spin.valueChanged.connect(lambda v: self._apply_gain(v * 100))
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
        self.depth_combo.currentIndexChanged.connect(self._apply_color_depth)
        self.raw_chk.toggled.connect(self._apply_raw)
        self.bin_combo.currentIndexChanged.connect(self._apply_binning)
        self.res_combo.currentIndexChanged.connect(self._apply_resolution)
        self.btn_roi_full.clicked.connect(lambda: self._apply_roi('full'))
        self.btn_roi_2048.clicked.connect(lambda: self._apply_roi(2048))
        self.btn_roi_1024.clicked.connect(lambda: self._apply_roi(1024))
        self.btn_roi_512.clicked.connect(lambda: self._apply_roi(512))
        self.speed_spin.valueChanged.connect(self._apply_speed)

        # scripts
        self.btn_run_example_script.clicked.connect(self._run_example_script)

        # spectrometers
        self.btn_spec_refresh.clicked.connect(self._refresh_spectrometers)
        self.btn_spec_connect.clicked.connect(self._connect_spectrometer_from_list)
        self.btn_spec_disconnect.clicked.connect(self._disconnect_spectrometer)
        self.spectrometer_list.itemDoubleClicked.connect(
            lambda *_: self._connect_spectrometer_from_list()
        )
        self.spectrometer_manager.devices_changed.connect(self._on_spectrometers_changed)
        self.spectrometer_manager.device_connected.connect(self._on_spectrometer_connected)

    def _init_persistent_fields(self):
        def bind(spin, key):
            spin.setValue(self.profiles.get(key, spin.value()))
            spin.valueChanged.connect(lambda v, k=key: (self.profiles.set(k, float(v)), self.profiles.save()))
        for axis in ('x', 'y', 'z'):
            bind(getattr(self, f'step{axis}_spin'), f'jog.step.{axis}')
            bind(getattr(self, f'feed{axis}_spin'), f'jog.feed.{axis}')
            bind(getattr(self, f'abs{axis}_spin'), f'jog.abs.{axis}')
        # autofocus
        bind(self.af_range, 'autofocus.range_mm')
        bind(self.af_coarse, 'autofocus.coarse_step_mm')
        bind(self.af_fine, 'autofocus.fine_step_mm')
        # focus stacking
        bind(self.stack_range, 'focus_stack.range_mm')
        bind(self.stack_step, 'focus_stack.step_mm')

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

    def _on_autoprefix_toggled(self, checked: bool):
        self.auto_prefix = checked
        self.profiles.set('capture.auto_prefix', checked)
        self.profiles.save()

    def _on_format_changed(self, text: str):
        self.capture_format = text.lower()
        self.profiles.set('capture.format', self.capture_format)
        self.profiles.save()

    def _on_scale_bar_toggled(self, checked: bool):
        self.measure_view.set_scale_bar(checked, self.current_lens.um_per_px)
        self.profiles.set('ui.scale_bar', checked)
        self.profiles.save()

    def _on_edf_toggled(self, checked: bool):
        self.chk_delete_stack.setEnabled(checked)
        if not checked:
            self.chk_delete_stack.setChecked(False)

    def _on_lens_changed(self, index: int):
        name = self.lens_combo.itemData(index)
        if not name:
            return
        lens = self.lenses.get(name)
        if not lens:
            lens = Lens(name, 1.0)
            self.lenses[name] = lens
        self.current_lens = lens
        # Update the live preview scale bar immediately so that both the
        # on-screen view and any subsequent captures use the latest
        # calibration. This also refreshes cached scale values used when
        # drawing the scale bar on captured images.
        self.measure_view.set_scale_bar(
            self.chk_scale_bar.isChecked(), lens.um_per_px
        )
        self.profiles.set('measurement.current_lens', name)
        self.profiles.set(f'measurement.lenses.{name}.um_per_px', lens.um_per_px)
        self.profiles.save()
        self._update_lens_for_resolution()

    def _add_lens(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Lens", "Lens name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        lens = self.lenses.get(name)
        if lens is None:
            lens = Lens(name, 1.0)
            self.lenses[name] = lens
            self.profiles.set(f"measurement.lenses.{name}.um_per_px", 1.0)
            self.profiles.save()
        self.current_lens = lens
        self._refresh_lens_combo()
        idx = self.lens_combo.findData(name)
        if idx >= 0:
            self.lens_combo.setCurrentIndex(idx)
            self._on_lens_changed(idx)

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

    def _apply_focus_bias_for_target(self, x: float, y: float, z_target: float):
        area = None
        if self.focus_mgr.areas:
            area = self.focus_mgr.select_area(x, y)
            if area is None:
                area = max(self.focus_mgr.areas, key=lambda a: a.priority)
        if area and getattr(area, "model", None):
            predicted = float(area.model.predict(x, y))
            has_z_changed = (
                self._last_requested_z is None
                or not math.isclose(z_target, self._last_requested_z, rel_tol=1e-9, abs_tol=1e-9)
            )
            new_bias = float(z_target - predicted)
            if has_z_changed and math.isfinite(new_bias):
                if not math.isclose(new_bias, self.focus_mgr.z_bias, rel_tol=1e-9, abs_tol=1e-9):
                    log(
                        f"Adjusting leveling bias to {new_bias:+.6f} "
                        f"(target={z_target:.6f}, predicted={predicted:.6f})"
                    )
                    self.focus_mgr.z_bias = new_bias
        self._update_level_equation_label(area)

    def _update_level_equation_label(self, area=None):
        bias = getattr(self.focus_mgr, "z_bias", 0.0)
        bias_text = f"{bias:+.6f} mm"
        if area is None and self.focus_mgr.areas:
            area = max(self.focus_mgr.areas, key=lambda a: a.priority)
        model = getattr(area, "model", None) if area else None
        if model:
            eq_text = model.equation()
            if eq_text.startswith("z = "):
                eq_text = eq_text[4:]
            eq_text = eq_text.strip()
            self.level_equation.setText(f"z = {eq_text} {bias_text} (bias)")
        elif self.focus_mgr.areas:
            self.level_equation.setText(f"Leveling bias: {bias_text}")
        else:
            self.level_equation.setText(f"Bias: {bias_text}")

    def _move_to_coords(self):
        if not self.stage_worker:
            log("Move ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return
        x = self.absx_spin.value()
        y = self.absy_spin.value()
        z_target = self.absz_spin.value()
        self._apply_focus_bias_for_target(x, y, z_target)
        self._last_requested_z = z_target
        z = z_target
        if self.leveling_enabled:
            z += self.focus_mgr.z_offset(x, y, z_target)
        feed = max(self.feedx_spin.value(), self.feedy_spin.value(), self.feedz_spin.value())
        log(f"Move to: x={x} y={y} z={z} F={feed}")
        self.stage_worker.enqueue(self.stage.move_absolute, x, y, z, feed, True)
        self.stage_worker.enqueue(self.stage.wait_for_moves)
        self.stage_worker.enqueue(self.stage.get_position, callback=self._on_stage_position)

    def _set_movement_controls_enabled(self, enabled: bool):
        controls = [
            self.btn_home_all,
            self.btn_home_x,
            self.btn_home_y,
            self.btn_home_z,
            self.btn_xm,
            self.btn_xp,
            self.btn_ym,
            self.btn_yp,
            self.btn_zm,
            self.btn_zp,
            self.btn_move_to_coords,
        ]
        for btn in controls:
            btn.setEnabled(enabled)

    @QtCore.Slot(str)
    def _append_log(self, line: str):
        self.log_view.appendPlainText(line)

    # --------------------------- CONNECT ---------------------------

    def _auto_connect_async(self):
        self._connect_camera()
        self._connect_stage_async()
        self._connect_dimmer_async()

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

    # --------------------------- DIMMER ---------------------------

    def _update_dimmer_controls_enabled(self, enabled: bool):
        for widget in (
            self.dimmer_toggle,
            self.dimmer_brightness_slider,
            self.dimmer_brightness_spin,
        ):
            widget.setEnabled(enabled)
        self.btn_dimmer_disconnect.setEnabled(enabled and self.dimmer is not None)

    def _connect_dimmer_async(self):
        host = (self.dimmer_host_edit.text() or "").strip()
        if not host:
            if self.sender():
                self.dimmer_status.setText("Illumination: host required")
            return

        self.dimmer_status.setText("Illumination: connecting…")
        self._update_dimmer_controls_enabled(False)

        def do_connect():
            dimmer = ShellyDimmer(host)
            state = dimmer.connect()
            return dimmer, state

        if self._dimmer_conn_thread:
            self._dimmer_conn_thread.quit()
            self._dimmer_conn_thread.wait()
        self._dimmer_conn_thread, self._dimmer_conn_worker = run_async(do_connect)
        self._dimmer_conn_worker.finished.connect(self._on_dimmer_connect)

    @QtCore.Slot(object, object)
    def _on_dimmer_connect(self, result, err):
        thread = self._dimmer_conn_thread
        self._dimmer_conn_thread = self._dimmer_conn_worker = None
        if err or not result:
            log(f"UI: dimmer connect failed: {err}")
            self._handle_dimmer_failure(err)
        else:
            dimmer, state = result
            self.dimmer = dimmer
            self.profiles.set("illumination.dimmer.host", dimmer.host)
            self.profiles.save()
            self._apply_dimmer_state(state)
            self.btn_dimmer_disconnect.setEnabled(True)
            log("UI: dimmer connected")
        if thread and thread != QtCore.QThread.currentThread():
            thread.wait()

    def _disconnect_dimmer(self):
        for attr in ("_dimmer_conn_thread", "_dimmer_op_thread"):
            thread = getattr(self, attr)
            worker_attr = f"{attr.replace('thread', 'worker')}"
            if thread:
                thread.quit()
                thread.wait()
            setattr(self, attr, None)
            setattr(self, worker_attr, None)
        self.dimmer = None
        self._update_dimmer_controls_enabled(False)
        try:
            self._updating_dimmer_ui = True
            self.dimmer_toggle.setChecked(False)
            self.dimmer_brightness_slider.setValue(self.dimmer_brightness_spin.value())
        finally:
            self._updating_dimmer_ui = False
        self.dimmer_status.setText("Illumination: disconnected")

    def _run_dimmer_action(self, fn):
        if not self.dimmer:
            self.dimmer_status.setText("Illumination: not connected")
            return
        self.dimmer_status.setText("Illumination: updating…")
        self._update_dimmer_controls_enabled(False)
        if self._dimmer_op_thread:
            self._dimmer_op_thread.quit()
            self._dimmer_op_thread.wait()
        self._dimmer_op_thread, self._dimmer_op_worker = run_async(fn)
        self._dimmer_op_worker.finished.connect(self._on_dimmer_action_done)

    @QtCore.Slot(object, object)
    def _on_dimmer_action_done(self, state, err):
        thread = self._dimmer_op_thread
        self._dimmer_op_thread = self._dimmer_op_worker = None
        if err or state is None:
            log(f"UI: dimmer command failed: {err}")
            self._handle_dimmer_failure(err)
        else:
            self._apply_dimmer_state(state)
        if thread and thread != QtCore.QThread.currentThread():
            thread.wait()

    def _apply_dimmer_state(self, state: ShellyState):
        if state is None:
            self._handle_dimmer_failure("no state")
            return
        self._updating_dimmer_ui = True
        try:
            self.dimmer_toggle.setChecked(state.on)
            self.dimmer_brightness_spin.setValue(state.brightness)
            self.dimmer_brightness_slider.setValue(state.brightness)
        finally:
            self._updating_dimmer_ui = False
        status = "on" if state.on else "off"
        self.dimmer_status.setText(f"Illumination: {status} ({state.brightness}%)")
        self._update_dimmer_controls_enabled(True)
        self.btn_dimmer_disconnect.setEnabled(True)
        self.profiles.set("illumination.dimmer.on", state.on)
        self.profiles.set("illumination.dimmer.brightness", state.brightness)
        self.profiles.save()

    def _on_dimmer_toggle(self, checked: bool):
        if self._updating_dimmer_ui:
            return
        self._run_dimmer_action(lambda: self.dimmer.set_on(checked) if self.dimmer else None)

    def _on_dimmer_brightness_changed(self):
        if self._updating_dimmer_ui:
            return
        value = int(self.dimmer_brightness_spin.value())
        self._run_dimmer_action(lambda: self.dimmer.set_brightness(value) if self.dimmer else None)

    def _handle_dimmer_failure(self, err=None):
        self.dimmer = None
        self._update_dimmer_controls_enabled(False)
        if err:
            self.dimmer_status.setText(f"Illumination: error ({err})")
        else:
            self.dimmer_status.setText("Illumination: unavailable")

    def _show_camera_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Cameras")
        lay = QtWidgets.QVBoxLayout(dlg)
        lst = QtWidgets.QListWidget()
        lay.addWidget(lst)
        for dev_id, name in list_cameras():
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.UserRole, dev_id)
            if self.camera and getattr(self.camera, "device_id", None) == dev_id:
                item.setCheckState(QtCore.Qt.Checked)
            lst.addItem(item)
        lst.itemDoubleClicked.connect(lambda it: self._on_camera_item_double(it, dlg))
        dlg.exec()

    def _on_camera_item_double(self, item, dlg):
        dev_id = item.data(QtCore.Qt.UserRole)
        if self.camera and getattr(self.camera, "device_id", None) == dev_id:
            self._disconnect_camera()
        else:
            self._disconnect_camera()
            self._connect_camera(dev_id)
        dlg.accept()

    def _show_stage_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Stages")
        lay = QtWidgets.QVBoxLayout(dlg)
        lst = QtWidgets.QListWidget()
        lay.addWidget(lst)
        ports = list(list_marlin_ports())
        if self.stage and getattr(self.stage, "port", None):
            cur = self.stage.port
            if cur not in ports:
                ports.insert(0, cur)
        for port in ports:
            item = QtWidgets.QListWidgetItem(port)
            item.setData(QtCore.Qt.UserRole, port)
            if self.stage and getattr(self.stage, "port", None) == port:
                item.setCheckState(QtCore.Qt.Checked)
            lst.addItem(item)
        lst.itemDoubleClicked.connect(lambda it: self._on_stage_item_double(it, dlg))
        dlg.exec()

    def _on_stage_item_double(self, item, dlg):
        port = item.data(QtCore.Qt.UserRole)
        if self.stage and getattr(self.stage, "port", None) == port:
            self._disconnect_stage()
        else:
            self._disconnect_stage()
            self._connect_stage_async(port)
        dlg.accept()

    def _open_spectroscopy(self):
        if self.spectroscopy_window is None:
            self.spectroscopy_window = SpectroscopyWindow(
                self.spectrometer_manager, self.profiles, self
            )
            self.spectroscopy_window.destroyed.connect(
                lambda: setattr(self, "spectroscopy_window", None)
            )
        self.spectroscopy_window.show()
        self.spectroscopy_window.raise_()
        self.spectroscopy_window.activateWindow()

    def _refresh_spectrometers(self):
        devices = self.spectrometer_manager.refresh()
        self._populate_spectrometer_list(devices)

    def _on_spectrometers_changed(self, devices):
        self._populate_spectrometer_list(devices)

    def _populate_spectrometer_list(self, devices):
        selection = self._current_spectrometer_descriptor()
        active_desc = self.spectrometer_manager.active_descriptor
        if selection is None and active_desc:
            selection = active_desc
        self.spectrometer_list.blockSignals(True)
        self.spectrometer_list.clear()
        for desc in devices:
            label = f"{desc.model} ({desc.serial_number})\n{desc.path}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, desc)
            if selection and selection == desc:
                item.setSelected(True)
            self.spectrometer_list.addItem(item)
        self.spectrometer_list.blockSignals(False)
        if selection:
            self._select_spectrometer_descriptor(selection)
        self._update_spectrometer_buttons()

    def _current_spectrometer_descriptor(self):
        item = self.spectrometer_list.currentItem()
        if item:
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, SpectrometerDescriptor):
                return data
        return None

    def _select_spectrometer_descriptor(self, desc: SpectrometerDescriptor):
        for i in range(self.spectrometer_list.count()):
            item = self.spectrometer_list.item(i)
            if item and item.data(QtCore.Qt.UserRole) == desc:
                item.setSelected(True)
                self.spectrometer_list.setCurrentItem(item)
                break

    def _connect_spectrometer_from_list(self):
        desc = self._current_spectrometer_descriptor()
        if not desc:
            return
        try:
            dev = self.spectrometer_manager.connect(desc)
            wavelengths = dev.get_wavelengths()
            if self.spectroscopy_window:
                self.spectroscopy_window.session.set_wavelengths(wavelengths)
        except Exception as exc:
            log(f"Spectrometer connect failed: {exc}")
        self._update_spectrometer_buttons()

    def _disconnect_spectrometer(self):
        desc = self._current_spectrometer_descriptor()
        self.spectrometer_manager.disconnect(desc)
        self._update_spectrometer_buttons()

    def _on_spectrometer_connected(self, desc, device):
        if desc:
            text = f"Spectrometer: {desc.label()}"
        else:
            text = "Spectrometer: —"
        self.spectrometer_status.setText(text)
        if desc:
            self._select_spectrometer_descriptor(desc)
        self._update_spectrometer_buttons()

    def _update_spectrometer_buttons(self):
        desc = self._current_spectrometer_descriptor()
        active = self.spectrometer_manager.get_active(desc) if desc else None
        has_selection = bool(desc)
        self.btn_spec_connect.setEnabled(has_selection and active is None)
        self.btn_spec_disconnect.setEnabled(active is not None)

    # --------------------------- CONNECT/DISCONNECT ---------------------------

    def _connect_camera(self, dev_id=None):
        if self.camera is not None:
            log("UI: camera already connected; skip re-open")
            return
        try:
            cam = create_camera(dev_id)
            self.camera = cam
            self.cam_status.setText(f"Camera: {self.camera.name()}")
            self.camera.start_stream()
            self._populate_speed_levels()
            self._apply_speed()
            # populate after stream start so all options are available
            self._populate_color_depths()
            self._populate_binning()
            self._populate_resolutions()
            QtCore.QTimer.singleShot(0, self._populate_resolutions)
            self._apply_camera_profile()
            self._sync_cam_controls()
            self.preview_timer.start()
            self.fps_timer.start()
            self._update_cam_buttons()
            log("UI: camera connected")
        except Exception as e:
            log(f"UI: camera connect failed: {e}")

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
        self.measure_view.clear_image()
        self.res_combo.clear()
        self.bin_combo.clear()
        self.depth_combo.clear()
        self._update_cam_buttons()

    def _connect_stage_async(self, port=None):
        if self.stage is not None:
            log("UI: stage already connected; skip re-probe")
            return

        def connect_stage():
            p = port or find_marlin_port()
            if not p:
                return None
            return StageMarlin(p)

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
            return f"{v:.6f}" if v is not None else "—"
        coords_line = (
            f"Pos: X{_fmt(self._last_pos['x'])} "
            f"Y{_fmt(self._last_pos['y'])} "
            f"Z{_fmt(self._last_pos['z'])}"
        )
        if self.stage_bounds:
            b = self.stage_bounds
            limits_line = (
                f"Limits: X[{b['xmin']:.6f},{b['xmax']:.6f}] "
                f"Y[{b['ymin']:.6f},{b['ymax']:.6f}] "
                f"Z[{b['zmin']:.6f},{b['zmax']:.6f}]"
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
            processed = frame
            try:
                has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            except Exception:
                has_cuda = False
            if has_cuda:
                try:
                    gpu = cv2.cuda_GpuMat()
                    gpu.upload(frame)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2RGB)
                    processed = gpu.download()
                except Exception:
                    processed = (
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if frame.ndim == 3 and frame.shape[2] == 3
                        else frame
                    )
            else:
                processed = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.ndim == 3 and frame.shape[2] == 3
                    else frame
                )
            qimg = numpy_to_qimage(processed)
            self.measure_view.set_image(qimg)

        if self.autoexp_chk.isChecked():
            try:
                self.exp_spin.blockSignals(True)
                self.gain_spin.blockSignals(True)
                ms = float(self.camera.get_exposure_ms())
                gain = float(self.camera.get_gain())
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

    def _populate_speed_levels(self):
        if not self.camera or not hasattr(self.camera, "get_speed_range"):
            self.speed_spin.setEnabled(False)
            return
        self.speed_spin.blockSignals(True)
        try:
            rng = self.camera.get_speed_range()
            if rng:
                self.speed_spin.setRange(rng[0], rng[-1])
                cur = self.camera.get_speed_level()
                if cur is not None:
                    self.speed_spin.setValue(cur)
            self.speed_spin.setEnabled(bool(rng))
        except Exception:
            self.speed_spin.setEnabled(False)
        self.speed_spin.blockSignals(False)

    def _populate_color_depths(self):
        if not self.camera or not hasattr(self.camera, "list_color_depths"):
            self.depth_combo.clear()
            self.depth_combo.setEnabled(False)
            return
        self.depth_combo.blockSignals(True)
        self.depth_combo.clear()
        try:
            depths = self.camera.list_color_depths()
            for d in depths:
                self.depth_combo.addItem(f"{d}-bit", d)
            cur = depths[0] if depths else None
            if hasattr(self.camera, "get_color_depth"):
                try:
                    cur = int(self.camera.get_color_depth())
                except Exception:
                    pass
            if cur is not None:
                pos = self.depth_combo.findData(cur)
                if pos >= 0:
                    self.depth_combo.setCurrentIndex(pos)
        except Exception:
            pass
        self.depth_combo.setEnabled(self.depth_combo.count() > 0)
        self.depth_combo.blockSignals(False)

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

    def _apply_camera_profile(self):
        if not self.camera:
            return
        p = self.profiles
        # Exposure and auto-exposure
        auto = p.get('camera.auto_exposure', self.autoexp_chk.isChecked(), expected_type=bool)
        ms = p.get('camera.exposure_ms', self.exp_spin.value(), expected_type=(int, float))
        self.autoexp_chk.blockSignals(True)
        self.autoexp_chk.setChecked(auto)
        self.autoexp_chk.blockSignals(False)
        self.exp_spin.blockSignals(True)
        self.exp_spin.setValue(ms)
        self.exp_spin.blockSignals(False)
        self._apply_exposure()
        # Gain
        gain = p.get('camera.gain', self.gain_spin.value(), expected_type=(int, float))
        self.gain_spin.blockSignals(True)
        self.gain_spin.setValue(gain)
        self.gain_spin.blockSignals(False)
        if not auto:
            self._apply_gain()
        # Brightness
        val = p.get('camera.brightness', self.brightness_spin.value(), expected_type=(int, float))
        self.brightness_spin.blockSignals(True)
        self.brightness_spin.setValue(val)
        self.brightness_spin.blockSignals(False)
        self._apply_brightness()
        # Contrast
        val = p.get('camera.contrast', self.contrast_spin.value(), expected_type=(int, float))
        self.contrast_spin.blockSignals(True)
        self.contrast_spin.setValue(val)
        self.contrast_spin.blockSignals(False)
        self._apply_contrast()
        # Saturation
        val = p.get('camera.saturation', self.saturation_spin.value(), expected_type=(int, float))
        self.saturation_spin.blockSignals(True)
        self.saturation_spin.setValue(val)
        self.saturation_spin.blockSignals(False)
        self._apply_saturation()
        # Hue
        val = p.get('camera.hue', self.hue_spin.value(), expected_type=(int, float))
        self.hue_spin.blockSignals(True)
        self.hue_spin.setValue(val)
        self.hue_spin.blockSignals(False)
        self._apply_hue()
        # Gamma
        val = p.get('camera.gamma', self.gamma_spin.value(), expected_type=(int, float))
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(val)
        self.gamma_spin.blockSignals(False)
        self._apply_gamma()
        # Color depth
        depth = p.get('camera.color_depth', self.depth_combo.currentData(), expected_type=(int, float))
        self.depth_combo.blockSignals(True)
        pos = self.depth_combo.findData(depth)
        if pos >= 0:
            self.depth_combo.setCurrentIndex(pos)
        self.depth_combo.blockSignals(False)
        self._apply_color_depth(self.depth_combo.currentIndex())
        # RAW mode
        raw = p.get('camera.raw', self.raw_chk.isChecked(), expected_type=bool)
        self.raw_chk.blockSignals(True)
        self.raw_chk.setChecked(raw)
        self.raw_chk.blockSignals(False)
        self._apply_raw(raw)
        # Binning
        bin_val = p.get('camera.binning', self.bin_combo.currentData(), expected_type=(int, float))
        self.bin_combo.blockSignals(True)
        pos = self.bin_combo.findData(bin_val)
        if pos >= 0:
            self.bin_combo.setCurrentIndex(pos)
        self.bin_combo.blockSignals(False)
        self._apply_binning(self.bin_combo.currentIndex())
        # Resolution
        res_idx = p.get('camera.resolution_index', self.res_combo.currentData(), expected_type=(int, float))
        self.res_combo.blockSignals(True)
        pos = self.res_combo.findData(res_idx)
        if pos >= 0:
            self.res_combo.setCurrentIndex(pos)
        self.res_combo.blockSignals(False)
        self._apply_resolution(self.res_combo.currentIndex())
        # USB speed level
        val = p.get('camera.usb_speed', self.speed_spin.value(), expected_type=(int, float))
        self.speed_spin.blockSignals(True)
        self.speed_spin.setValue(val)
        self.speed_spin.blockSignals(False)
        self._apply_speed()
    def _update_camera_control_availability(self, cam=None):
        cam = cam if cam is not None else self.camera
        has = lambda attr: cam is not None and hasattr(cam, attr)
        auto = self.autoexp_chk.isChecked()
        self.autoexp_chk.setEnabled(has("set_exposure_ms"))
        self.exp_spin.setEnabled(has("set_exposure_ms") and not auto)
        self.gain_spin.setEnabled(has("set_gain") and not auto)
        self.brightness_spin.setEnabled(has("set_brightness"))
        self.brightness_slider.setEnabled(has("set_brightness"))
        self.contrast_spin.setEnabled(has("set_contrast"))
        self.contrast_slider.setEnabled(has("set_contrast"))
        self.saturation_spin.setEnabled(has("set_saturation"))
        self.saturation_slider.setEnabled(has("set_saturation"))
        self.hue_spin.setEnabled(has("set_hue"))
        self.hue_slider.setEnabled(has("set_hue"))
        self.gamma_spin.setEnabled(has("set_gamma"))
        self.gamma_slider.setEnabled(has("set_gamma"))
        self.raw_chk.setEnabled(has("set_raw_fast_mono"))
        roi = has("set_center_roi")
        self.btn_roi_full.setEnabled(roi)
        self.btn_roi_2048.setEnabled(roi)
        self.btn_roi_1024.setEnabled(roi)
        self.btn_roi_512.setEnabled(roi)

    def _update_stage_buttons(self):
        connected = self.stage is not None and self.stage_worker is not None
        self._set_movement_controls_enabled(connected)
        controls = [
            self.btn_autofocus,
            self.btn_focus_stack,
            self.btn_raster_p1,
            self.btn_raster_p2,
            self.btn_raster_p3,
            self.btn_raster_p4,
            self.btn_run_raster,
            self.btn_level_p1,
            self.btn_level_p2,
            self.btn_level_p3,
            self.btn_start_level,
            self.btn_apply_level,
            self.btn_disable_level,
            self.btn_level_continue,
            self.btn_run_example_script,
        ]
        for w in controls:
            w.setEnabled(connected)
        self._update_stop_button()

    def _update_cam_buttons(self):
        connected = self.camera is not None
        controls = [
            self.btn_capture,
            self.chk_reticle,
            self.chk_scale_bar,
            self.measure_button,
            self.btn_clear_screen,
            self.capture_dir_edit,
            self.btn_browse_dir,
            self.capture_name_edit,
            self.autonumber_chk,
            self.autoprefix_chk,
            self.format_combo,
            self.depth_combo,
            self.raw_chk,
            self.bin_combo,
            self.res_combo,
            self.btn_roi_full,
            self.btn_roi_2048,
            self.btn_roi_1024,
            self.btn_roi_512,
            self.speed_spin,
        ]
        for w in controls:
            w.setEnabled(connected)
        self._update_camera_control_availability(self.camera if connected else None)

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
                gain = float(self.camera.get_gain())
                self.exp_spin.setValue(ms)
                self.gain_spin.setValue(gain)
            except Exception:
                pass
            finally:
                self.exp_spin.blockSignals(False)
                self.gain_spin.blockSignals(False)
        self._update_camera_control_availability()

    def _apply_gain(self, again=None):
        if not self.camera:
            return
        if again is None:
            again = self.gain_spin.value() * 100
        self.camera.set_gain(int(again))

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

    def _apply_color_depth(self, i: int):
        if not self.camera:
            return
        depth = self.depth_combo.currentData()
        if depth is None:
            return
        try:
            self.camera.set_color_depth(int(depth))
        except Exception:
            pass

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

    # ------------------------ CALIBRATION HANDLING ------------------------

    def _current_res_key(self) -> str | None:
        """Return current resolution as ``"{w}x{h}"`` or ``None``."""
        if not self.camera:
            return None
        try:
            idx = int(self.camera.get_resolution_index())
            res_list = getattr(self.camera, "resolutions", [])
            for ridx, w, h in res_list:
                if ridx == idx:
                    return f"{w}x{h}"
        except Exception:
            return None
        return None

    def _update_lens_for_resolution(self):
        """Update current lens calibration for active resolution."""
        res_key = self._current_res_key()
        lens = self.current_lens
        if res_key:
            if res_key in lens.calibrations:
                lens.um_per_px = lens.calibrations[res_key]
            elif lens.calibrations:
                # scale from first known calibration
                k, v = next(iter(lens.calibrations.items()))
                try:
                    w0, _ = map(int, k.split("x"))
                    w, _ = map(int, res_key.split("x"))
                    lens.um_per_px = v * (w0 / w)
                except Exception:
                    lens.um_per_px = v
            lens.calibrations[res_key] = lens.um_per_px
        self._refresh_lens_combo()
        self.measure_view.set_scale_bar(
            self.chk_scale_bar.isChecked(), lens.um_per_px
        )

    def _apply_resolution(self, i: int):
        if not self.camera: return
        idx = self.res_combo.currentData()
        if idx is None: return
        self.camera.set_resolution_index(int(idx))
        self._update_lens_for_resolution()

    def _apply_roi(self, mode):
        if not self.camera: return
        if mode == 'full':
            # reset ROI to full frame
            self.camera.set_center_roi(0, 0)
        else:
            side = int(mode)
            self.camera.set_center_roi(side, side)

    def _apply_speed(self):
        if not self.camera or not hasattr(self.camera, "set_speed_level"):
            return
        self.camera.set_speed_level(int(self.speed_spin.value()))

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
        if self.leveling_enabled:
            x0 = self._last_pos["x"] or 0.0
            y0 = self._last_pos["y"] or 0.0
            z0 = (
                self._last_requested_z
                if self._last_requested_z is not None
                else self._last_pos["z"] or 0.0
            )
            x = x0 + dx
            y = y0 + dy
            z_target = z0 + dz
            self._apply_focus_bias_for_target(x, y, z_target)
            self._last_requested_z = z_target
            z = z_target + self.focus_mgr.z_offset(x, y, z_target)
            self.stage_worker.enqueue(
                self.stage.move_absolute,
                x,
                y,
                z,
                f,
                wait_ok,
                callback=callback,
            )
        else:
            if dz:
                x0 = self._last_pos["x"] or 0.0
                y0 = self._last_pos["y"] or 0.0
                z0 = self._last_pos["z"] or 0.0
                z_target = z0 + dz
                self._apply_focus_bias_for_target(x0 + dx, y0 + dy, z_target)
                self._last_requested_z = z_target
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
        auto_prefix = self.auto_prefix
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
            try:
                has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            except Exception:
                has_cuda = False
            try:
                img = self.camera.snap(use_cuda=has_cuda)
            except TypeError:
                img = self.camera.snap()
            if img is not None:
                if has_cuda:
                    try:
                        gpu = cv2.cuda_GpuMat()
                        gpu.upload(img)
                        if self.chk_scale_bar.isChecked():
                            img = draw_scale_bar(gpu, self.current_lens.um_per_px)
                        else:
                            img = gpu.download()
                    except Exception as e:
                        log(f"CUDA capture path failed: {e}; falling back to CPU")
                        has_cuda = False
                        if self.chk_scale_bar.isChecked():
                            try:
                                img = draw_scale_bar(img, self.current_lens.um_per_px)
                            except Exception as e:
                                log(f"Scale bar draw error: {e}")
                else:
                    if self.chk_scale_bar.isChecked():
                        try:
                            img = draw_scale_bar(img, self.current_lens.um_per_px)
                        except Exception as e:
                            log(f"Scale bar draw error: {e}")
                pos = self.stage.get_position()
                meta = {
                    "Camera": self.camera.name(),
                    "Position": pos,
                    "Lens": self.current_lens.name,
                    "LensUmPerPx": self.current_lens.um_per_px,
                    "Exposure_ms": getattr(self.camera, "get_exposure_ms", lambda: None)(),
                    "Gain": getattr(self.camera, "get_gain", lambda: None)(),
                    "Time": datetime.datetime.now().isoformat(),
                }
                self.image_writer.save_single(
                    img,
                    directory=directory,
                    filename=name,
                    auto_prefix=auto_prefix,
                    auto_number=auto_num,
                    fmt=self.capture_format,
                    metadata=meta,
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
        if err:
            log(f"Autofocus error: {err}")
            QtWidgets.QMessageBox.critical(self, "Autofocus", str(err))
        else:
            log(f"Autofocus: best ΔZ={best:.6f} mm")
            QtWidgets.QMessageBox.information(
                self, "Autofocus", f"Best Z offset (relative): {best:.6f} mm"
            )
            if self.stage_worker:
                self.stage_worker.enqueue(
                    self.stage.get_position, callback=self._on_stage_position
                )

    @QtCore.Slot()
    def _cleanup_autofocus_thread(self):
        self._af_thread = None
        self._af_worker = None

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
            af = AutoFocus(
                self.stage,
                self.camera,
                position_cb=lambda: self.stage_worker.enqueue(
                    self.stage.get_position, callback=self._on_stage_position
                ),
            )
            best_z = af.coarse_to_fine(
                metric=metric,
                z_range_mm=float(self.af_range.value()),
                coarse_step_mm=float(self.af_coarse.value()),
                fine_step_mm=float(self.af_fine.value()),
                feed_mm_per_min=self.feedz_spin.value(),
            )
            return best_z

        log(f"Autofocus: metric={metric.value}")
        t, w = run_async(do_af)
        self._af_thread, self._af_worker = t, w
        self._af_thread.finished.connect(self._cleanup_autofocus_thread)
        w.finished.connect(self._on_autofocus_done)

    @QtCore.Slot()
    def _cleanup_leveling_thread(self):
        self._set_movement_controls_enabled(True)
        self._level_thread = None
        self._level_worker = None
        self._leveling = False
        self._level_continue_event.set()
        self.level_prompt.hide()
        self.btn_level_continue.hide()
        self._update_stop_button()

    @QtCore.Slot(object, object)
    def _on_leveling_done(self, area, err):
        self._set_movement_controls_enabled(True)
        self.btn_start_level.setEnabled(True)
        if err:
            log(f"Leveling error: {err}")
            self._set_leveling_status("Error")
            QtWidgets.QMessageBox.critical(self, "Leveling", str(err))
            return

        self._set_leveling_status("Complete")
        if area:
            self.focus_mgr.areas.clear()
            self.focus_mgr.z_bias = 0.0
            self.focus_mgr.add_area(area)
            model = getattr(area, "model", None)
            if model:
                eq = model.equation()
                log(f"Leveling model ({model.kind.value}): {eq}")
            else:
                log("Leveling complete but model data was unavailable")
            self._update_level_equation_label(area)
        else:
            log("Leveling complete but no area was returned; existing data retained")
            self._update_level_equation_label()

    @QtCore.Slot(str)
    def _set_leveling_status(self, text: str):
        self.level_status.setText(text)

    @QtCore.Slot()
    def _apply_leveling(self):
        if self.focus_mgr.areas:
            self.leveling_enabled = True
            self._set_leveling_status("Enabled")
            self._update_level_equation_label()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Leveling", "No leveling data to apply.")

    @QtCore.Slot()
    def _disable_leveling(self):
        self.leveling_enabled = False
        self._set_leveling_status("Disabled")
        self._update_level_equation_label()

    @QtCore.Slot(str)
    def _set_level_prompt(self, text: str):
        self.level_prompt.setText(text)
        self.level_prompt.setVisible(True)
        self.btn_level_continue.setVisible(True)

    @QtCore.Slot()
    def _on_level_continue(self):
        self.btn_level_continue.setVisible(False)
        self._level_continue_event.set()

    def _stage_workspace_polygon(self):
        """Return a polygon covering the full XY travel of the stage."""

        bounds = {}

        stage_bounds = getattr(self, "stage_bounds", None)
        if stage_bounds:
            for key in ("xmin", "xmax", "ymin", "ymax"):
                val = stage_bounds.get(key)
                if val is not None:
                    bounds[key] = float(val)

        fallback = getattr(self, "_stage_bounds_fallback", None)
        if fallback:
            for key in ("xmin", "xmax", "ymin", "ymax"):
                if key in bounds:
                    continue
                val = fallback.get(key)
                if val is not None:
                    bounds[key] = float(val)

        stage = getattr(self, "stage", None)
        if stage is not None:
            try:
                live = stage.get_bounds()
            except Exception as exc:
                log(f"Stage: failed to query bounds for workspace polygon: {exc}")
            else:
                if live:
                    for key in ("xmin", "xmax", "ymin", "ymax"):
                        val = live.get(key)
                        if val is not None:
                            bounds[key] = float(val)

        required = ("xmin", "xmax", "ymin", "ymax")
        if any(key not in bounds for key in required):
            return None

        xmin = bounds["xmin"]
        xmax = bounds["xmax"]
        ymin = bounds["ymin"]
        ymax = bounds["ymax"]

        return [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        ]

    def _set_level_point(self, idx: int):
        if not self.stage_worker:
            log("Leveling point ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return

        def cb(pos):
            if not pos:
                return
            try:
                x, y, _ = pos
            except Exception:
                return
            if idx == 1:
                self.level_x1_spin.setValue(x)
                self.level_y1_spin.setValue(y)
            elif idx == 2:
                self.level_x2_spin.setValue(x)
                self.level_y2_spin.setValue(y)
            elif idx == 3:
                self.level_x3_spin.setValue(x)
                self.level_y3_spin.setValue(y)

        self.stage_worker.enqueue(self.stage.get_position, callback=cb)

    def _run_leveling(self):
        if self._leveling:
            log("Leveling ignored: already running")
            return
        if not self.stage:
            log("Leveling ignored: stage not connected")
            return
        auto_mode = self.level_mode.currentText() == "Auto"
        if auto_mode and not self.camera:
            log("Leveling ignored: camera not connected")
            return

        method = self.level_method.currentText()
        kind = SurfaceKind[self.level_poly.currentText().upper()]
        rows = self.level_rows.value()
        cols = self.level_cols.value()
        metric = FocusMetric(self.metric_combo.currentText())
        z_range = float(self.af_range.value())
        coarse = float(self.af_coarse.value())
        fine = float(self.af_fine.value())
        feed_xy = self.feedx_spin.value()
        feed_z = self.feedz_spin.value()
        stage = self.stage
        camera = self.camera
        self._leveling = True
        self.btn_start_level.setEnabled(False)
        self._set_leveling_status("Starting...")

        def do_level():
            x1 = self.level_x1_spin.value()
            y1 = self.level_y1_spin.value()
            x2 = self.level_x2_spin.value()
            y2 = self.level_y2_spin.value()
            x3 = self.level_x3_spin.value()
            y3 = self.level_y3_spin.value()
            xs_vals = [x1, x2, x3]
            ys_vals = [y1, y2, y3]
            xmin, xmax = min(xs_vals), max(xs_vals)
            ymin, ymax = min(ys_vals), max(ys_vals)
            if method == "Three-point":
                coords = [(x1, y1), (x2, y2), (x3, y3)]
                total = len(coords)
                pts = []
                for idx, (x, y) in enumerate(coords, 1):
                    QtCore.QMetaObject.invokeMethod(
                        self,
                        "_set_leveling_status",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, f"Point {idx}/{total}"),
                    )
                    stage.move_absolute(x=x, y=y, feed_mm_per_min=feed_xy)
                    stage.wait_for_moves()
                    if auto_mode:
                        af = AutoFocus(stage, camera)
                        _ = af.coarse_to_fine(
                            metric=metric,
                            z_range_mm=z_range,
                            coarse_step_mm=coarse,
                            fine_step_mm=fine,
                            feed_mm_per_min=feed_z,
                        )
                    else:
                        self._level_continue_event.clear()
                        msg = (
                            f"Manually focus at point {idx} of {total} then press Next to continue."
                        )
                        QtCore.QMetaObject.invokeMethod(
                            self,
                            "_set_level_prompt",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, msg),
                        )
                        self._level_continue_event.wait()
                    pos = stage.get_position()
                    if pos:
                        x_meas, y_meas, z = pos
                    else:
                        x_meas, y_meas, z = x, y, 0.0
                    pts.append((x_meas, y_meas, z))
            else:
                xs = np.linspace(xmin, xmax, cols)
                ys = np.linspace(ymin, ymax, rows)
                coords = [(x, y) for y in ys for x in xs]
                total = len(coords)
                pts = []
                for idx, (x, y) in enumerate(coords, 1):
                    QtCore.QMetaObject.invokeMethod(
                        self,
                        "_set_leveling_status",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, f"Point {idx}/{total}"),
                    )
                    stage.move_absolute(x=x, y=y, feed_mm_per_min=feed_xy)
                    stage.wait_for_moves()
                    if auto_mode:
                        af = AutoFocus(stage, camera)
                        _ = af.coarse_to_fine(
                            metric=metric,
                            z_range_mm=z_range,
                            coarse_step_mm=coarse,
                            fine_step_mm=fine,
                            feed_mm_per_min=feed_z,
                        )
                    else:
                        self._level_continue_event.clear()
                        msg = (
                            f"Manually focus at point {idx} of {total} then press Next to continue."
                        )
                        QtCore.QMetaObject.invokeMethod(
                            self,
                            "_set_level_prompt",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, msg),
                        )
                        self._level_continue_event.wait()
                    pos = stage.get_position()
                    if pos:
                        x_meas, y_meas, z = pos
                    else:
                        x_meas, y_meas, z = x, y, 0.0
                    pts.append((x_meas, y_meas, z))
            model = SurfaceModel(kind)
            model.fit(pts)
            polygon = self._stage_workspace_polygon()
            if polygon is None:
                log(
                    "Leveling: stage bounds unavailable; using probed area bounding box"
                )
                polygon = [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                ]
            area = Area(
                "bed",
                polygon,
                model,
            )
            return area
        # Allow manual stage movement while leveling so users can position
        # the stage if needed. Previously movement controls were disabled
        # here which prevented manual adjustments during the leveling
        # process.
        t, w = run_async(do_level)
        self._level_thread, self._level_worker = t, w
        self._level_thread.finished.connect(self._cleanup_leveling_thread)
        w.finished.connect(self._on_leveling_done)
        self._update_stop_button()

    @QtCore.Slot()
    def _cleanup_focus_stack_thread(self):
        self._stack_thread = None
        self._stack_worker = None
        self._update_stop_button()

    @QtCore.Slot(object, object)
    def _on_focus_stack_done(self, best_idx, err):
        self.btn_focus_stack.setEnabled(True)
        if err:
            log(f"Focus stack error: {err}")
            QtWidgets.QMessageBox.critical(self, "Focus Stack", str(err))
        else:
            msg = "Focus stack complete"
            if best_idx is not None:
                msg += f"; best index {best_idx}"
            if self._last_stack_dir:
                msg += f"\nSaved to {self._last_stack_dir}"
            log(msg)
            QtWidgets.QMessageBox.information(self, "Focus Stack", msg)

    def _run_focus_stack(self):
        if not (self.stage and self.camera):
            log("Focus stack ignored: stage or camera not connected")
            return
        step = float(self.stack_step.value())
        if step <= 0:
            QtWidgets.QMessageBox.warning(self, "Focus Stack", "Step must be > 0")
            return
        rng = float(self.stack_range.value())
        feed = self.feedz_spin.value()
        writer = ImageWriter(self.capture_dir)
        stack_dir = os.path.join(self.capture_dir, self.capture_name)
        os.makedirs(stack_dir, exist_ok=True)
        self._last_stack_dir = stack_dir
        self.btn_focus_stack.setEnabled(False)

        def do_stack():
            af = AutoFocus(self.stage, self.camera)
            return af.focus_stack(
                rng,
                step,
                writer,
                directory=stack_dir,
                base_name=self.capture_name,
                metric=FocusMetric.LAPLACIAN,
                feed_mm_per_min=feed,
                fmt=self.capture_format,
                lens_name=self.current_lens.name,
                fuse_edf=self.chk_edf.isChecked(),
                delete_stack=self.chk_delete_stack.isChecked(),
            )

        log(f"Focus stack: half-range={rng} step={step} dir={stack_dir}")
        t, w = run_async(do_stack)
        self._stack_thread, self._stack_worker = t, w
        self._stack_thread.finished.connect(self._cleanup_focus_stack_thread)
        w.finished.connect(self._on_focus_stack_done)
        self._update_stop_button()

    def _set_raster_point(self, idx: int):
        if not self.stage_worker:
            log("Raster point ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return

        def cb(pos):
            if not pos:
                return
            try:
                x, y, _ = pos
            except Exception:
                return
            if idx == 1:
                self.rast_x1_spin.setValue(x)
                self.rast_y1_spin.setValue(y)
            elif idx == 2:
                self.rast_x2_spin.setValue(x)
                self.rast_y2_spin.setValue(y)
            elif idx == 3:
                self.rast_x3_spin.setValue(x)
                self.rast_y3_spin.setValue(y)
            elif idx == 4:
                self.rast_x4_spin.setValue(x)
                self.rast_y4_spin.setValue(y)

        self.stage_worker.enqueue(self.stage.get_position, callback=cb)

    def _run_raster(self):
        if not (self.stage and self.camera):
            log("Raster ignored: stage or camera not connected")
            return
        if self._raster_thread or self._raster_runner:
            log("Raster ignored: raster already running")
            QtWidgets.QMessageBox.warning(
                self, "Raster", "A raster operation is already in progress."
            )
            return
        cfg_kwargs = dict(
            rows=self.rows_spin.value(),
            cols=self.cols_spin.value(),
            feed_x_mm_min=self.feedx_spin.value(),
            feed_y_mm_min=self.feedy_spin.value(),
            autofocus=self.chk_raster_af.isChecked(),
            capture=self.chk_raster_capture.isChecked(),
            stack=self.chk_raster_stack.isChecked(),
            stack_range_mm=float(self.stack_range.value()),
            stack_step_mm=float(self.stack_step.value()),
            fuse_edf=self.chk_edf.isChecked(),
            delete_stack=self.chk_delete_stack.isChecked(),
            af_range_mm=float(self.af_range.value()),
            af_coarse_step_mm=float(self.af_coarse.value()),
            af_fine_step_mm=float(self.af_fine.value()),
        )
        # common required points
        x1 = self.rast_x1_spin.value()
        y1 = self.rast_y1_spin.value()
        x2 = self.rast_x2_spin.value()
        y2 = self.rast_y2_spin.value()
        mode = self.raster_mode_combo.currentText()

        if mode == "2-point":
            if x1 == x2 and y1 == y2:
                QtWidgets.QMessageBox.warning(
                    self, "Raster", "Points 1 and 2 must be distinct."
                )
                return
            left, right = sorted([x1, x2])
            top, bottom = sorted([y1, y2])
            cfg_kwargs.update(
                x1_mm=left,
                y1_mm=top,
                x2_mm=right,
                y2_mm=top,
                x3_mm=left,
                y3_mm=bottom,
                x4_mm=right,
                y4_mm=bottom,
                mode="rectangle",
            )
        elif mode == "3-point":
            x3 = self.rast_x3_spin.value()
            y3 = self.rast_y3_spin.value()
            if (x3, y3) in [(x1, y1), (x2, y2)]:
                QtWidgets.QMessageBox.warning(
                    self, "Raster", "Point 3 must be distinct from points 1 and 2."
                )
                return
            cfg_kwargs.update(
                x1_mm=x1,
                y1_mm=y1,
                x2_mm=x2,
                y2_mm=y2,
                x3_mm=x3,
                y3_mm=y3,
                mode="parallelogram",
            )
        elif mode == "4-point":
            x3 = self.rast_x3_spin.value()
            y3 = self.rast_y3_spin.value()
            x4 = self.rast_x4_spin.value()
            y4 = self.rast_y4_spin.value()
            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            if len(set(points)) < 4:
                QtWidgets.QMessageBox.warning(
                    self, "Raster", "All four points must be distinct."
                )
                return
            cfg_kwargs.update(
                x1_mm=x1,
                y1_mm=y1,
                x2_mm=x2,
                y2_mm=y2,
                x3_mm=x3,
                y3_mm=y3,
                x4_mm=x4,
                y4_mm=y4,
                mode="trapezoid",
            )
        else:
            cfg_kwargs.update(
                x1_mm=x1,
                y1_mm=y1,
                x2_mm=x2,
                y2_mm=y2,
                mode="rectangle",
            )

        cfg = RasterConfig(**cfg_kwargs)

        directory = self.capture_dir
        name = self.capture_name
        auto_prefix = self.auto_prefix
        auto_num = self.auto_number
        fmt = self.capture_format

        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            log(f"Raster aborted: cannot create directory {directory}: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Raster", f"Unable to create directory:\n{directory}\n{e}"
            )
            return

        if not name:
            log("Raster aborted: filename empty")
            QtWidgets.QMessageBox.critical(self, "Raster", "Filename cannot be empty.")
            return
        if re.search(r"[\\/:*?\"<>|]", name):
            log("Raster aborted: illegal characters in filename")
            QtWidgets.QMessageBox.critical(
                self,
                "Raster",
                "Filename contains illegal characters (\\ / : * ? \" < > |).",
            )
            return

        runner = RasterRunner(
            self.stage,
            self.camera,
            self.image_writer,
            cfg,
            directory=directory,
            base_name=name,
            auto_prefix=auto_prefix,
            auto_number=auto_num,
            fmt=fmt,
            position_cb=lambda pos: self.stage_worker.enqueue(
                self.stage.get_position, callback=self._on_stage_position
            ),
            lens_name=self.current_lens.name,
            lens_um_per_px=self.current_lens.um_per_px,
            scale_bar_um_per_px=self.current_lens.um_per_px if self.chk_scale_bar.isChecked() else None,
        )
        self._raster_runner = runner

        def do_raster():
            runner.run()
            return True

        log("Raster: starting")
        self._set_movement_controls_enabled(False)
        t, w = run_async(do_raster)
        self._raster_thread, self._raster_worker = t, w
        self.btn_run_raster.setEnabled(False)
        self._update_stop_button()
        w.finished.connect(self._on_raster_finished)

    def _stop_all(self):
        self._set_movement_controls_enabled(False)
        if self._raster_runner:
            log("Raster: stop requested")
            self._raster_runner.stop()
            if self._raster_thread:
                self._raster_thread.quit()
                self._raster_thread.wait()
        if self._level_thread:
            log("Leveling: stop requested")
            self._level_thread.requestInterruption()
            self._level_continue_event.set()
        if self._stack_thread:
            log("Focus stack: stop requested")
            self._stack_thread.requestInterruption()

    def _update_stop_button(self):
        active = bool(self._raster_runner or self._level_thread or self._stack_thread)
        self.btn_stop.setEnabled(active)

    @QtCore.Slot(object, object)
    def _on_raster_finished(self, res, err):
        thread = self._raster_thread
        if thread and thread.isRunning():
            thread.wait()
        self._raster_runner = None
        self._raster_thread = None
        self._raster_worker = None
        self._set_movement_controls_enabled(True)
        self.btn_run_raster.setEnabled(True)
        self._update_stop_button()
        if self.stage_worker:
            self.stage_worker.enqueue(
                self.stage.get_position, callback=self._on_stage_position
            )
        log("Raster: done" if not err else f"Raster error: {err}")

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

    # --------------------------- MEASUREMENT ---------------------------

    def _calibrate(self):
        self.measure_view.start_calibration()

    def _start_ruler(self):
        """Begin ruler measurement using the active lens calibration.

        This bypasses the previous interactive calibration prompt and
        instead directly uses the stored ``µm per pixel`` value for the
        currently selected lens.  It assumes that ``self.current_lens`` has
        been calibrated beforehand (either via a prior calibration run or
        values loaded from profiles).
        """

        self.measure_view.start_ruler(self.current_lens.um_per_px)

    def _on_calibration_done(self, pixels: float):
        microns, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Calibration",
            f"Measured {pixels:.2f} px. Enter real-world distance (µm):",
            self.current_lens.um_per_px * pixels,
            0.000001,
            1e9,
            6,
        )
        if ok and pixels > 0:
            um_per_px = microns / pixels
            self.current_lens.um_per_px = um_per_px
            res_key = self._current_res_key() or "default"
            self.current_lens.calibrations[res_key] = um_per_px
            self.profiles.set(
                f"measurement.lenses.{self.current_lens.name}.calibrations.{res_key}",
                um_per_px,
            )
            self.profiles.set(
                f"measurement.lenses.{self.current_lens.name}.um_per_px", um_per_px
            )
            self.profiles.save()
            self._refresh_lens_combo()
            self.measure_view.set_scale_bar(
                self.chk_scale_bar.isChecked(), um_per_px
            )

    # --------------------------- PERSISTENCE ---------------------------

    def _persistent_widgets(self):
        """Return (widget, profile_path) pairs for UI persistence.

        Extend this list when introducing new widgets that should have their
        values loaded on startup and saved on close.
        """
        return [
            (self.stepx_spin, "ui.jog.stepx"),
            (self.stepy_spin, "ui.jog.stepy"),
            (self.stepz_spin, "ui.jog.stepz"),
            (self.feedx_spin, "ui.jog.feedx"),
            (self.feedy_spin, "ui.jog.feedy"),
            (self.feedz_spin, "ui.jog.feedz"),
            (self.absx_spin, "ui.move.absx"),
            (self.absy_spin, "ui.move.absy"),
            (self.absz_spin, "ui.move.absz"),
            # camera settings
            (self.exp_spin, "camera.exposure_ms"),
            (self.autoexp_chk, "camera.auto_exposure"),
            (self.gain_spin, "camera.gain"),
            (self.brightness_spin, "camera.brightness"),
            (self.contrast_spin, "camera.contrast"),
            (self.saturation_spin, "camera.saturation"),
            (self.hue_spin, "camera.hue"),
            (self.gamma_spin, "camera.gamma"),
            (self.depth_combo, "camera.color_depth"),
            (self.raw_chk, "camera.raw"),
            (self.bin_combo, "camera.binning"),
            (self.res_combo, "camera.resolution_index"),
            (self.speed_spin, "camera.usb_speed"),
            (self.lens_combo, "measurement.current_lens"),
            (self.chk_scale_bar, "ui.scale_bar"),
            (self.dimmer_host_edit, "illumination.dimmer.host"),
            (self.dimmer_toggle, "illumination.dimmer.on"),
            (self.dimmer_brightness_spin, "illumination.dimmer.brightness"),
        ]

    # --------------------------- PROFILES ---------------------------

    def _reload_profiles(self):
        self.profiles = Profiles.load_or_create()
        self.profile_combo.clear()
        self.profile_combo.addItems(self.profiles.list_profile_names())

    # --------------------------- CLOSE ---------------------------

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.profiles.set('ui.dark_mode', self.act_dark_mode.isChecked())
        for w, path in self._persistent_widgets():
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                val = round(w.value(), w.decimals())
            elif isinstance(w, QtWidgets.QAbstractSpinBox):
                val = w.value()
            elif isinstance(w, QtWidgets.QCheckBox):
                val = w.isChecked()
            elif isinstance(w, QtWidgets.QComboBox):
                data = w.currentData()
                val = data if data is not None else w.currentText()
            elif isinstance(w, QtWidgets.QLineEdit):
                val = w.text()
            else:
                continue
            self.profiles.set(path, val)
        self._persist_illumination_rows(save=False)
        self.profiles.save()
        try:
            if self.spectroscopy_window:
                self.spectroscopy_window.close()
            self.spectrometer_manager.disconnect()
            self._stop_all()
            self._disconnect_dimmer()
            if self._raster_thread:
                self._raster_thread.quit()
                self._raster_thread.wait()
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
            if hasattr(self, "system_tab"):
                self.system_tab.stop()
        finally:
            return super().closeEvent(e)
