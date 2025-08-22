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

import time


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MicroStage App v0.1")
        self.resize(1280, 860)

        # device handles
        self.stage = None
        self.camera = None

        # persistent serial worker
        self.stage_thread = None
        self.stage_worker = None

        # async connect helper refs
        self._conn_thread = None
        self._conn_worker = None

        # image writer (per-run folder)
        self.image_writer = ImageWriter()

        # profiles
        self.profiles = Profiles.load_or_create()

        # preview timer
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(33)  # ~30 FPS
        self.preview_timer.timeout.connect(self._on_preview)

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

        hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)

        # Left column: device + profiles
        leftw = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(leftw)
        self.stage_status = QtWidgets.QLabel("Stage: —")
        self.cam_status = QtWidgets.QLabel("Camera: —")
        self.btn_connect = QtWidgets.QPushButton("Connect Devices")
        self.profile_combo = QtWidgets.QComboBox()
        self.btn_reload_profiles = QtWidgets.QPushButton("Reload Profiles")
        left.addWidget(self.stage_status)
        left.addWidget(self.cam_status)
        left.addWidget(self.btn_connect)
        left.addSpacing(8)
        left.addWidget(QtWidgets.QLabel("Profile:"))
        left.addWidget(self.profile_combo)
        left.addWidget(self.btn_reload_profiles)
        left.addStretch(1)

        # Center: live preview + capture
        centerw = QtWidgets.QWidget()
        center = QtWidgets.QVBoxLayout(centerw)
        self.live_label = QtWidgets.QLabel()
        self.live_label.setMinimumSize(800, 600)
        self.live_label.setAlignment(QtCore.Qt.AlignCenter)
        self.btn_capture = QtWidgets.QPushButton("Capture")
        center.addWidget(self.live_label, 1)
        center.addWidget(self.btn_capture, 0)

        # Right: tabs
        rightw = QtWidgets.QTabWidget()

        # -- Jog tab
        jog = QtWidgets.QWidget()
        j = QtWidgets.QGridLayout(jog)
        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setDecimals(3)
        self.step_spin.setRange(0.001, 1000.0)  # allow big moves
        self.step_spin.setValue(0.100)
        self.feed_spin = QtWidgets.QDoubleSpinBox()
        self.feed_spin.setRange(0.1, 500.0)  # mm/s -> we convert to mm/min when sending
        self.feed_spin.setValue(5.0)
        self.btn_home = QtWidgets.QPushButton("Home XYZ")
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
        j.addWidget(self.btn_home, 2, 0, 1, 2)
        j.addWidget(self.btn_xm, 3, 0)
        j.addWidget(self.btn_xp, 3, 1)
        j.addWidget(self.btn_ym, 4, 0)
        j.addWidget(self.btn_yp, 4, 1)
        j.addWidget(self.btn_zm, 5, 0)
        j.addWidget(self.btn_zp, 5, 1)
        rightw.addTab(jog, "Jog")

        # -- Autofocus tab
        af = QtWidgets.QWidget()
        a = QtWidgets.QGridLayout(af)
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems([m.value for m in FocusMetric])
        self.af_range = QtWidgets.QDoubleSpinBox()
        self.af_range.setRange(0.01, 5.0)
        self.af_range.setValue(0.5)
        self.af_coarse = QtWidgets.QDoubleSpinBox()
        self.af_coarse.setRange(0.001, 1.0)
        self.af_coarse.setValue(0.05)
        self.af_fine = QtWidgets.QDoubleSpinBox()
        self.af_fine.setRange(0.0005, 0.2)
        self.af_fine.setValue(0.01)
        self.btn_autofocus = QtWidgets.QPushButton("Run Autofocus")
        a.addWidget(QtWidgets.QLabel("Metric:"), 0, 0)
        a.addWidget(self.metric_combo, 0, 1)
        a.addWidget(QtWidgets.QLabel("Range (mm):"), 1, 0)
        a.addWidget(self.af_range, 1, 1)
        a.addWidget(QtWidgets.QLabel("Coarse step (mm):"), 2, 0)
        a.addWidget(self.af_coarse, 2, 1)
        a.addWidget(QtWidgets.QLabel("Fine step (mm):"), 3, 0)
        a.addWidget(self.af_fine, 3, 1)
        a.addWidget(self.btn_autofocus, 4, 0, 1, 2)
        rightw.addTab(af, "Autofocus")

        # -- Raster tab
        rast = QtWidgets.QWidget()
        r = QtWidgets.QGridLayout(rast)
        self.rows_spin = QtWidgets.QSpinBox()
        self.rows_spin.setRange(1, 1000)
        self.rows_spin.setValue(5)
        self.cols_spin = QtWidgets.QSpinBox()
        self.cols_spin.setRange(1, 1000)
        self.cols_spin.setValue(5)
        self.pitchx_spin = QtWidgets.QDoubleSpinBox()
        self.pitchx_spin.setRange(0.001, 50.0)
        self.pitchx_spin.setValue(1.0)
        self.pitchy_spin = QtWidgets.QDoubleSpinBox()
        self.pitchy_spin.setRange(0.001, 50.0)
        self.pitchy_spin.setValue(1.0)
        self.btn_run_raster = QtWidgets.QPushButton("Run Raster")
        r.addWidget(QtWidgets.QLabel("Rows:"), 0, 0)
        r.addWidget(self.rows_spin, 0, 1)
        r.addWidget(QtWidgets.QLabel("Cols:"), 1, 0)
        r.addWidget(self.cols_spin, 1, 1)
        r.addWidget(QtWidgets.QLabel("Pitch X (mm):"), 2, 0)
        r.addWidget(self.pitchx_spin, 2, 1)
        r.addWidget(QtWidgets.QLabel("Pitch Y (mm):"), 3, 0)
        r.addWidget(self.pitchy_spin, 3, 1)
        r.addWidget(self.btn_run_raster, 4, 0, 1, 2)
        rightw.addTab(rast, "Raster")

        # -- Scripts tab
        scripts = QtWidgets.QWidget()
        s = QtWidgets.QVBoxLayout(scripts)
        self.btn_run_example_script = QtWidgets.QPushButton("Run Example Script (Z stack)")
        s.addWidget(self.btn_run_example_script)
        s.addStretch(1)
        rightw.addTab(scripts, "Scripts")

        # Bottom log pane
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)

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

    def _connect_signals(self):
        self.btn_connect.clicked.connect(self._auto_connect_async)
        self.btn_capture.clicked.connect(self._capture)
        self.btn_home.clicked.connect(self._home)
        self.btn_xm.clicked.connect(lambda: self._jog(dx=-self.step_spin.value()))
        self.btn_xp.clicked.connect(lambda: self._jog(dx=+self.step_spin.value()))
        self.btn_ym.clicked.connect(lambda: self._jog(dy=-self.step_spin.value()))
        self.btn_yp.clicked.connect(lambda: self._jog(dy=+self.step_spin.value()))
        self.btn_zm.clicked.connect(lambda: self._jog(dz=-self.step_spin.value()))
        self.btn_zp.clicked.connect(lambda: self._jog(dz=+self.step_spin.value()))
        self.btn_autofocus.clicked.connect(self._run_autofocus)
        self.btn_run_raster.clicked.connect(self._run_raster)
        self.btn_run_example_script.clicked.connect(self._run_example_script)
        self.btn_reload_profiles.clicked.connect(self._reload_profiles)

    @QtCore.Slot(str)
    def _append_log(self, line: str):
        self.log_view.appendPlainText(line)

    # --------------------------- CONNECT ---------------------------

    def _auto_connect_async(self):
        # CAMERA (quick); don’t reopen if we already have one
        if self.camera is None:
            try:
                cam = create_camera()
                self.camera = cam
                self.cam_status.setText(f"Camera: {self.camera.name()}")
                self.camera.start_stream()
                self.preview_timer.start()
                log("UI: camera connected")
            except Exception as e:
                log(f"UI: camera connect failed: {e}")
        else:
            log("UI: camera already connected; skip re-open")

        # If stage already running, don't re-probe
        if self.stage is not None:
            log("UI: stage already connected; skip re-probe")
            return

        # STAGE async
        def connect_stage():
            port = find_marlin_port()
            if not port:
                return None
            return StageMarlin(port)

        self._conn_thread, self._conn_worker = run_async(connect_stage)

        def _done(stage, err):
            if err:
                log(f"UI: stage connect failed: {err}")
                self.stage_status.setText("Stage: not found")
                return
            if stage:
                self.stage = stage
                self.stage_status.setText("Stage: connected")
                log("UI: stage connected (async)")
                self._attach_stage_worker()
            else:
                self.stage_status.setText("Stage: not found")
                log("UI: stage not found")

        self._conn_worker.finished.connect(_done)

    def _attach_stage_worker(self):
        if not self.stage or self.stage_thread:
            return
        self.stage_thread = QtCore.QThread(self)
        self.stage_worker = SerialWorker(self.stage)
        self.stage_worker.moveToThread(self.stage_thread)
        self.stage_thread.started.connect(self.stage_worker.loop)
        self.stage_thread.start()

    # --------------------------- PREVIEW ---------------------------

    def _on_preview(self):
        if not self.camera:
            return
        frame = self.camera.get_latest_frame()
        if frame is None:
            return
        qimg = numpy_to_qimage(frame)
        self.live_label.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(
                self.live_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    # --------------------------- STAGE OPS ---------------------------

    def _home(self):
        if not self.stage_worker:
            log("Home ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return
        log("Home: G28")
        self.stage_worker.enqueue(self.stage.home_xyz)

    def _jog(self, dx=0, dy=0, dz=0):
        if not self.stage_worker:
            log("Jog ignored: stage not connected")
            QtWidgets.QMessageBox.warning(self, "Stage", "Stage not connected.")
            return
        f = max(1.0, float(self.feed_spin.value()) * 60.0)  # mm/s -> mm/min
        log(f"Jog: dx={dx} dy={dy} dz={dz} F={f}")
        self.stage_worker.enqueue(self.stage.move_relative, dx, dy, dz, f, False)

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
        self._last_thread = t
        self._last_worker = w
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
        self._last_thread = t
        self._last_worker = w

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
        self._last_thread = t
        self._last_worker = w
        w.finished.connect(lambda res, err: log("Raster: done" if not err else f"Raster error: {err}"))

    def _run_example_script(self):
        if not (self.stage and self.camera):
            log("Script ignored: stage or camera not connected")
            return
        from ..scripts.zstack_example import run as zrun

        def do_script():
            zrun(self.stage, self.camera, self.image_writer)
            return True

        log("Script: Z-stack example")
        t, w = run_async(do_script)
        self._last_thread = t
        self._last_worker = w
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
