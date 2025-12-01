from dataclasses import dataclass
import os
import time
import datetime
from math import isclose
from threading import Event
from typing import Optional

import numpy as np

from ..spectroscopy.io import (
    HypercubeData,
    ensure_data_directory,
    save_hypercube_hdf5,
    save_hypercube_npz,
)
from ..utils.img import draw_scale_bar

try:
    from .autofocus import AutoFocus, FocusMetric
except Exception:  # pragma: no cover - autofocus deps may be missing
    AutoFocus = None
    class FocusMetric:
        LAPLACIAN = None

@dataclass
class RasterConfig:
    """Configuration for raster scans.

    For ``mode="parallelogram"`` the three defined points correspond to the
    top-left (``x1_mm``, ``y1_mm``), top-right (``x2_mm``, ``y2_mm``), and
    bottom-right (``x3_mm``, ``y3_mm``) corners of the area. The remaining
    corner is inferred from these coordinates.

    When ``stack`` is enabled a small focus stack is captured at each tile,
    stepping the stage through ``stack_range_mm`` in increments of
    ``stack_step_mm``.
    """

    rows: int = 5
    cols: int = 5
    x1_mm: float = 0.0
    y1_mm: float = 0.0
    x2_mm: float = 1.0
    y2_mm: float = 0.0
    x3_mm: float = 0.0
    y3_mm: float = 1.0
    x4_mm: float = 1.0
    y4_mm: float = 1.0
    mode: str = "rectangle"  # rectangle, parallelogram, trapezoid
    serpentine: bool = True
    feed_x_mm_min: float = 50.0
    feed_y_mm_min: float = 50.0
    autofocus: bool = False
    capture: bool = True
    stack: bool = False
    stack_range_mm: float = 0.5
    stack_step_mm: float = 0.01
    fuse_edf: bool = False
    delete_stack: bool = False
    af_range_mm: float = 0.5
    af_coarse_step_mm: float = 0.01
    af_fine_step_mm: float = 0.002

class RasterRunner:
    def __init__(
        self,
        stage,
        camera,
        writer,
        cfg: RasterConfig,
        directory=None,
        base_name="tile",
        auto_prefix=False,
        auto_number=False,
        fmt="tif",
        position_cb=None,
        lens_name=None,
        lens_um_per_px: Optional[float] = None,
        scale_bar_um_per_px: Optional[float] = None,
        spectrometer=None,
        spectrometer_lock=None,
        spectrometer_integration_ms: Optional[float] = None,
        spectrometer_averages: Optional[int] = None,
        wavelengths=None,
        spectral_directory: Optional[str] = None,
    ):
        self.stage = stage
        self.camera = camera
        self.writer = writer
        self.cfg = cfg
        self.directory = directory
        self.base_name = base_name
        self.auto_prefix = auto_prefix
        self.auto_number = auto_number
        self.fmt = fmt
        self.position_cb = position_cb
        self.lens_name = lens_name
        self.lens_um_per_px = lens_um_per_px
        self.scale_bar_um_per_px = scale_bar_um_per_px
        self.spectrometer = spectrometer
        self.spectrometer_lock = spectrometer_lock
        self.spectrometer_integration_ms = spectrometer_integration_ms
        self.spectrometer_averages = spectrometer_averages
        self.wavelengths = np.asarray(wavelengths, dtype=float) if wavelengths is not None else None
        self.spectral_directory = spectral_directory

        self.coord_matrix = None
        self._spectral_cube = None
        self._spectral_coords = None
        self._spectral_ts = None
        self._stop = False

    def _build_coord_matrix(self):
        """Generate the coordinate matrix for the configured raster mode.

        For ``mode="parallelogram"`` the supplied points are interpreted as
        top-left, top-right, and bottom-right corners respectively.
        """
        if self.coord_matrix is not None:
            return self.coord_matrix

        cfg = self.cfg
        matrix = []
        if cfg.mode == "rectangle":
            col_dx = (cfg.x2_mm - cfg.x1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
            col_dy = (cfg.y2_mm - cfg.y1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
            row_dx = (cfg.x3_mm - cfg.x1_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
            row_dy = (cfg.y3_mm - cfg.y1_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
            for r in range(cfg.rows):
                base_x = cfg.x1_mm + row_dx * r
                base_y = cfg.y1_mm + row_dy * r
                row = []
                for c in range(cfg.cols):
                    x = base_x + col_dx * c
                    y = base_y + col_dy * c
                    row.append((x, y))
                matrix.append(row)
        elif cfg.mode == "parallelogram":
            col_vec_x = (cfg.x2_mm - cfg.x1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
            col_vec_y = (cfg.y2_mm - cfg.y1_mm) / (cfg.cols - 1) if cfg.cols > 1 else 0.0
            row_vec_x = (cfg.x3_mm - cfg.x2_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
            row_vec_y = (cfg.y3_mm - cfg.y2_mm) / (cfg.rows - 1) if cfg.rows > 1 else 0.0
            for r in range(cfg.rows):
                row = []
                for c in range(cfg.cols):
                    x = cfg.x1_mm + c * col_vec_x + r * row_vec_x
                    y = cfg.y1_mm + c * col_vec_y + r * row_vec_y
                    row.append((x, y))
                matrix.append(row)
        elif cfg.mode == "trapezoid":
            for r in range(cfg.rows):
                t_r = r / (cfg.rows - 1) if cfg.rows > 1 else 0.0
                start_x = cfg.x1_mm + (cfg.x3_mm - cfg.x1_mm) * t_r
                start_y = cfg.y1_mm + (cfg.y3_mm - cfg.y1_mm) * t_r
                end_x = cfg.x2_mm + (cfg.x4_mm - cfg.x2_mm) * t_r
                end_y = cfg.y2_mm + (cfg.y4_mm - cfg.y2_mm) * t_r
                row = []
                for c in range(cfg.cols):
                    t_c = c / (cfg.cols - 1) if cfg.cols > 1 else 0.0
                    x = start_x + (end_x - start_x) * t_c
                    y = start_y + (end_y - start_y) * t_c
                    row.append((x, y))
                matrix.append(row)
        else:  # pragma: no cover - validation
            raise ValueError(f"Unknown raster mode: {cfg.mode}")

        self.coord_matrix = matrix
        return matrix

    def stop(self):
        """Request that the raster scan stop after the current move."""
        self._stop = True

    def _maybe_capture_spectrum(self, row: int, col: int, x_mm: float, y_mm: float) -> None:
        if self.spectrometer is None:
            return
        lock = getattr(self.spectrometer_lock, "lock", None)
        unlock = getattr(self.spectrometer_lock, "unlock", None)
        if lock:
            lock()
        try:
            if self.spectrometer_integration_ms is not None:
                self.spectrometer.set_integration_time_ms(self.spectrometer_integration_ms)
            if self.spectrometer_averages is not None:
                self.spectrometer.set_averages(self.spectrometer_averages)
            spectrum = np.asarray(self.spectrometer.capture(), dtype=float)
        finally:
            if unlock:
                unlock()

        if self.wavelengths is None:
            try:
                self.wavelengths = np.asarray(self.spectrometer.get_wavelengths(), dtype=float)
            except Exception:
                self.wavelengths = np.arange(len(spectrum), dtype=float)

        if self._spectral_cube is None and self.wavelengths is not None:
            wl_len = len(self.wavelengths)
            self._spectral_cube = np.full((self.cfg.rows, self.cfg.cols, wl_len), np.nan, dtype=float)
            self._spectral_coords = np.full((self.cfg.rows, self.cfg.cols, 2), np.nan, dtype=float)
            self._spectral_ts = np.full((self.cfg.rows, self.cfg.cols), np.nan, dtype=float)

        if self._spectral_cube is None:
            return

        target_len = self._spectral_cube.shape[-1]
        if spectrum.shape[-1] != target_len:
            spectrum = np.resize(spectrum, target_len)
        self._spectral_cube[row, col, :] = spectrum[:target_len]
        self._spectral_coords[row, col, :] = (x_mm, y_mm)
        self._spectral_ts[row, col] = time.time()

    def _save_spectral_hypercube(self) -> None:
        if self._spectral_cube is None or self.wavelengths is None:
            return
        directory = ensure_data_directory(self.spectral_directory or self.directory)
        base = os.path.join(directory, f"{self.base_name}_spectra")
        descriptor = getattr(self.spectrometer, "descriptor", None)
        meta_device = None
        if descriptor is not None:
            label_fn = getattr(descriptor, "label", None)
            meta_device = label_fn() if callable(label_fn) else str(descriptor)
        metadata = {
            "rows": self.cfg.rows,
            "cols": self.cfg.cols,
            "mode": self.cfg.mode,
            "serpentine": self.cfg.serpentine,
            "autofocus": self.cfg.autofocus,
            "stack": self.cfg.stack,
            "timestamp": time.time(),
            "device": meta_device,
            "integration_ms": self.spectrometer_integration_ms,
            "averages": self.spectrometer_averages,
        }
        data = HypercubeData(
            wavelengths_nm=np.asarray(self.wavelengths, dtype=float),
            spectra=np.asarray(self._spectral_cube, dtype=float),
            coords_xy_mm=np.asarray(self._spectral_coords, dtype=float),
            timestamps_s=np.asarray(self._spectral_ts, dtype=float),
            metadata=metadata,
        )
        save_hypercube_npz(base + ".npz", data)
        save_hypercube_hdf5(base + ".h5", data)

    def run(self, stop_event: Optional[Event] = None):
        """Execute raster scan and capture images for each tile.

        The coordinate matrix is generated based on :class:`RasterConfig.mode`
        and then traversed in either serpentine or raster order.
        """

        coord_matrix = self._build_coord_matrix()

        if stop_event and stop_event.is_set():
            return

        start_x, start_y = coord_matrix[0][0]
        try:
            pos = self.stage.get_position()
        except Exception:
            pos = None
        if (
            pos is None
            or not (
                isclose(pos[0], start_x, abs_tol=1e-6)
                and isclose(pos[1], start_y, abs_tol=1e-6)
            )
        ):
            if stop_event and stop_event.is_set():
                return
            self.stage.move_absolute(x=start_x, y=start_y)
            self.stage.wait_for_moves()
            if self._stop:
                return
        current_x, current_y = start_x, start_y

        for r in range(self.cfg.rows):
            forward = (r % 2 == 0) or (not self.cfg.serpentine)
            cols = range(self.cfg.cols) if forward else range(self.cfg.cols - 1, -1, -1)
            for c in cols:
                target_x, target_y = coord_matrix[r][c]
                dx = target_x - current_x
                dy = target_y - current_y
                if dx or dy:
                    self.stage.move_relative(dx=dx, dy=dy)
                    current_x, current_y = target_x, target_y

                self.stage.wait_for_moves()
                if self._stop:
                    return
                if self.position_cb:
                    try:
                        pos = self.stage.get_position()
                    except Exception:
                        pos = None
                    self.position_cb(pos)
                time.sleep(0.03)

                do_af = bool(self.cfg.autofocus and AutoFocus)
                do_capture = bool(self.cfg.capture)
                do_stack = bool(self.cfg.stack and AutoFocus)

                if do_af:
                    af = AutoFocus(self.stage, self.camera)
                    af.coarse_to_fine(
                        metric=FocusMetric.LAPLACIAN,
                        z_range_mm=self.cfg.af_range_mm,
                        coarse_step_mm=self.cfg.af_coarse_step_mm,
                        fine_step_mm=self.cfg.af_fine_step_mm,
                    )
                    time.sleep(1)

                self._maybe_capture_spectrum(r, c, current_x, current_y)

                if do_capture:
                    img = self.camera.snap()
                    if img is not None:
                        if self.scale_bar_um_per_px is not None:
                            img = draw_scale_bar(img, self.scale_bar_um_per_px)
                        save_c = c
                        fname = f"{self.base_name}_r{r:04d}_c{save_c:04d}"
                        pos = self.stage.get_position()
                        metadata = {
                            "Camera": self.camera.name(),
                            "Position": pos,
                            "Lens": self.lens_name,
                            "LensUmPerPx": self.lens_um_per_px,
                            "Exposure_ms": getattr(self.camera, "get_exposure_ms", lambda: None)(),
                            "Gain": getattr(self.camera, "get_gain", lambda: None)(),
                            "Time": datetime.datetime.now().isoformat(),
                            "Row": r,
                            "Column": save_c,
                        }
                        self.writer.save_single(
                            img,
                            directory=self.directory,
                            filename=fname,
                            auto_prefix=self.auto_prefix,
                            auto_number=self.auto_number,
                            fmt=self.fmt,
                            metadata=metadata,
                        )
                    time.sleep(1)

                if do_stack:
                    af = AutoFocus(self.stage, self.camera)
                    stack_dir = os.path.join(
                        self.directory or self.writer.run_dir,
                        f"{self.base_name}_r{r:04d}_c{c:04d}_stack",
                    )
                    af.focus_stack(
                        range_mm=self.cfg.stack_range_mm,
                        step_mm=self.cfg.stack_step_mm,
                        writer=self.writer,
                        directory=stack_dir,
                        lens_name=self.lens_name,
                        fmt=self.fmt,
                        fuse_edf=self.cfg.fuse_edf,
                        delete_stack=self.cfg.delete_stack,
                        base_name=f"{self.base_name}_r{r:04d}_c{c:04d}",
                    )
                    time.sleep(1)

        if self._spectral_cube is not None and self.wavelengths is not None:
            self._save_spectral_hypercube()

