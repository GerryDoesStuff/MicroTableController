from enum import Enum
import math
import os
import time
import logging
from typing import Optional, List

import numpy as np
import cv2

from ..io.storage import ImageWriter

logger = logging.getLogger(__name__)

try:
    _HAS_CUDA = (
        hasattr(cv2, "cuda")
        and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
    )
except Exception:
    _HAS_CUDA = False

logger.info("Autofocus metrics using %s", "CUDA" if _HAS_CUDA else "CPU")

class FocusMetric(str, Enum):
    LAPLACIAN = "LaplacianVar"
    TENENGRAD = "Tenengrad"

def metric_value(img, metric: FocusMetric):
    """Compute a focus metric for an image.

    Parameters
    ----------
    img : np.ndarray
        Input image. May be either a single-channel grayscale image or a
        three-channel RGB image.
    metric : FocusMetric
        The metric to compute.

    Returns
    -------
    float
        Calculated metric value.

    Raises
    ------
    ValueError
        If the image does not have 1 or 3 channels.
    """
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 1:
            gray = img[..., 0]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    logger.debug(
        "Computing %s metric on %s", metric, "CUDA" if _HAS_CUDA else "CPU"
    )

    if metric == FocusMetric.LAPLACIAN:
        if _HAS_CUDA:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(gray)
            # Assume 8-bit input; fallback will handle other types
            lap_filter = cv2.cuda.createLaplacianFilter(
                cv2.CV_8UC1, cv2.CV_64F
            )
            lap_gpu = lap_filter.apply(gpu_mat)
            lap = lap_gpu.download()
        else:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    elif metric == FocusMetric.TENENGRAD:
        if _HAS_CUDA:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(gray)
            sobel_x = cv2.cuda.createSobelFilter(
                cv2.CV_8UC1, cv2.CV_64F, 1, 0, ksize=3
            )
            sobel_y = cv2.cuda.createSobelFilter(
                cv2.CV_8UC1, cv2.CV_64F, 0, 1, ksize=3
            )
            gx_gpu = sobel_x.apply(gpu_mat)
            gy_gpu = sobel_y.apply(gpu_mat)
            gx = gx_gpu.download()
            gy = gy_gpu.download()
        else:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx * gx + gy * gy))
    else:
        raise ValueError(metric)

class AutoFocus:
    def __init__(self, stage, camera):
        self.stage = stage
        self.camera = camera

    def coarse_to_fine(
        self,
        metric: FocusMetric,
        z_range_mm=0.5,
        coarse_step_mm=0.01,
        fine_step_mm=0.002,
        feed_mm_per_min=240,
    ):
        if coarse_step_mm <= 0 or fine_step_mm <= 0:
            raise ValueError("coarse_step_mm and fine_step_mm must be > 0")
        samples = []
        steps = int(max(1, round(z_range_mm / coarse_step_mm)))
        zs = [(-steps + i) * coarse_step_mm for i in range(2 * steps + 1)]
        cumulative = 0.0
        for dz in zs:
            move = dz - cumulative
            self.stage.move_relative(dz=move, feed_mm_per_min=feed_mm_per_min)
            cumulative = dz
            self.stage.wait_for_moves()
            time.sleep(0.03)
            img = self.camera.snap()
            if img is None:
                continue
            samples.append((dz, metric_value(img, metric)))
        if not samples:
            return 0.0
        best_dz, _ = max(samples, key=lambda t: t[1])
        # Go to coarse best position
        self.stage.move_relative(dz=(best_dz - cumulative), feed_mm_per_min=feed_mm_per_min)
        self.stage.wait_for_moves()

        # Fine sweep around coarse best
        fine_range = 0.1 * z_range_mm
        fine_steps = int(max(1, math.floor(fine_range / fine_step_mm)))
        offsets = [(-fine_steps + i) * fine_step_mm for i in range(2 * fine_steps + 1)]
        fine_samples = []
        cumulative = 0.0
        for offset in offsets:
            move = offset - cumulative
            self.stage.move_relative(dz=move, feed_mm_per_min=feed_mm_per_min)
            self.stage.wait_for_moves()
            time.sleep(0.02)
            img = self.camera.snap()
            if img is None:
                continue
            fine_samples.append((best_dz + offset, metric_value(img, metric)))
            cumulative = offset

        if not fine_samples:
            # Return to coarse best if no fine samples were collected
            self.stage.move_relative(dz=-cumulative, feed_mm_per_min=feed_mm_per_min)
            self.stage.wait_for_moves()
            return best_dz

        best_fine_dz, _ = max(fine_samples, key=lambda t: t[1])
        # Move to the best fine position
        self.stage.move_relative(
            dz=(best_fine_dz - (best_dz + cumulative)), feed_mm_per_min=feed_mm_per_min
        )
        self.stage.wait_for_moves()
        return best_fine_dz

    def focus_stack(
        self,
        range_mm: float,
        step_mm: float,
        writer: ImageWriter,
        *,
        directory: Optional[str] = None,
        base_name: str = "",
        metric: Optional[FocusMetric] = None,
        feed_mm_per_min: float = 240,
        fmt: str = "png",
        lens_name: Optional[str] = None,
        use_mm: bool = False,
        fuse_edf: bool = False,
        delete_stack: bool = False,
    ) -> Optional[int]:
        """Sweep Z over ``range_mm`` in ``step_mm`` increments and capture frames.

        Parameters
        ----------
        range_mm : float
            Half of the sweep distance in millimeters. The stage will move from
            ``-range_mm`` to ``+range_mm`` relative to the starting position,
            covering a total span of ``2 * range_mm``.
        step_mm : float
            Step size in millimeters for each captured frame.
        writer : ImageWriter
            Destination image writer used to save the stack.
        directory : str, optional
            Directory in which to save images. If ``None``, ``writer.run_dir``
            is used.
        base_name : str, optional
            Base name used when saving each frame. If empty, frames are saved
            using only the depth index or depth value.
        metric : FocusMetric, optional
            If provided, compute the metric for each frame and return the index
            of the sharpest frame.
        feed_mm_per_min : float
            Feed rate for Z movement.
        fmt : str
            Image format passed to :meth:`ImageWriter.save_single`.
        lens_name : str, optional
            Name of the lens used for capture; included in image metadata.
        use_mm : bool, optional
            If ``True``, use the depth value in millimeters when constructing
            filenames; otherwise use a simple depth index.
        fuse_edf : bool, optional
            If ``True``, fuse the captured stack using EDF and save the fused
            image alongside the stack frames.
        delete_stack : bool, optional
            When ``True`` and EDF fusion is performed, delete the individual
            stack frame files after saving the fused image.

        Returns
        -------
        Optional[int]
            Index of the frame with highest focus metric, if ``metric`` is
            provided; otherwise ``None``.
        """

        if step_mm <= 0:
            raise ValueError("step_mm must be > 0")

        directory = directory or writer.run_dir
        os.makedirs(directory, exist_ok=True)

        steps = int(max(1, round(range_mm / step_mm)))
        zs = [(-steps + i) * step_mm for i in range(2 * steps + 1)]
        cumulative = 0.0
        metrics = []
        images = [] if fuse_edf else None
        fmt_lower = (fmt or "bmp").lower()
        ext_map = {
            "bmp": "bmp",
            "tif": "tif",
            "tiff": "tif",
            "png": "png",
            "jpg": "jpg",
            "jpeg": "jpg",
        }
        file_ext = ext_map.get(fmt_lower, "bmp")
        stack_files: Optional[List[str]] = [] if delete_stack else None

        get_exp = getattr(self.camera, "get_exposure_ms", None)
        exposure_s = 0.0
        if get_exp:
            try:
                exposure_s = float(get_exp()) / 1000.0
            except Exception:
                exposure_s = 0.0

        for i, dz in enumerate(zs):
            move = dz - cumulative
            self.stage.move_relative(dz=move, feed_mm_per_min=feed_mm_per_min)
            self.stage.wait_for_moves()
            if i == 0:
                time.sleep(0.5)
            time.sleep(exposure_s + 0.25)
            img = self.camera.snap()
            if img is None:
                if metric:
                    metrics.append(float("-inf"))
                continue
            if images is not None:
                if img.ndim == 3:
                    images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                else:
                    images.append(img)
            pos = self.stage.get_position()
            metadata = {
                "Camera": self.camera.name(),
                "Position": pos,
                "Lens": lens_name,
            }
            if use_mm:
                base = f"{dz:.4f}mm"
            else:
                base = f"{i:04d}"
            fname = f"{base_name}_{base}" if base_name else base
            if stack_files is not None:
                stack_files.append(os.path.join(directory, f"{fname}.{file_ext}"))
            writer.save_single(
                img,
                directory=directory,
                filename=fname,
                auto_number=False,
                fmt=fmt,
                metadata=metadata,
            )
            if metric:
                metrics.append(metric_value(img, metric))
            cumulative = dz

        # Return to starting position
        self.stage.move_relative(dz=-cumulative, feed_mm_per_min=feed_mm_per_min)
        self.stage.wait_for_moves()

        if images:
            from ..analysis.edf import fuse_stack as _edf_fuse_stack

            fused = _edf_fuse_stack(images, use_cuda=True)
            if fused.ndim == 3:
                fused = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)
            fname = f"{base_name}_edf" if base_name else "fused"
            metadata = {
                "Camera": self.camera.name(),
                "Lens": lens_name,
            }
            writer.save_single(
                fused,
                directory=directory,
                filename=fname,
                auto_number=False,
                fmt=fmt,
                metadata=metadata,
            )
            if stack_files:
                for frame_path in stack_files:
                    try:
                        os.remove(frame_path)
                    except FileNotFoundError:
                        continue
                    except Exception:
                        logger.warning(
                            "Failed to delete focus stack frame %s", frame_path, exc_info=True
                        )

        if metric and metrics:
            best_idx = int(np.argmax(metrics))
            return best_idx
        return None
