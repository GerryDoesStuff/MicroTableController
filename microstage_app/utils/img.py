import math
import logging

import numpy as np
from PySide6 import QtCore, QtGui
import cv2


# Scaling factors for the scale bar drawing used across the application
VERT_SCALE = 2  # line thickness multiplier
TEXT_SCALE = 4  # font size multiplier


logger = logging.getLogger(__name__)


def _scaled_font(base_font: QtGui.QFont) -> QtGui.QFont:
    """Return a copy of ``base_font`` scaled like :meth:`MeasureView.drawForeground`."""

    font = QtGui.QFont(base_font)
    point_size = font.pointSizeF()
    if point_size > 0:
        font.setPointSizeF(point_size * TEXT_SCALE)
    else:
        pixel_size = font.pixelSize()
        if pixel_size <= 0:
            pixel_size = 1
        font.setPixelSize(pixel_size * TEXT_SCALE)
    return font


def _scale_bar_geometry(width: int, height: int, um_per_px: float) -> tuple[float, int, int, int]:
    """Compute the unit length, pixel length, and anchor point for the bar."""

    max_um = 0.2 * width * um_per_px
    exp = math.floor(math.log10(max_um)) if max_um > 0 else 0
    nice_um = 10 ** exp
    for m in (5, 2, 1):
        candidate = m * (10 ** exp)
        if candidate <= max_um:
            nice_um = candidate
            break

    length_px = int(round(nice_um / um_per_px)) if um_per_px > 0 else 0
    max_length = max(0, width - 40)
    if length_px > max_length:
        length_px = max_length
        nice_um = length_px * um_per_px

    margin = 20
    x0 = int(round(width - margin - length_px))
    y0 = int(round(height - margin))
    return nice_um, length_px, x0, y0


def _paint_scale_bar(
    painter: QtGui.QPainter,
    width: int,
    height: int,
    um_per_px: float,
    *,
    draw_line: bool = True,
) -> tuple[str, QtGui.QFontMetricsF]:
    """Draw the scale bar and return the rendered label and font metrics."""

    nice_um, length_px, x0, y0 = _scale_bar_geometry(width, height, um_per_px)
    label = (
        f"{nice_um/1000:.2f} mm" if nice_um >= 1000 else f"{nice_um:.0f} µm"
    )

    painter.save()
    try:
        if draw_line and length_px > 0:
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 2 * VERT_SCALE))
            painter.drawLine(float(x0), float(y0), float(x0 + length_px), float(y0))

        font = _scaled_font(painter.font())
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        metrics = QtGui.QFontMetricsF(font)
        baseline = y0 - (7 * TEXT_SCALE) - metrics.descent()
        painter.drawText(float(x0), float(baseline), label)
    finally:
        painter.restore()

    return label, metrics


def _has_cuda() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


_HAS_CUDA = _has_cuda()
logger.info("Scale-bar drawing using %s", "CUDA" if _HAS_CUDA else "CPU")


def numpy_to_qimage(img: np.ndarray) -> QtGui.QImage:
    if img.ndim == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        return qimg.copy()
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w, _ = img.shape
        qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        return qimg.copy()
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")


def qimage_to_numpy(qimg: QtGui.QImage) -> np.ndarray:
    converted = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
    width = converted.width()
    height = converted.height()
    ptr = converted.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=converted.sizeInBytes())
    arr = arr.reshape((height, converted.bytesPerLine()))
    arr = arr[:, : width * 3]
    return arr.reshape((height, width, 3)).copy()


def _draw_scale_bar_cpu(
    img: np.ndarray,
    um_per_px: float,
    *,
    draw_line: bool = True,
) -> np.ndarray:
    """CPU implementation of the scale bar drawing using Qt."""

    if um_per_px <= 0:
        return img.copy()

    qimg = numpy_to_qimage(img)
    if qimg.format() != QtGui.QImage.Format_ARGB32:
        qimg = qimg.convertToFormat(QtGui.QImage.Format_ARGB32)

    painter = QtGui.QPainter(qimg)
    try:
        _paint_scale_bar(painter, qimg.width(), qimg.height(), um_per_px, draw_line=draw_line)
    finally:
        painter.end()

    return qimage_to_numpy(qimg)


def draw_scale_bar(img, um_per_px: float):
    """Draw a scale bar on ``img`` using GPU acceleration when available."""

    if um_per_px <= 0:
        return img if isinstance(img, np.ndarray) else img.download()

    logger.debug(
        "draw_scale_bar using %s path",
        "CUDA" if _HAS_CUDA and isinstance(img, cv2.cuda_GpuMat) else "CPU",
    )

    def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if _HAS_CUDA and isinstance(img, cv2.cuda_GpuMat):
        arr = img.download()
        arr = _ensure_rgb(arr)
        return _draw_scale_bar_cpu(arr, um_per_px)

    if isinstance(img, np.ndarray):
        arr = _ensure_rgb(img)
        return _draw_scale_bar_cpu(arr, um_per_px)

    raise TypeError("Unsupported image type for draw_scale_bar")
