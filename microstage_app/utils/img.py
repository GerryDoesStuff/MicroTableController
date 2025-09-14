import math
import subprocess

import numpy as np
from PySide6 import QtGui
from PIL import Image, ImageDraw, ImageFont
import cv2

from .log import log

# Scaling factors for the scale bar drawing used across the application
VERT_SCALE = 2  # line thickness multiplier
TEXT_SCALE = 4  # font size multiplier


def _has_cuda() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _draw_scale_bar_cpu(img: np.ndarray, um_per_px: float,
                        *, draw_line: bool = True) -> np.ndarray:
    """CPU implementation of the scale bar drawing."""

    h, w, _ = img.shape

    # Compute a "nice" length that fits within ~20% of the image width
    max_um = 0.2 * w * um_per_px
    exp = math.floor(math.log10(max_um)) if max_um > 0 else 0
    nice_um = 10 ** exp
    for m in (5, 2, 1):
        candidate = m * (10 ** exp)
        if candidate <= max_um:
            nice_um = candidate
            break

    # Scale the length and clamp to image bounds
    length_px = int(round(nice_um / um_per_px))
    max_length = w - 40  # leave 20px margin on each side
    if length_px > max_length:
        length_px = max_length
        nice_um = length_px * um_per_px

    margin = 20
    x0 = int(round(w - margin - length_px))
    y0 = int(round(h - margin))

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    if draw_line:
        draw.line(
            [(x0, y0), (x0 + length_px, y0)],
            fill=(255, 255, 255),
            width=2 * VERT_SCALE,
        )

    label = (
        f"{nice_um/1000:.2f} mm" if nice_um >= 1000 else f"{nice_um:.0f} µm"
    )

    base_font = ImageFont.load_default()
    font_size = base_font.size * TEXT_SCALE
    font = base_font.font_variant(size=font_size)

    qapp = QtGui.QGuiApplication.instance()
    font_path = ""
    if qapp is not None:
        family = qapp.font().family()
        try:
            res = subprocess.run(
                ["fc-match", "-f", "%{file}\n", family],
                check=True,
                capture_output=True,
                text=True,
            )
            font_path = res.stdout.strip()
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            log(
                f"WARNING: failed to load scale bar font {font_path or family}: {e}; using default font"
            )

    bbox = draw.textbbox((0, 0), label, font=font)
    th = bbox[3] - bbox[1]
    draw.text(
        (x0, y0 - (7 * TEXT_SCALE) - th),
        label,
        fill=(255, 255, 255),
        font=font,
    )

    return np.array(pil)

def numpy_to_qimage(img: np.ndarray) -> QtGui.QImage:
    if img.ndim == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        return qimg.copy()
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w, _ = img.shape
        qimg = QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        return qimg.copy()
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")


def draw_scale_bar(img, um_per_px: float):
    """Draw a scale bar on ``img`` using GPU acceleration when available.

    ``img`` may be a :class:`numpy.ndarray` or ``cv2.cuda_GpuMat``. In the GPU
    path, the line is rendered on the device and text is overlaid after
    downloading the frame.
    """

    if um_per_px <= 0:
        return img if isinstance(img, np.ndarray) else img.download()

    has_cuda = _has_cuda()

    if has_cuda and isinstance(img, cv2.cuda_GpuMat):
        w, h = img.size()
        h = int(h)
        w = int(w)

        # Compute geometry
        max_um = 0.2 * w * um_per_px
        exp = math.floor(math.log10(max_um)) if max_um > 0 else 0
        nice_um = 10 ** exp
        for m in (5, 2, 1):
            candidate = m * (10 ** exp)
            if candidate <= max_um:
                nice_um = candidate
                break

        length_px = int(round(nice_um / um_per_px))
        max_length = w - 40
        if length_px > max_length:
            length_px = max_length
            nice_um = length_px * um_per_px

        margin = 20
        x0 = int(round(w - margin - length_px))
        y0 = int(round(h - margin))
        thickness = 2 * VERT_SCALE
        y1 = max(0, y0 - thickness)
        roi = img.rowRange(y1, y0).colRange(x0, x0 + length_px)
        roi.setTo((255, 255, 255))

        arr = img.download()
        return _draw_scale_bar_cpu(arr, um_per_px, draw_line=False)

    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            if has_cuda:
                try:
                    gm = cv2.cuda_GpuMat()
                    gm.upload(img)
                    gm = cv2.cuda.cvtColor(gm, cv2.COLOR_GRAY2RGB)
                    arr = gm.download()
                except Exception:
                    arr = np.repeat(img[:, :, None], 3, axis=2)
            else:
                arr = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 3:
            arr = img
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        return _draw_scale_bar_cpu(arr, um_per_px)

    raise TypeError("Unsupported image type for draw_scale_bar")
