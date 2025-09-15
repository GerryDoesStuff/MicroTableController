import math
import subprocess
import logging

import numpy as np
from PySide6 import QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageFont
import cv2


# Scaling factors for the scale bar drawing used across the application
VERT_SCALE = 2  # line thickness multiplier
TEXT_SCALE = 4  # font size multiplier


_scale_font_cache = None
_font_error_reported = False
logger = logging.getLogger(__name__)


def _show_font_error_dialog(message: str) -> None:
    """Display a user-visible error when fonts cannot be loaded."""

    global _font_error_reported

    if _font_error_reported:
        return

    qapp = QtGui.QGuiApplication.instance()
    if qapp is None:
        return

    try:
        QtWidgets.QMessageBox.warning(None, "Scale Bar Font Error", message)
    except Exception as exc:  # pragma: no cover - GUI availability dependent
        logger.debug("Unable to display font error dialog: %s", exc)
    finally:
        _font_error_reported = True


def _load_default_scale_font(reason: str, font_size: int) -> ImageFont.ImageFont:
    """Load a bundled or Pillow default font and log the fallback reason."""

    for candidate in ("DejaVuSans.ttf",):
        try:
            font = ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
        logger.warning(
            "%s; using bundled font '%s' for scale bar text.",
            reason,
            candidate,
        )
        return font

    try:
        font = ImageFont.load_default()
    except Exception as exc:  # pragma: no cover - Pillow default rarely fails
        logger.error("%s; unable to load Pillow default font: %s", reason, exc)
        _show_font_error_dialog(
            "MicroStage was unable to load any font for the scale bar overlay. "
            "The text labels will not be displayed."
        )
        raise RuntimeError("No usable font available for scale bar") from exc

    logger.warning(
        "%s; using Pillow default bitmap font for scale bar text.",
        reason,
    )
    return font


def _load_scale_font() -> ImageFont.ImageFont:
    """Return a PIL font matching the application's QFont.

    The font file is resolved once using ``fc-match`` and the resulting font is
    cached for reuse.  When the lookup fails, a bundled or Pillow default font
    is used so the scale bar overlay can still render text.  The size is scaled
    by :data:`TEXT_SCALE` to mirror :func:`MeasureView.drawForeground`.
    """

    global _scale_font_cache

    if _scale_font_cache is not None:
        return _scale_font_cache

    qapp = QtGui.QGuiApplication.instance()
    if qapp is None:
        raise RuntimeError("QGuiApplication instance required to load font")

    qfont = qapp.font()
    ps = qfont.pointSizeF()
    if ps > 0:
        base_size = ps
    else:
        base_size = qfont.pixelSize()

    font_size = int(round(base_size * TEXT_SCALE))

    family = qfont.family()
    font: ImageFont.ImageFont

    try:
        res = subprocess.run(
            ["fc-match", "-f", "%{file}\n", family],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        font = _load_default_scale_font(
            "System font lookup utility 'fc-match' is not available: "
            f"{exc}",
            font_size,
        )
        logger.debug("fc-match not found while resolving '%s': %s", family, exc)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        if stderr:
            logger.debug(
                "fc-match error while resolving '%s': %s", family, stderr
            )
        reason = (
            f"System font lookup failed for '{family}' (exit code {exc.returncode})"
        )
        if stderr:
            reason = f"{reason}: {stderr}"
        font = _load_default_scale_font(reason, font_size)
    else:
        font_path = res.stdout.strip()
        if not font_path:
            font = _load_default_scale_font(
                f"System font lookup returned an empty path for '{family}'",
                font_size,
            )
        else:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except OSError as exc:
                font = _load_default_scale_font(
                    f"Failed to load system font '{font_path}': {exc}",
                    font_size,
                )
                logger.debug("Unable to load font '%s': %s", font_path, exc)

    _scale_font_cache = font
    return _scale_font_cache


def _has_cuda() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


_HAS_CUDA = _has_cuda()
logger.info("Scale-bar drawing using %s", "CUDA" if _HAS_CUDA else "CPU")


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

    try:
        font = _load_scale_font()
    except RuntimeError as exc:
        logger.error("Scale bar font unavailable; skipping text overlay: %s", exc)
        font = None
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Unexpected error while loading scale bar font; skipping text overlay")
        font = None

    if font is not None:
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

    logger.debug(
        "draw_scale_bar using %s path",
        "CUDA" if _HAS_CUDA and isinstance(img, cv2.cuda_GpuMat) else "CPU",
    )

    if _HAS_CUDA and isinstance(img, cv2.cuda_GpuMat):
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
            if _HAS_CUDA:
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
