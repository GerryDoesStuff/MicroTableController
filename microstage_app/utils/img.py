import math

import numpy as np
from PySide6 import QtGui
from PIL import Image, ImageDraw, ImageFont

# Scaling factors for the scale bar drawing used across the application
VERT_SCALE = 2  # line thickness multiplier
TEXT_SCALE = 4  # font size multiplier

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


def draw_scale_bar(img: np.ndarray, um_per_px: float) -> np.ndarray:
    """Draw a scale bar in-place on an RGB image array.

    Parameters
    ----------
    img:
        ``HxWx3`` RGB image data. Grayscale ``HxW`` arrays will be expanded to
        RGB before drawing.
    um_per_px:
        Microns per pixel for the image. Must be positive.

    Returns
    -------
    np.ndarray
        The image with the scale bar overlay. A new array is returned as the
        underlying conversion through :mod:`PIL` requires a copy.
    """

    if um_per_px <= 0:
        return img

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

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
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = base_font.font_variant(size=font_size)

    bbox = draw.textbbox((0, 0), label, font=font)
    th = bbox[3] - bbox[1]
    draw.text(
        (x0, y0 - (7 * TEXT_SCALE) - th),
        label,
        fill=(255, 255, 255),
        font=font,
    )

    return np.array(pil)
