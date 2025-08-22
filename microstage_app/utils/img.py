import numpy as np
from PySide6 import QtGui

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
