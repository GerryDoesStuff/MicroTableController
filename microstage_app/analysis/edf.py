from __future__ import annotations
from typing import List
import numpy as np

try:
    import microstage_edf_gpu as _edf_impl
except Exception as exc:  # pragma: no cover - handled at runtime
    _edf_impl = None
    _edf_import_error = exc
else:
    _edf_import_error = None


def fuse_stack(images: List[np.ndarray], use_cuda: bool = True) -> np.ndarray:
    """Fuse a stack of images into a single sharp image.

    Parameters
    ----------
    images : list of numpy.ndarray
        Sequence of images forming the focus stack. Each image may be either
        grayscale or BGR. All images should share the same dimensions.
    use_cuda : bool, optional
        If ``True`` (default), attempt to use CUDA acceleration when available.

    Returns
    -------
    numpy.ndarray
        The fused image as an unsigned 8-bit array in BGR order.
    """
    if _edf_impl is None:  # pragma: no cover - import failure at runtime
        raise ImportError("microstage_edf_gpu could not be imported") from _edf_import_error

    fused, _ = _edf_impl.fuse_stack(
        images,
        color_mode="lab_l_only",
        focus="tenengrad",
        levels=5,
        sobel_ksize=3,
        linear_fusion=False,
        use_sat_weight=False,
        sat_gamma=1.0,
        clahe_l=False,
        clahe_clip=2.0,
        clahe_tiles=8,
        use_gpu=use_cuda,
    )
    return fused
