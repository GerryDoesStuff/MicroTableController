"""Utilities for package-wide resources."""

from __future__ import annotations

import atexit
from contextlib import AbstractContextManager
from importlib import resources
from pathlib import Path
from typing import Optional

__all__ = ["get_dejavu_sans_path"]

_DEJAVU_SANS_RESOURCE = resources.files("microstage_app") / "fonts" / "DejaVuSans.ttf"

_font_ctx: Optional[AbstractContextManager[Path]] = None
_font_path: Optional[Path] = None
_cleanup_registered = False


def _cleanup_font_resource() -> None:
    """Release the temporary resource created by :func:`get_dejavu_sans_path`."""

    global _font_ctx

    if _font_ctx is not None:
        _font_ctx.__exit__(None, None, None)
        _font_ctx = None


def get_dejavu_sans_path() -> Path:
    """Return the filesystem path to the bundled DejaVu Sans font.

    The path is resolved via :mod:`importlib.resources` so the font can be
    located even when this package is imported from a zip archive.  The
    temporary resource provided by :func:`importlib.resources.as_file` is kept
    alive for the lifetime of the process so the returned :class:`~pathlib.Path`
    remains valid for repeated calls.
    """

    global _font_ctx, _font_path, _cleanup_registered

    if _font_path is not None:
        return _font_path

    ctx = resources.as_file(_DEJAVU_SANS_RESOURCE)
    font_path = Path(ctx.__enter__())
    _font_ctx = ctx
    _font_path = font_path

    if not _cleanup_registered:
        atexit.register(_cleanup_font_resource)
        _cleanup_registered = True

    return font_path
