from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Lens:
    """Calibration data for a microscope lens."""

    name: str
    um_per_px: float


__all__ = ["Lens"]
