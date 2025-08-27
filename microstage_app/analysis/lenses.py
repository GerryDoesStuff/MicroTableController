from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Lens:
    """Calibration data for a microscope lens.

    Parameters
    ----------
    name:
        Display name of the lens.
    um_per_px:
        Current calibration in microns-per-pixel for the active
        camera resolution.
    calibrations:
        Optional mapping of resolution strings (``"{width}x{height}"``)
        to microns-per-pixel values.  When switching camera
        resolutions the appropriate calibration can be looked up
        or scaled based on this mapping.
    """

    name: str
    um_per_px: float
    calibrations: Dict[str, float] = field(default_factory=dict)


__all__ = ["Lens"]
