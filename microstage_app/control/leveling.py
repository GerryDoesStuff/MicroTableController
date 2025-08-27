from __future__ import annotations

from enum import Enum
from typing import Sequence, Tuple, List

try:
    from .autofocus import AutoFocus, FocusMetric
except Exception:  # pragma: no cover - autofocus dependencies might be missing
    AutoFocus = None  # type: ignore
    FocusMetric = None  # type: ignore

from .focus_planes import SurfaceModel, SurfaceKind


class LevelingMode(str, Enum):
    """Available surface fitting modes for leveling."""

    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"


def three_point_level(
    stage,
    camera,
    points: Sequence[Tuple[float, float]],
    mode: LevelingMode = LevelingMode.LINEAR,
) -> SurfaceModel:
    """Fit a focus surface from measurements at multiple XY points.

    Parameters
    ----------
    stage : Stage-like object
        Object providing ``move_absolute``, ``wait_for_moves`` and
        ``get_position`` methods.
    camera : camera-like object
        Used with :class:`AutoFocus` when available.
    points : Sequence[Tuple[float, float]]
        XY coordinates in millimetres to probe.
    mode : LevelingMode
        Surface fitting model to use.

    Returns
    -------
    SurfaceModel
        Fitted surface model for the measured points.

    Raises
    ------
    ValueError
        If insufficient points are supplied for the requested ``mode``.
    """

    required = {
        LevelingMode.LINEAR: 3,
        LevelingMode.QUADRATIC: 6,
        LevelingMode.CUBIC: 10,
    }
    n = len(points)
    if n < required[mode]:
        raise ValueError(
            f"{mode.value} leveling requires at least {required[mode]} points, got {n}"
        )

    samples: List[Tuple[float, float, float]] = []
    for x, y in points:
        stage.move_absolute(x=x, y=y)
        stage.wait_for_moves()

        if AutoFocus and camera is not None:  # pragma: no branch
            try:
                af = AutoFocus(stage, camera)
                af.coarse_to_fine(metric=FocusMetric.LAPLACIAN)
                stage.wait_for_moves()
            except Exception:
                pass

        pos = stage.get_position()
        if pos is None or len(pos) < 3:
            raise RuntimeError("stage did not return a valid position")
        z = float(pos[2])
        samples.append((x, y, z))

    model = SurfaceModel(kind=SurfaceKind(mode.value))
    model.fit(samples)
    return model
