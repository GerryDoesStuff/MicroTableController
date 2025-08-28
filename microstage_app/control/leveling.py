from __future__ import annotations

from enum import Enum
from threading import Event
from typing import Iterable, Sequence, Tuple, List

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
    stop_event: Event | None = None,
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
    stop_event : threading.Event, optional
        Event used to signal cancellation. If set, probing stops and a
        ``RuntimeError`` is raised.

    Returns
    -------
    SurfaceModel
        Fitted surface model for the measured points.

    Raises
    ------
    ValueError
        If insufficient points are supplied for the requested ``mode``.
    RuntimeError
        If ``stop_event`` is set during execution or the stage fails to
        return a valid position.
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
        if stop_event and stop_event.is_set():
            raise RuntimeError("operation cancelled")
        stage.move_absolute(x=x, y=y)
        stage.wait_for_moves()
        if stop_event and stop_event.is_set():
            raise RuntimeError("operation cancelled")

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


def _grid_coords(
    rect: Tuple[float, float, float, float], rows: int, cols: int
) -> Iterable[Tuple[float, float]]:
    """Generate XY coordinates for a regular grid within ``rect``.

    Parameters
    ----------
    rect : Tuple[float, float, float, float]
        ``(x1, y1, x2, y2)`` rectangle bounds in millimetres.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    """

    x1, y1, x2, y2 = rect
    dx = (x2 - x1) / (cols - 1) if cols > 1 else 0.0
    dy = (y2 - y1) / (rows - 1) if rows > 1 else 0.0
    for r in range(rows):
        y = y1 + r * dy
        for c in range(cols):
            x = x1 + c * dx
            yield x, y


def _probe_point(
    stage,
    camera,
    x: float,
    y: float,
    autofocus: bool,
    stop_event: Event | None = None,
) -> Tuple[float, float, float]:
    """Move to ``(x, y)`` and record the current Z position.

    Parameters
    ----------
    stage : Stage-like object
        Provides movement and position queries.
    camera : camera-like object
        Used with :class:`AutoFocus` when ``autofocus`` is True.
    x, y : float
        XY coordinates in millimetres to probe.
    autofocus : bool
        Whether to run autofocus at the point or wait for manual focus.
    stop_event : threading.Event, optional
        Event used to signal cancellation. When set, a ``RuntimeError``
        is raised.

    Returns
    -------
    Tuple[float, float, float]
        The probed ``(x, y, z)`` position.

    Raises
    ------
    RuntimeError
        If the stage fails to report a position or ``stop_event`` is set.
    """

    if stop_event and stop_event.is_set():
        raise RuntimeError("operation cancelled")

    stage.move_absolute(x=x, y=y)
    stage.wait_for_moves()

    if stop_event and stop_event.is_set():
        raise RuntimeError("operation cancelled")

    if autofocus:
        if AutoFocus and camera is not None:  # pragma: no branch
            try:
                af = AutoFocus(stage, camera)
                af.coarse_to_fine(metric=FocusMetric.LAPLACIAN)
                stage.wait_for_moves()
            except Exception:
                pass
    else:
        input("Focus at the current point and press Enter to continue...")

    if stop_event and stop_event.is_set():
        raise RuntimeError("operation cancelled")

    pos = stage.get_position()
    if pos is None or len(pos) < 3:
        raise RuntimeError("stage did not return a valid position")
    return x, y, float(pos[2])


def grid_level(
    stage,
    camera,
    rect: Tuple[float, float, float, float],
    rows: int,
    cols: int,
    mode: LevelingMode = LevelingMode.LINEAR,
    autofocus: bool = True,
    stop_event: Event | None = None,
) -> SurfaceModel:
    """Fit a surface model by probing a grid of points.

    Parameters
    ----------
    stage : Stage-like object
        Provides ``move_absolute``, ``wait_for_moves`` and ``get_position``.
    camera : camera-like object
        Used with :class:`AutoFocus` when ``autofocus`` is True.
    rect : Tuple[float, float, float, float]
        Rectangle bounds ``(x1, y1, x2, y2)`` in millimetres.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    mode : LevelingMode
        Surface fitting model to use.
    autofocus : bool, default True
        If True use autofocus at each node, otherwise wait for user
        confirmation after manual focusing.
    stop_event : threading.Event, optional
        Event used to signal cancellation. If set, a ``RuntimeError`` is
        raised.

    Returns
    -------
    SurfaceModel
        Fitted surface model for the probed grid points.
    
    Raises
    ------
    RuntimeError
        If ``stop_event`` is set during execution or a probe fails to
        return a valid position.
    """

    samples: List[Tuple[float, float, float]] = []
    for x, y in _grid_coords(rect, rows, cols):
        if stop_event and stop_event.is_set():
            raise RuntimeError("operation cancelled")
        samples.append(_probe_point(stage, camera, x, y, autofocus, stop_event))

    model = SurfaceModel(kind=SurfaceKind(mode.value))
    model.fit(samples)
    return model
