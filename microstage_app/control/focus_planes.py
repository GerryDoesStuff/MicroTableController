from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np

class SurfaceKind(str, Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"


@dataclass
class SurfaceModel:
    """Polynomial surface model.

    At least 3/6/10 points are required to fit linear, quadratic or cubic
    surfaces respectively.
    """

    kind: SurfaceKind = SurfaceKind.LINEAR
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0,)))

    def fit(self, pts: List[Tuple[float, float, float]]):
        P = np.array(pts, dtype=float)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        if self.kind is SurfaceKind.LINEAR:
            A = np.c_[np.ones(len(x)), x, y]
        elif self.kind is SurfaceKind.QUADRATIC:
            A = np.c_[
                np.ones(len(x)),
                x,
                y,
                x ** 2,
                x * y,
                y ** 2,
            ]
        elif self.kind is SurfaceKind.CUBIC:
            A = np.c_[
                np.ones(len(x)),
                x,
                y,
                x ** 2,
                x * y,
                y ** 2,
                x ** 3,
                (x ** 2) * y,
                x * (y ** 2),
                y ** 3,
            ]
        else:
            raise ValueError(self.kind)
        self.coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)

    def predict(self, x, y) -> float:
        if self.kind is SurfaceKind.LINEAR:
            a, b, c = self.coeffs
            return a + b * x + c * y
        elif self.kind is SurfaceKind.QUADRATIC:
            a, b, c, d, e, f = self.coeffs
            return a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2
        elif self.kind is SurfaceKind.CUBIC:
            a, b, c, d, e, f, g, h, i, j = self.coeffs
            return (
                a
                + b * x
                + c * y
                + d * x ** 2
                + e * x * y
                + f * y ** 2
                + g * x ** 3
                + h * (x ** 2) * y
                + i * x * (y ** 2)
                + j * y ** 3
            )
        else:
            raise ValueError(self.kind)

@dataclass
class Area:
    name: str
    polygon: List[Tuple[float, float]]
    model: SurfaceModel
    priority: int = 0

    def contains(self, x, y) -> bool:
        n = len(self.polygon)
        inside = False
        px, py = x, y
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i+1) % n]
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1):
                inside = not inside
        return inside

@dataclass
class FocusPlaneManager:
    areas: List[Area] = field(default_factory=list)

    def select_area(self, x, y) -> Optional[Area]:
        candidates = [a for a in self.areas if a.contains(x, y)]
        if not candidates: return None
        candidates.sort(key=lambda a: a.priority, reverse=True)
        return candidates[0]

    def z_offset(self, x, y, z_ref=0.0) -> float:
        a = self.select_area(x, y)
        if a is None: return 0.0
        return float(a.model.predict(x, y) - z_ref)

    def add_area(self, area: Area):
        self.areas.append(area)
