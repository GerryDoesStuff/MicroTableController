from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class PlaneModel:
    kind: str = "plane"  # 'plane' | 'quadratic'
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((3,)))

    def fit(self, pts: List[Tuple[float, float, float]]):
        P = np.array(pts, dtype=float)
        x, y, z = P[:,0], P[:,1], P[:,2]
        if self.kind == "plane":
            A = np.c_[np.ones(len(x)), x, y]
            self.coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
        elif self.kind == "quadratic":
            A = np.c_[np.ones(len(x)), x, y, x*x, x*y, y*y]
            self.coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
        else:
            raise ValueError(self.kind)

    def predict(self, x, y) -> float:
        if self.kind == "plane":
            a,b,c = self.coeffs
            return a + b*x + c*y
        else:
            a,b,c,d,e,f = self.coeffs
            return a + b*x + c*y + d*x*x + e*x*y + f*y*y

@dataclass
class Area:
    name: str
    polygon: List[Tuple[float, float]]
    model: PlaneModel
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
