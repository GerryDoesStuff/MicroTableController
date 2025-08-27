import numpy as np

from microstage_app.control.focus_planes import SurfaceModel, SurfaceKind


def test_surface_model_linear():
    pts = [(0, 0, 0), (1, 0, 1), (0, 1, 1)]
    m = SurfaceModel(kind=SurfaceKind.LINEAR)
    m.fit(pts)
    assert np.isclose(m.predict(2, 2), 4)


def test_surface_model_quadratic():
    pts = [
        (0, 0, 1),
        (1, 0, 7),
        (0, 1, 10),
        (1, 1, 21),
        (2, 1, 40),
        (1, 2, 47),
    ]
    m = SurfaceModel(kind=SurfaceKind.QUADRATIC)
    m.fit(pts)
    assert np.isclose(m.predict(3, -1), 31)


def test_surface_model_cubic():
    pts = [
        (0, 0, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 2),
        (-1, 0, -1),
        (0, -1, -1),
        (-1, -1, -2),
        (2, 0, 8),
        (0, 2, 8),
        (2, 2, 16),
    ]
    m = SurfaceModel(kind=SurfaceKind.CUBIC)
    m.fit(pts)
    expected = (-2) ** 3 + 3 ** 3
    assert np.isclose(m.predict(-2, 3), expected)

