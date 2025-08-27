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


def test_surface_model_linear_coefficients():
    a, b, c = 1.5, -2.0, 0.5
    coords = [(0, 0), (1, 0), (0, 1), (2, -1), (-1, 2)]
    pts = [(x, y, a + b * x + c * y) for x, y in coords]
    m = SurfaceModel(kind=SurfaceKind.LINEAR)
    m.fit(pts)
    assert np.allclose(m.coeffs, [a, b, c])


def test_surface_model_quadratic_coefficients():
    a, b, c, d, e, f = 1, 2, 3, 4, 5, 6
    coords = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    pts = [
        (
            x,
            y,
            a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2,
        )
        for x, y in coords
    ]
    m = SurfaceModel(kind=SurfaceKind.QUADRATIC)
    m.fit(pts)
    assert np.allclose(m.coeffs, [a, b, c, d, e, f])


def test_surface_model_cubic_coefficients():
    a, b, c, d, e, f, g, h, i, j = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    coords = [(x, y) for x in (-1, 0, 1, 2) for y in (-1, 0, 1, 2)]
    pts = [
        (
            x,
            y,
            a
            + b * x
            + c * y
            + d * x ** 2
            + e * x * y
            + f * y ** 2
            + g * x ** 3
            + h * (x ** 2) * y
            + i * x * (y ** 2)
            + j * y ** 3,
        )
        for x, y in coords
    ]
    m = SurfaceModel(kind=SurfaceKind.CUBIC)
    m.fit(pts)
    assert np.allclose(m.coeffs, [a, b, c, d, e, f, g, h, i, j])

