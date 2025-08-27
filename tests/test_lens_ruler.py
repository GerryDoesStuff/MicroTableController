import sys, types
from pathlib import Path
import pytest

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a minimal cv2 stub so the measure module can be imported
cv2_stub = types.SimpleNamespace(
    RETR_EXTERNAL=0,
    CHAIN_APPROX_SIMPLE=0,
    findContours=lambda *args, **kwargs: ([], None),
    moments=lambda *args, **kwargs: {"m00": 0, "m10": 0, "m01": 0},
)
sys.modules.setdefault("cv2", cv2_stub)

from microstage_app.analysis import Lens
from microstage_app.analysis.measure import measure_distance
from microstage_app.control.profiles import Profiles


def test_lens_calibration_persist(tmp_path):
    """Lens calibration values should persist via Profiles storage."""
    profile_path = tmp_path / "profiles.yaml"
    orig_path = Profiles.PATH
    Profiles.PATH = str(profile_path)
    if profile_path.exists():
        profile_path.unlink()

    profiles = Profiles.load_or_create()
    lens = Lens("40x", 0.25)
    res_key = "100x100"
    profiles.set(
        f"measurement.lenses.{lens.name}.{res_key}", lens.um_per_px
    )
    profiles.save()

    profiles2 = Profiles.load_or_create()
    stored = profiles2.get(
        f"measurement.lenses.{lens.name}.{res_key}", None, expected_type=float
    )
    assert stored == pytest.approx(lens.um_per_px)

    Profiles.PATH = orig_path


def test_micron_per_pixel_from_calibration_lines():
    """Compute µm-per-pixel from drawn calibration lines."""
    # two segments: length 5px and 3px -> total 8px
    lines = [((0, 0), (3, 4)), ((3, 4), (6, 4))]
    total_pixels = sum(measure_distance(p1, p2, 1.0) for p1, p2 in lines)
    assert total_pixels == pytest.approx(8.0)

    actual_um = 80.0
    um_per_px = actual_um / total_pixels
    assert um_per_px == pytest.approx(10.0)


def test_ruler_pixels_to_microns():
    """Ruler overlay converts pixel segments to microns using current lens."""
    lens = Lens("10x", 0.5)
    segments = [((0, 0), (0, 6)), ((0, 6), (8, 6))]  # 6px vertical + 8px horizontal = 14px
    microns = sum(measure_distance(p1, p2, lens.um_per_px) for p1, p2 in segments)
    assert microns == pytest.approx(14 * lens.um_per_px)
