import sys, types
from pathlib import Path
import pytest

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a minimal cv2 stub so analysis package imports succeed
cv2_stub = types.SimpleNamespace(
    RETR_EXTERNAL=0,
    CHAIN_APPROX_SIMPLE=0,
    findContours=lambda *args, **kwargs: ([], None),
    moments=lambda *args, **kwargs: {"m00": 0, "m10": 0, "m01": 0},
)
sys.modules.setdefault("cv2", cv2_stub)

from microstage_app.analysis.lenses import Lens
from microstage_app.control.profiles import Profiles


def test_default_lens_calibration(tmp_path):
    """Calibrating a default lens persists via Profiles."""
    profile_path = tmp_path / "profiles.yaml"
    orig_path = Profiles.PATH
    Profiles.PATH = str(profile_path)
    if profile_path.exists():
        profile_path.unlink()

    profiles = Profiles.load_or_create()
    lens = Lens("10x", 0.75)
    profiles.set(f"measurement.lenses.{lens.name}", lens.um_per_px)
    profiles.save()

    profiles2 = Profiles.load_or_create()
    stored = profiles2.get(
        f"measurement.lenses.{lens.name}", None, expected_type=float
    )
    assert stored == pytest.approx(lens.um_per_px)

    Profiles.PATH = orig_path
