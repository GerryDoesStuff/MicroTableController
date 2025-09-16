"""Packaging smoke tests ensuring bundled assets ship correctly."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONTS = (
    "microstage_app/fonts/DejaVuSans.ttf",
    "microstage_app/fonts/DejaVuSans.LICENSE",
)


def _missing_members(members: set[str]) -> list[str]:
    """Return a sorted list of packaged font assets that are absent."""

    missing: list[str] = []
    for expected in FONTS:
        if not any(member.endswith(expected) for member in members):
            missing.append(expected)
    return missing


def _run_setup_command(args: list[str], dist_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run a ``setup.py`` command and return the completed process."""

    cmd = [sys.executable, "setup.py", *args, "--dist-dir", str(dist_dir)]
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


@pytest.fixture
def dist_dir(tmp_path):
    """Provide a temporary directory for build artifacts and ensure cleanup."""

    dist_path = tmp_path / "dist"
    dist_path.mkdir()
    try:
        yield dist_path
    finally:
        for path in (PROJECT_ROOT / "build", PROJECT_ROOT / "dist", PROJECT_ROOT / "microstage_app.egg-info"):
            shutil.rmtree(path, ignore_errors=True)


def test_sdist_includes_dejavu_assets(dist_dir: Path) -> None:
    """The generated source distribution should contain the bundled fonts."""

    result = _run_setup_command(["sdist"], dist_dir)
    if result.returncode != 0:
        raise AssertionError(f"sdist build failed: {result.stderr}")

    archives = list(dist_dir.glob("*.tar.gz"))
    assert archives, "sdist build did not produce any archives"

    archive_path = archives[0]
    with tarfile.open(archive_path, "r:gz") as tar:
        names = set(tar.getnames())

    missing = _missing_members(names)
    assert not missing, f"sdist archive missing expected assets: {missing}"


def test_wheel_includes_dejavu_assets(dist_dir: Path) -> None:
    """The built wheel should contain the bundled font resources."""

    result = _run_setup_command(["bdist_wheel"], dist_dir)
    if result.returncode != 0:
        stderr = result.stderr or result.stdout
        if "invalid command 'bdist_wheel'" in stderr:
            pytest.skip("wheel package is not available; skipping wheel packaging check")
        raise AssertionError(f"wheel build failed: {stderr}")

    wheels = list(dist_dir.glob("*.whl"))
    assert wheels, "Wheel build did not produce any files"

    wheel_path = wheels[0]
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())

    missing = _missing_members(names)
    assert not missing, f"Wheel archive missing expected assets: {missing}"
