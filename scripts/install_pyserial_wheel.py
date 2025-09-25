"""Install the bundled pyserial wheel into the embedded interpreter."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    wheel_path = script_dir / "wheels" / "pyserial-3.5-py2.py3-none-any.whl"
    target = repo_root / "python" / "Lib" / "site-packages"

    if not wheel_path.exists():
        raise FileNotFoundError(f"Bundled wheel not found: {wheel_path}")

    target.mkdir(parents=True, exist_ok=True)

    legacy = target / "serial"
    if legacy.exists():
        shutil.rmtree(legacy)
    for dist_info in target.glob("pyserial-*.dist-info"):
        shutil.rmtree(dist_info, ignore_errors=True)

    with zipfile.ZipFile(wheel_path) as archive:
        archive.extractall(target)


if __name__ == "__main__":
    main()
