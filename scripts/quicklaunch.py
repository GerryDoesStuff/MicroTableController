"""Quick-launch entry point for MicroStage App.

This helper resolves the repository root, locates the preferred interpreter,
prepares the embedded runtime when necessary, and spawns the GUI using
``pythonw`` so that no console window is shown to end users.
"""
from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

PACKAGE_MARKER = Path("microstage_app") / "__init__.py"


class RepoRootNotFound(RuntimeError):
    """Raised when the repository root cannot be located."""


def _has_package(path: Path) -> bool:
    return (path / PACKAGE_MARKER).is_file()


def resolve_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root that contains ``microstage_app``.

    The search first inspects the provided starting directory (defaults to the
    current working directory). If the package is not found there, the search
    iterates over immediate child directories matching ``MicroTableController*``
    in priority order. This mirrors the layout used in packaged distributions
    where the project may live one directory deeper than the launcher.
    """

    start_dir = Path.cwd() if start is None else start
    if _has_package(start_dir):
        return start_dir

    try:
        children = list(start_dir.iterdir())
    except PermissionError:
        children = []

    for child in children:
        if not child.is_dir():
            continue
        name = child.name
        if fnmatch.fnmatch(name, "MicroTableController*") or _has_package(child):
            if _has_package(child):
                return child

    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, script_dir.parent):
        if _has_package(candidate):
            return candidate

    raise RepoRootNotFound(
        "Could not locate the repository root containing microstage_app/__init__.py"
    )


def detect_interpreter(repo_root: Path) -> Tuple[Path, str]:
    """Return the preferred interpreter and a tag describing its source."""

    embedded_pythonw = repo_root / "python" / "pythonw.exe"
    if embedded_pythonw.is_file():
        return embedded_pythonw, "embedded"

    venv_pythonw = repo_root / ".venv" / "Scripts" / "pythonw.exe"
    if venv_pythonw.is_file():
        return venv_pythonw, "venv"

    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        return venv_python, "venv"

    python_exe = repo_root / "python" / "python.exe"
    if python_exe.is_file():
        return python_exe, "embedded"

    sys_executable = Path(sys.executable)
    if sys_executable.is_file():
        return sys_executable, "system"

    raise FileNotFoundError("No suitable Python interpreter could be located.")


def ensure_embedded_ready(repo_root: Path) -> None:
    ensure_script = repo_root / "scripts" / "ensure_embedded_python_ready.cmd"
    if not ensure_script.is_file():
        return

    if os.name != "nt":
        # Embedded runtimes are only supported on Windows; nothing to do.
        return

    subprocess.run(str(ensure_script), check=True, cwd=repo_root, shell=True)


def launch() -> None:
    repo_root = resolve_repo_root()
    interpreter, source = detect_interpreter(repo_root)

    if source == "embedded":
        ensure_embedded_ready(repo_root)

    cmd = [str(interpreter), "-m", "microstage_app"]

    try:
        subprocess.Popen(cmd, cwd=repo_root)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Failed to start interpreter {interpreter!s}") from exc


if __name__ == "__main__":
    launch()
