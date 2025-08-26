from pathlib import Path
import sys

import numpy as np
import tifffile

# Ensure the repository root is on sys.path so ``microstage_app`` is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from microstage_app.io.storage import ImageWriter


def test_save_to_custom_directory(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    writer.save_single(img, directory=str(out_dir), filename="foo")
    assert (out_dir / "foo.bmf").exists()


def test_filename_without_auto_numbering_overwrites(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img1 = np.zeros((2, 2, 3), dtype=np.uint8)
    img2 = np.ones((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "non_auto"
    writer.save_single(img1, directory=str(out_dir), filename="foo", auto_number=False)
    writer.save_single(img2, directory=str(out_dir), filename="foo", auto_number=False)
    assert (out_dir / "foo.bmf").exists()
    assert not (out_dir / "foo_1.bmf").exists()
    saved = tifffile.imread(out_dir / "foo.bmf")
    assert (saved == img2).all()


def test_filename_generation_with_auto_numbering(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "auto"
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    assert (out_dir / "foo.bmf").exists()
    assert (out_dir / "foo_1.bmf").exists()
    assert (out_dir / "foo_2.bmf").exists()


def test_creates_missing_directory(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "missing" / "subdir"
    writer.save_single(img, directory=str(out_dir), filename="foo")
    assert (out_dir / "foo.bmf").exists()


def test_save_with_explicit_format(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "fmt"
    writer.save_single(img, directory=str(out_dir), filename="foo", fmt="tif")
    assert (out_dir / "foo.tif").exists()
