from pathlib import Path
import sys
import json

import numpy as np
import pytest
import tifffile
from PIL import Image

# Ensure the repository root is on sys.path so ``microstage_app`` is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from microstage_app.io.storage import ImageWriter


def test_save_to_custom_directory(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    writer.save_single(img, directory=str(out_dir), filename="foo")
    assert (out_dir / "foo.bmp").exists()


def test_filename_without_auto_numbering_overwrites(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img1 = np.zeros((2, 2, 3), dtype=np.uint8)
    img2 = np.ones((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "non_auto"
    writer.save_single(img1, directory=str(out_dir), filename="foo", auto_number=False)
    writer.save_single(img2, directory=str(out_dir), filename="foo", auto_number=False)
    assert (out_dir / "foo.bmp").exists()
    assert not (out_dir / "foo_1.bmp").exists()
    saved = np.array(Image.open(out_dir / "foo.bmp"))
    assert (saved == img2).all()


def test_filename_generation_with_auto_numbering(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "auto"
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    assert not (out_dir / "foo.bmp").exists()
    assert (out_dir / "foo_1.bmp").exists()
    assert (out_dir / "foo_2.bmp").exists()


def test_creates_missing_directory(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "missing" / "subdir"
    writer.save_single(img, directory=str(out_dir), filename="foo")
    assert (out_dir / "foo.bmp").exists()


def test_save_with_explicit_format(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "fmt"
    writer.save_single(img, directory=str(out_dir), filename="foo", fmt="tif")
    assert (out_dir / "foo.tif").exists()


def _load_png(path):
    with Image.open(path) as img:
        return img.info


def _load_tif(path):
    with tifffile.TiffFile(path) as tif:
        return json.loads(tif.pages[0].tags["ImageDescription"].value)


def _load_jpg(path):
    with Image.open(path) as img:
        exif = img.getexif()
    return {
        "camera": exif.get(271),
        "position": exif.get(270),
        "lens": exif.get(42036),
    }


@pytest.mark.parametrize(
    "fmt, metadata, loader",
    [
        ("png", {"camera": "cam", "position": "pos", "lens": "lens"}, _load_png),
        ("tif", {"camera": "cam", "position": "pos", "lens": "lens"}, _load_tif),
        ("jpg", {271: "cam", 270: "pos", 42036: "lens"}, _load_jpg),
    ],
)
def test_save_with_metadata_roundtrip(tmp_path, fmt, metadata, loader):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    writer.save_single(img, directory=str(tmp_path), filename=f"meta_{fmt}", fmt=fmt, metadata=metadata)
    ext = "tif" if fmt == "tif" else fmt
    meta = loader(tmp_path / f"meta_{fmt}.{ext}")
    assert meta["camera"] == "cam"
    assert meta["position"] == "pos"
    assert meta["lens"] == "lens"
