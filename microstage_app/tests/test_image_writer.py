import numpy as np
from microstage_app.io.storage import ImageWriter
import tifffile
from PIL import Image
import json


def test_save_single_custom_dir_and_name(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "custom"
    writer.save_single(img, directory=str(out_dir), filename="foo")
    assert (out_dir / "foo.bmp").exists()


def test_save_single_autonumber(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "auto"
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    assert not (out_dir / "foo.bmp").exists()
    assert (out_dir / "foo_1.bmp").exists()
    assert (out_dir / "foo_2.bmp").exists()


def test_save_png(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "png"
    writer.save_single(img, directory=str(out_dir), filename="foo", fmt="png")
    assert (out_dir / "foo.png").exists()


def test_save_with_metadata(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)

    # TIFF metadata roundtrip
    tiff_dir = tmp_path / "tiff"
    meta = {"foo": "bar"}
    writer.save_single(img, directory=str(tiff_dir), filename="foo", fmt="tif", metadata=meta)
    with tifffile.TiffFile(tiff_dir / "foo.tif") as tif:
        desc = tif.pages[0].tags["ImageDescription"].value
        data = json.loads(desc)
    assert data["foo"] == "bar"

    # PNG metadata roundtrip
    png_dir = tmp_path / "png"
    writer.save_single(img, directory=str(png_dir), filename="foo", fmt="png", metadata=meta)
    with Image.open(png_dir / "foo.png") as im:
        assert im.info["foo"] == "bar"

    # JPEG EXIF metadata roundtrip
    jpg_dir = tmp_path / "jpg"
    exif_meta = {272: "camera", 42037: "lens"}
    writer.save_single(img, directory=str(jpg_dir), filename="foo", fmt="jpg", metadata=exif_meta)
    with Image.open(jpg_dir / "foo.jpg") as im:
        exif = im.getexif()
        assert exif[272] == "camera"
        assert exif[42037] == "lens"
