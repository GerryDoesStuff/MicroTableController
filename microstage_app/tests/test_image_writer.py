import numpy as np
from microstage_app.io.storage import ImageWriter


def test_save_single_custom_dir_and_name(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "custom"
    writer.save_single(img, directory=str(out_dir), filename="foo")
    assert (out_dir / "foo.bmf").exists()


def test_save_single_autonumber(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "auto"
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    writer.save_single(img, directory=str(out_dir), filename="foo", auto_number=True)
    assert (out_dir / "foo.bmf").exists()
    assert (out_dir / "foo_1.bmf").exists()
    assert (out_dir / "foo_2.bmf").exists()


def test_save_png(tmp_path):
    writer = ImageWriter(base_dir=str(tmp_path / "runs"))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_dir = tmp_path / "png"
    writer.save_single(img, directory=str(out_dir), filename="foo", fmt="png")
    assert (out_dir / "foo.png").exists()
