import os, datetime, json
import tifffile
from PIL import Image, PngImagePlugin, ExifTags

EXIF_TAGS_REVERSE = {name.lower(): tag for tag, name in ExifTags.TAGS.items()}
# Common alias for camera manufacturer
EXIF_TAGS_REVERSE.setdefault("camera", EXIF_TAGS_REVERSE.get("make", 271))

class ImageWriter:
    def __init__(self, base_dir='runs'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.base_dir, ts)
        os.makedirs(self.run_dir, exist_ok=True)

    def save_single(
        self,
        img_rgb,
        directory=None,
        filename="capture",
        auto_number=False,
        fmt="bmp",
        metadata=None,
    ):
        """Save a single image.

        Parameters
        ----------
        img_rgb : array-like
            RGB image data.
        directory : str or None
            Destination directory. Defaults to ``self.run_dir``.
        filename : str
            Base filename without extension.
        auto_number : bool
            If ``True``, append ``_n`` to ``filename`` where ``n`` increments
            to avoid overwriting existing files.
        fmt : str
            Image format/extension. ``bmp`` (default). Supported formats are
            ``bmp``, ``tif``, ``png`` and ``jpg``.
        metadata : dict or None
            Optional metadata to embed in the image file when supported.
        """

        directory = directory or self.run_dir
        os.makedirs(directory, exist_ok=True)
        fmt = fmt.lower()
        ext = {
            "bmp": "bmp",
            "tif": "tif",
            "tiff": "tif",
            "png": "png",
            "jpg": "jpg",
            "jpeg": "jpg",
        }.get(fmt, "bmp")

        if auto_number:
            n = 1
            while True:
                path = os.path.join(directory, f"{filename}_{n}.{ext}")
                if not os.path.exists(path):
                    break
                n += 1
        else:
            path = os.path.join(directory, f"{filename}.{ext}")

        if ext == "tif":
            self._save_tiff(path, img_rgb, metadata)
        elif ext == "png":
            self._save_png(path, img_rgb, metadata)
        elif ext == "jpg":
            self._save_jpg(path, img_rgb, metadata)
        elif ext == "bmp":
            self._save_bmp(path, img_rgb, metadata)
        else:
            self._save_bmp(path, img_rgb, metadata)

    def save_tile(self, img_rgb, row, col):
        path = os.path.join(self.run_dir, f'tile_r{row:04d}_c{col:04d}.tif')
        self._save_tiff(path, img_rgb)

    def _save_tiff(self, path, img_rgb, metadata=None):
        """Save image as TIFF with optional metadata."""
        tifffile.imwrite(path, img_rgb, photometric="rgb", metadata=metadata)

    def _save_png(self, path, img_rgb, metadata=None):
        """Save image as PNG, embedding metadata if provided."""
        if metadata:
            pnginfo = PngImagePlugin.PngInfo()
            for key, value in metadata.items():
                pnginfo.add_text(str(key), str(value))
            Image.fromarray(img_rgb).save(path, format="PNG", pnginfo=pnginfo)
        else:
            Image.fromarray(img_rgb).save(path, format="PNG")

    def _save_jpg(self, path, img_rgb, metadata=None):
        """Save image as JPEG, embedding EXIF metadata if provided."""
        if metadata:
            exif = Image.Exif()
            leftover = {}
            for key, value in metadata.items():
                tag = None
                if isinstance(key, int):
                    tag = key
                else:
                    key_str = str(key)
                    if key_str.isdigit():
                        tag = int(key_str)
                    else:
                        tag = EXIF_TAGS_REVERSE.get(key_str.lower())
                if tag is not None:
                    try:
                        exif[int(tag)] = str(value)
                    except Exception:
                        continue
                else:
                    leftover[str(key)] = value
            if leftover:
                json_blob = json.dumps(leftover)
                if 270 not in exif:
                    exif[270] = json_blob
                else:
                    exif[0x9286] = json_blob
            Image.fromarray(img_rgb).save(path, format="JPEG", exif=exif.tobytes())
        else:
            Image.fromarray(img_rgb).save(path, format="JPEG")

    def _save_bmp(self, path, img_rgb, metadata=None):
        """Save image as BMP.

        The BMP format lacks a standard way to embed metadata, so any
        provided metadata is ignored.
        """
        Image.fromarray(img_rgb).save(path, format="BMP")
