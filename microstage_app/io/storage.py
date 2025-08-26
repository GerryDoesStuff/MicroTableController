import os, datetime
import tifffile

class ImageWriter:
    def __init__(self, base_dir='runs'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.base_dir, ts)
        os.makedirs(self.run_dir, exist_ok=True)

    def save_single(self, img_rgb, directory=None, filename="capture", auto_number=False):
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
        """

        directory = directory or self.run_dir
        os.makedirs(directory, exist_ok=True)
        base = os.path.join(directory, f"{filename}.tif")
        path = base
        if auto_number and os.path.exists(path):
            n = 1
            while True:
                path = os.path.join(directory, f"{filename}_{n}.tif")
                if not os.path.exists(path):
                    break
                n += 1
        self._save_tiff(path, img_rgb)

    def save_tile(self, img_rgb, row, col):
        path = os.path.join(self.run_dir, f'tile_r{row:04d}_c{col:04d}.tif')
        self._save_tiff(path, img_rgb)

    def _save_tiff(self, path, img_rgb):
        tifffile.imwrite(path, img_rgb, photometric='rgb')
