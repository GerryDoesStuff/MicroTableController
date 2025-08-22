import os, datetime
import tifffile

class ImageWriter:
    def __init__(self, base_dir='runs'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.base_dir, ts)
        os.makedirs(self.run_dir, exist_ok=True)

    def save_single(self, img_rgb):
        path = os.path.join(self.run_dir, 'capture.tif')
        self._save_tiff(path, img_rgb)

    def save_tile(self, img_rgb, row, col):
        path = os.path.join(self.run_dir, f'tile_r{row:04d}_c{col:04d}.tif')
        self._save_tiff(path, img_rgb)

    def _save_tiff(self, path, img_rgb):
        tifffile.imwrite(path, img_rgb, photometric='rgb')
