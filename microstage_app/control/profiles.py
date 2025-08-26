import yaml, os

DEFAULTS = {
    'stage': {'feed_mm_s': 20.0 / 60.0, 'settle_ms': 30},
    'camera': {'exposure_ms': 10.0, 'gain': 1.0, 'binning': 1},
    'scan_presets': {
        'raster': {'pitch_x_mm': 1.0, 'pitch_y_mm': 1.0, 'rows': 5, 'cols': 5}
    },
    # persistent capture settings
    'capture': {'dir': '', 'name': 'capture', 'auto_number': False, 'format': 'bmp'},
}

class Profiles:
    PATH = os.path.abspath('profiles.yaml')
    @classmethod
    def load_or_create(cls):
        if not os.path.exists(cls.PATH):
            with open(cls.PATH, 'w') as f:
                yaml.safe_dump(DEFAULTS, f)
        with open(cls.PATH, 'r') as f:
            data = yaml.safe_load(f) or {}
        p = Profiles(); p.data = data; return p
    def list_profile_names(self): return ['default']
    def get(self, path: str, default=None):
        cur = self.data
        for key in path.split('.'):
            if key in cur: cur = cur[key]
            else: return default
        return cur

    def set(self, path: str, value):
        cur = self.data
        keys = path.split('.')
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = value

    def save(self):
        with open(self.PATH, 'w') as f:
            yaml.safe_dump(self.data, f)
