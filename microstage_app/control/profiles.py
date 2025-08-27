import yaml, os, copy
from ..utils.log import log

DEFAULTS = {
    'version': 1,
    'stage': {'feed_mm_s': 50.0 / 60.0, 'settle_ms': 30},
    'camera': {
        'exposure_ms': 10.0,
        'auto_exposure': False,
        'gain': 100,
        'brightness': 0,
        'contrast': 0,
        'saturation': 128,
        'hue': 0,
        'gamma': 100,
        'raw': False,
        'binning': 1,
        'resolution_index': 0,
        'speed_level': 0,
        'display_decimation': 1,
    },
    'scan_presets': {
        'raster': {
            'x1_mm': 0.0,
            'y1_mm': 0.0,
            'x2_mm': 4.0,
            'y2_mm': 4.0,
            'rows': 5,
            'cols': 5,
        }
    },
    # persistent capture settings
    'capture': {'dir': '', 'name': 'capture', 'auto_number': False, 'format': 'bmp'},
    # jog UI persistence
    'jog': {
        'step': {'x': 0.1, 'y': 0.1, 'z': 0.1},
        'feed': {'x': 50.0, 'y': 50.0, 'z': 50.0},
        'abs': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    },
}


class Profiles:
    PATH = os.path.abspath('profiles.yaml')
    VERSION = DEFAULTS['version']

    @classmethod
    def load_or_create(cls):
        if not os.path.exists(cls.PATH):
            with open(cls.PATH, 'w') as f:
                yaml.safe_dump(DEFAULTS, f)
        with open(cls.PATH, 'r') as f:
            data = yaml.safe_load(f) or {}
        if cls.migrate(data):
            with open(cls.PATH, 'w') as f:
                yaml.safe_dump(data, f)
        p = Profiles(); p.data = data; return p

    @classmethod
    def migrate(cls, data: dict) -> bool:
        """Upgrade profile data in-place. Returns True if modified."""
        changed = False
        version = data.get('version', 0)
        if version < cls.VERSION:
            def merge(defaults, target):
                nonlocal changed
                for key, val in defaults.items():
                    if key not in target:
                        target[key] = copy.deepcopy(val)
                        changed = True
                    elif isinstance(val, dict) and isinstance(target[key], dict):
                        merge(val, target[key])

            merge(DEFAULTS, data)
            data['version'] = cls.VERSION
            changed = True
        return changed
    def list_profile_names(self): return ['default']
    def get(self, path: str, default=None, *, expected_type=None, min_value=None, max_value=None):
        """Retrieve a value from the profile with basic validation.

        Parameters
        ----------
        path: str
            Dot separated path within the profile data.
        default: any
            Value to return if the path doesn't exist or validation fails.
            The type of ``default`` is also used as the expected type when
            ``expected_type`` is not provided.
        expected_type: type or tuple[type], optional
            Expected python type(s) for the value.
        min_value: float, optional
            Minimum numeric value allowed (inclusive).
        max_value: float, optional
            Maximum numeric value allowed (inclusive).
        """
        cur = self.data
        for key in path.split('.'):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        val = cur
        if expected_type is None and default is not None:
            expected_type = type(default)
        if expected_type is not None and not isinstance(val, expected_type):
            log(f"WARNING: profile '{path}' has invalid type {type(val).__name__}, expected {expected_type}; using default {default!r}")
            return default
        if isinstance(val, (int, float)):
            if min_value is not None and val < min_value:
                log(f"WARNING: profile '{path}' value {val} below minimum {min_value}; using default {default!r}")
                return default
            if max_value is not None and val > max_value:
                log(f"WARNING: profile '{path}' value {val} above maximum {max_value}; using default {default!r}")
                return default
        return val

    def set(self, path: str, value):
        cur = self.data
        keys = path.split('.')
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        cur[keys[-1]] = value

    def save(self):
        with open(self.PATH, 'w') as f:
            yaml.safe_dump(self.data, f)
