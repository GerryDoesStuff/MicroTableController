import yaml, os, copy
from ..utils.log import log

DEFAULT_ILLUMINATION_LIGHTS = [
    {'name': f'Light {i}', 'host': '', 'enabled': False, 'brightness': 0}
    for i in range(1, 6)
]

DEFAULTS = {
    'version': 11,
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
        'color_depth': 8,
        'binning': 1,
        'resolution_index': 1,
        'usb_speed': 5,
    },
    'scan_presets': {
        'raster': {
            'x1_mm': 0.0,
            'y1_mm': 0.0,
            'x2_mm': 4.0,
            'y2_mm': 0.0,
            'x3_mm': 0.0,
            'y3_mm': 4.0,
            'x4_mm': 4.0,
            'y4_mm': 4.0,
            'rows': 5,
            'cols': 5,
            'mode': 'rectangle',
        }
    },
    'measurement': {
        'lenses': {
            '5x': {'um_per_px': 1.0, 'calibrations': {}},
            '10x': {'um_per_px': 1.0, 'calibrations': {}},
            '20x': {'um_per_px': 1.0, 'calibrations': {}},
            '50x': {'um_per_px': 1.0, 'calibrations': {}},
        }
    },
    # persistent capture settings
    'capture': {
        'dir': '',
        'name': 'capture',
        'auto_prefix': False,
        'auto_number': False,
        'format': 'png',
    },
    # jog UI persistence
    'jog': {
        'step': {'x': 0.1, 'y': 0.1, 'z': 0.1},
        'feed': {'x': 50.0, 'y': 50.0, 'z': 50.0},
        'abs': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    },
    # autofocus and focus stack settings
    'autofocus': {
        'range_mm': 0.5,
        'coarse_step_mm': 0.01,
        'fine_step_mm': 0.002,
    },
    'focus_stack': {
        'range_mm': 0.5,
        'step_mm': 0.01,
    },
    'ui': {
        'dark_mode': False,
        'auto_connect_on_start': True,
    },
    'spectroscopy': {
        'geometry': '',
        'window_state': '',
        'compact': False,
        'splitter_state': '',
        'data_dir': '',
        'default_device': '',
        'last_mode': 'Absorbance',
        'last_params': {},
        'integration_ms': 10.0,
        'averages': 1,
        'smoothing': 5,
        'subtract_dark': False,
    },
    'illumination': {
        'dimmer': {
            'host': '',
            'on': False,
            'brightness': 0,
        },
        'lights': copy.deepcopy(DEFAULT_ILLUMINATION_LIGHTS),
    },
}


class Profiles:
    PATH = os.path.abspath('profiles.yaml')
    VERSION = DEFAULTS['version']
    MAX_LIGHTS = len(DEFAULT_ILLUMINATION_LIGHTS)

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

    @staticmethod
    def illumination_light_defaults():
        return copy.deepcopy(DEFAULT_ILLUMINATION_LIGHTS)

    @classmethod
    def sanitize_illumination_lights(cls, lights):
        """Return a sanitized list of illumination light configs.

        Ensures a list of length ``MAX_LIGHTS`` containing dictionaries with
        ``name`` (str), ``host`` (str), ``enabled`` (bool), and ``brightness``
        (int 0-100).  Extra entries are discarded and missing/invalid values
        are replaced with defaults.
        """

        defaults = cls.illumination_light_defaults()
        sanitized = []
        changed = False

        entries = lights if isinstance(lights, list) else []
        if not isinstance(lights, list):
            changed = True

        for idx in range(cls.MAX_LIGHTS):
            base = defaults[idx]
            entry = entries[idx] if idx < len(entries) and isinstance(entries[idx], dict) else {}
            if not isinstance(entry, dict):
                changed = True
                entry = {}

            name = entry.get('name', base['name'])
            if isinstance(name, str):
                name = name.strip()
            else:
                name = base['name']
                changed = True

            host = entry.get('host', base['host'])
            if isinstance(host, str):
                host = host.strip()
            else:
                host = base['host']
                changed = True

            enabled = entry.get('enabled', base['enabled'])
            if not isinstance(enabled, bool):
                enabled = bool(enabled) if isinstance(enabled, (int, float)) else base['enabled']
                changed = True

            brightness = entry.get('brightness', base['brightness'])
            if not isinstance(brightness, (int, float)):
                brightness = base['brightness']
                changed = True
            brightness = max(0, min(100, int(brightness)))
            if brightness != entry.get('brightness'):
                changed = True

            sanitized.append({
                'name': name,
                'host': host,
                'enabled': enabled,
                'brightness': brightness,
            })

        if len(entries) > cls.MAX_LIGHTS:
            changed = True

        return sanitized, changed

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
            # migrate old single pixel_size into lenses dict if present
            meas = data.get('measurement', {})
            if isinstance(meas, dict) and 'pixel_size' in meas:
                meas.setdefault('lenses', {})
                meas['lenses']['10x'] = meas.pop('pixel_size')
                changed = True
            # migrate legacy lens entries (floats or flat dicts)
            lenses = meas.get('lenses', {}) if isinstance(meas, dict) else {}
            if isinstance(lenses, dict):
                for lname, cfg in list(lenses.items()):
                    if isinstance(cfg, (int, float)):
                        lenses[lname] = {
                            'um_per_px': float(cfg),
                            'calibrations': {},
                        }
                        changed = True
                    elif isinstance(cfg, dict):
                        um = cfg.get('um_per_px')
                        cal = cfg.get('calibrations') if isinstance(cfg.get('calibrations'), dict) else {}
                        extras = {
                            k: v for k, v in cfg.items() if k not in ('um_per_px', 'calibrations')
                            and isinstance(v, (int, float))
                        }
                        if um is None:
                            um = next(iter(extras.values()), 1.0)
                            changed = True
                        if extras:
                            cal.update(extras)
                            changed = True
                        lenses[lname] = {
                            'um_per_px': float(um),
                            'calibrations': cal,
                        }
            illum = data.get('illumination')
            if not isinstance(illum, dict):
                data['illumination'] = copy.deepcopy(DEFAULTS['illumination'])
                illum = data['illumination']
                changed = True

            lights_raw = illum.get('lights') if isinstance(illum, dict) else None
            lights, lights_changed = cls.sanitize_illumination_lights(lights_raw)

            dim_cfg = illum.get('dimmer') if isinstance(illum, dict) else {}
            if isinstance(dim_cfg, dict) and lights:
                first = lights[0]
                migrated = False
                host = dim_cfg.get('host')
                if isinstance(host, str) and not first.get('host'):
                    first['host'] = host.strip()
                    migrated = True
                on = dim_cfg.get('on')
                if isinstance(on, bool):
                    first['enabled'] = on
                    migrated = True
                bright = dim_cfg.get('brightness')
                if isinstance(bright, (int, float)):
                    first['brightness'] = max(0, min(100, int(bright)))
                    migrated = True
                if migrated:
                    lights[0] = first
                    changed = True

            if lights_changed:
                changed = True
            illum['lights'] = lights
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
