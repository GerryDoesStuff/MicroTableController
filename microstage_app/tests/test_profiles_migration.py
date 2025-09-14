import yaml
from microstage_app.control.profiles import Profiles, DEFAULTS


def test_profiles_migration(monkeypatch, tmp_path):
    old = {'camera': {'gain': 2.0}, 'measurement': {'lenses': {'10x': 1.23}}}
    pfile = tmp_path / 'profiles.yaml'
    pfile.write_text(yaml.safe_dump(old))
    monkeypatch.setattr(Profiles, 'PATH', str(pfile))
    p = Profiles.load_or_create()
    assert p.data['version'] == Profiles.VERSION
    assert p.data['camera']['gain'] == 2.0
    assert p.data['camera']['exposure_ms'] == DEFAULTS['camera']['exposure_ms']
    assert p.data['measurement']['lenses']['10x']['um_per_px'] == 1.23
    assert p.data['measurement']['lenses']['10x']['calibrations'] == {}
    saved = yaml.safe_load(pfile.read_text())
    assert saved['version'] == Profiles.VERSION
    assert 'capture' in saved
    assert saved['measurement']['lenses']['10x']['um_per_px'] == 1.23
    assert saved['measurement']['lenses']['10x']['calibrations'] == {}
