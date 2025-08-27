import yaml
from microstage_app.control.profiles import Profiles, DEFAULTS


def test_profiles_migration(monkeypatch, tmp_path):
    old = {'camera': {'gain': 2.0}}
    pfile = tmp_path / 'profiles.yaml'
    pfile.write_text(yaml.safe_dump(old))
    monkeypatch.setattr(Profiles, 'PATH', str(pfile))
    p = Profiles.load_or_create()
    assert p.data['version'] == Profiles.VERSION
    assert p.data['camera']['gain'] == 2.0
    assert p.data['camera']['exposure_ms'] == DEFAULTS['camera']['exposure_ms']
    saved = yaml.safe_load(pfile.read_text())
    assert saved['version'] == Profiles.VERSION
    assert 'capture' in saved
