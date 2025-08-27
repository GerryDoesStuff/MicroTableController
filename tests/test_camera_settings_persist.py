import os
import sys
from pathlib import Path

import pytest
from PySide6 import QtWidgets

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from microstage_app.ui.main_window import MainWindow
from microstage_app.control.profiles import Profiles


def test_camera_settings_persist(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    profile_path = tmp_path / "profiles.yaml"
    orig_path = Profiles.PATH
    Profiles.PATH = str(profile_path)
    if profile_path.exists():
        profile_path.unlink()

    MainWindow._auto_connect_async = lambda self: None

    w = MainWindow()
    w.exp_spin.setValue(20.0)
    w.autoexp_chk.setChecked(True)
    w.gain_spin.setValue(1.5)
    w.brightness_spin.setValue(40)
    w.contrast_spin.setValue(-10)
    w.saturation_spin.setValue(200)
    w.hue_spin.setValue(5)
    w.gamma_spin.setValue(70)
    w.raw_chk.setChecked(True)
    w.speed_spin.setValue(2)
    w.decim_spin.setValue(3)
    w.close()

    w2 = MainWindow()
    assert w2.exp_spin.value() == 20.0
    assert w2.autoexp_chk.isChecked() is True
    assert w2.gain_spin.value() == pytest.approx(1.5)
    assert w2.brightness_spin.value() == 40
    assert w2.contrast_spin.value() == -10
    assert w2.saturation_spin.value() == 200
    assert w2.hue_spin.value() == 5
    assert w2.gamma_spin.value() == 70
    assert w2.raw_chk.isChecked() is True
    assert w2.speed_spin.value() == 2
    assert w2.decim_spin.value() == 3
    w2.close()
    Profiles.PATH = orig_path
