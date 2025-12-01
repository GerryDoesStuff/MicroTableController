import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))


@pytest.fixture
def wavelengths():
    return np.linspace(400.0, 800.0, 16)


@pytest.fixture
def raw_spectrum(wavelengths):
    return np.linspace(10.0, 100.0, wavelengths.size)


@pytest.fixture
def dark_spectrum(wavelengths):
    return np.linspace(1.0, 5.0, wavelengths.size)


@pytest.fixture
def reference_spectrum(wavelengths):
    return np.linspace(80.0, 120.0, wavelengths.size)


@pytest.fixture
def response_curve(wavelengths):
    return np.linspace(0.5, 1.5, wavelengths.size)
