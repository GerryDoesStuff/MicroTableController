from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import h5py
import numpy as np


DEFAULT_SPECTROSCOPY_DIR = os.getenv(
    "SPECTROSCOPY_DATA_DIR", os.path.join(Path.home(), "MicroStage", "spectra")
)


def default_data_directory() -> str:
    """Return the configured default spectroscopy data directory.

    The directory is defined by the ``SPECTROSCOPY_DATA_DIR`` environment variable
    and falls back to ``~/MicroStage/spectra`` when unset.
    """

    return os.path.abspath(DEFAULT_SPECTROSCOPY_DIR)


def ensure_data_directory(path: str | None) -> str:
    """Create the spectroscopy data directory if needed and return its path."""

    target = os.path.abspath(path or default_data_directory())
    os.makedirs(target, exist_ok=True)
    return target


def _attach_metadata_hdf5(group: h5py.Group, metadata: Mapping[str, object]) -> None:
    for key, value in metadata.items():
        try:
            group.attrs[key] = value
        except TypeError:
            group.attrs[key] = json.dumps(value)


def save_spectrum_csv(
    path: str, wavelengths_nm: Sequence[float], intensities: Sequence[float], metadata: Mapping[str, object]
) -> None:
    """Save a single spectrum to CSV with metadata as comments."""

    lines = [f"# {key}: {value}" for key, value in metadata.items()]
    header = "\n".join(lines + ["wavelength_nm,intensity"])
    np.savetxt(
        path,
        np.column_stack([wavelengths_nm, intensities]),
        delimiter=",",
        header=header,
        comments="",
    )


def save_spectrum_hdf5(
    path: str, wavelengths_nm: Sequence[float], intensities: Sequence[float], metadata: Mapping[str, object]
) -> None:
    """Save a single spectrum to an HDF5 file."""

    with h5py.File(path, "w") as h5:
        grp = h5.create_group("spectrum")
        grp.create_dataset("wavelength_nm", data=np.asarray(wavelengths_nm, dtype=float))
        grp.create_dataset("intensity", data=np.asarray(intensities, dtype=float))
        _attach_metadata_hdf5(grp, metadata)


@dataclass
class HypercubeData:
    wavelengths_nm: np.ndarray
    spectra: np.ndarray  # shape (rows, cols, wavelengths)
    coords_xy_mm: np.ndarray  # shape (rows, cols, 2)
    timestamps_s: np.ndarray  # shape (rows, cols)
    metadata: Dict[str, object]


def save_hypercube_npz(path: str, data: HypercubeData) -> None:
    np.savez_compressed(
        path,
        wavelengths_nm=data.wavelengths_nm,
        spectra=data.spectra,
        coords_xy_mm=data.coords_xy_mm,
        timestamps_s=data.timestamps_s,
        metadata=json.dumps(data.metadata),
    )


def save_hypercube_hdf5(path: str, data: HypercubeData) -> None:
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("hypercube")
        grp.create_dataset("wavelength_nm", data=data.wavelengths_nm)
        grp.create_dataset("spectra", data=data.spectra)
        grp.create_dataset("coords_xy_mm", data=data.coords_xy_mm)
        grp.create_dataset("timestamps_s", data=data.timestamps_s)
        _attach_metadata_hdf5(grp, data.metadata)


def save_time_series_npz(path: str, wavelengths_nm: Sequence[float], spectra: np.ndarray, timestamps_s: np.ndarray, metadata: Mapping[str, object]) -> None:
    np.savez_compressed(
        path,
        wavelengths_nm=np.asarray(wavelengths_nm, dtype=float),
        spectra=np.asarray(spectra, dtype=float),
        timestamps_s=np.asarray(timestamps_s, dtype=float),
        metadata=json.dumps(dict(metadata)),
    )


def save_time_series_hdf5(
    path: str, wavelengths_nm: Sequence[float], spectra: np.ndarray, timestamps_s: np.ndarray, metadata: Mapping[str, object]
) -> None:
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("time_series")
        grp.create_dataset("wavelength_nm", data=np.asarray(wavelengths_nm, dtype=float))
        grp.create_dataset("spectra", data=np.asarray(spectra, dtype=float))
        grp.create_dataset("timestamps_s", data=np.asarray(timestamps_s, dtype=float))
        _attach_metadata_hdf5(grp, metadata)

