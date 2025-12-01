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
    path: str,
    wavelengths_nm: Sequence[float],
    intensities: Sequence[float] | None,
    metadata: Mapping[str, object],
    *,
    raw_counts: Sequence[float] | None = None,
    include_processed: bool = True,
    include_raw: bool | None = None,
) -> None:
    """Save a single spectrum to CSV with metadata as comments.

    Parameters
    ----------
    path:
        Destination path.
    wavelengths_nm:
        X-axis values in nanometres.
    intensities:
        Processed spectral intensities. Required when ``include_processed`` is
        true.
    metadata:
        Arbitrary metadata stored as header comments.
    raw_counts:
        Optional raw detector counts aligned to ``wavelengths_nm``.
    include_processed:
        Whether to include the processed spectrum.
    include_raw:
        Whether to include the raw counts. When ``None`` this defaults to true
        if ``raw_counts`` is provided.
    """

    include_raw = bool(raw_counts) if include_raw is None else include_raw
    if not include_processed and not include_raw:
        raise ValueError("At least one of processed or raw spectra must be included")
    data_columns = [np.asarray(wavelengths_nm, dtype=float)]
    headers = ["wavelength_nm"]
    if include_processed:
        if intensities is None:
            raise ValueError("Processed intensities are required when include_processed is True")
        data_columns.append(np.asarray(intensities, dtype=float))
        headers.append("intensity")
    if include_raw:
        if raw_counts is None:
            raise ValueError("Raw counts must be supplied when include_raw is True")
        data_columns.append(np.asarray(raw_counts, dtype=float))
        headers.append("raw_counts")
    lines = [f"# {key}: {value}" for key, value in metadata.items()]
    header = "\n".join(lines + [",".join(headers)])
    np.savetxt(
        path,
        np.column_stack(data_columns),
        delimiter=",",
        header=header,
        comments="",
    )


def save_spectrum_hdf5(
    path: str,
    wavelengths_nm: Sequence[float],
    intensities: Sequence[float] | None,
    metadata: Mapping[str, object],
    *,
    raw_counts: Sequence[float] | None = None,
    include_processed: bool = True,
    include_raw: bool | None = None,
) -> None:
    """Save a single spectrum to an HDF5 file."""

    include_raw = bool(raw_counts) if include_raw is None else include_raw
    if not include_processed and not include_raw:
        raise ValueError("At least one of processed or raw spectra must be included")
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("spectrum")
        grp.create_dataset("wavelength_nm", data=np.asarray(wavelengths_nm, dtype=float))
        if include_processed:
            if intensities is None:
                raise ValueError("Processed intensities are required when include_processed is True")
            grp.create_dataset("intensity", data=np.asarray(intensities, dtype=float))
        if include_raw:
            if raw_counts is None:
                raise ValueError("Raw counts must be supplied when include_raw is True")
            grp.create_dataset("raw_counts", data=np.asarray(raw_counts, dtype=float))
        _attach_metadata_hdf5(grp, metadata)


def save_spectrum_jcamp(
    path: str,
    wavelengths_nm: Sequence[float],
    intensities: Sequence[float] | None,
    metadata: Mapping[str, object],
    *,
    raw_counts: Sequence[float] | None = None,
    include_processed: bool = True,
    include_raw: bool | None = None,
) -> None:
    """Save a spectrum to JCAMP-DX format.

    The export writes an ``XYDATA`` block for the processed spectrum and, when
    provided, a ``RAW_COUNTS`` block that mirrors the wavelength axis.
    """

    include_raw = bool(raw_counts) if include_raw is None else include_raw
    if not include_processed and not include_raw:
        raise ValueError("At least one of processed or raw spectra must be included")
    lines = [
        "##TITLE= Spectrum Export",
        "##JCAMP-DX= 5.00",
        "##DATA TYPE= GENERIC SPECTRUM",
        "##XUNITS= NM",
        "##YUNITS= COUNTS",
        f"##NPOINTS= {len(wavelengths_nm)}",
        f"##FIRSTX= {float(np.asarray(wavelengths_nm)[0])}",
        f"##LASTX= {float(np.asarray(wavelengths_nm)[-1])}",
    ]
    for key, value in metadata.items():
        lines.append(f"##${key}= {value}")
    def _write_xy_block(title: str, values: Sequence[float]) -> None:
        lines.append(title)
        for x, y in zip(wavelengths_nm, values):
            lines.append(f"{float(x):.6f} {float(y):.6f}")
    if include_processed:
        if intensities is None:
            raise ValueError("Processed intensities are required when include_processed is True")
        _write_xy_block("##XYDATA= (X Y)", intensities)
    if include_raw:
        if raw_counts is None:
            raise ValueError("Raw counts must be supplied when include_raw is True")
        _write_xy_block("##RAW_COUNTS= (X Y)", raw_counts)
    lines.append("##END=")
    Path(path).write_text("\n".join(lines))


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

