"""Simple CLI tool to probe available spectrometers."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

# Ensure repository root is on sys.path when running directly from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microstage_app.spectroscopy.devices import (
    SpectrometerDescriptor,
    SpectrometerManager,
)


logger = logging.getLogger(__name__)


def _choose_descriptor(
    devices: Iterable[SpectrometerDescriptor],
    *,
    serial_number: str | None = None,
    path: str | None = None,
) -> Optional[SpectrometerDescriptor]:
    for descriptor in devices:
        if serial_number and descriptor.serial_number != serial_number:
            continue
        if path and descriptor.path != path:
            continue
        return descriptor
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check spectrometer connectivity")
    parser.add_argument(
        "--serial",
        dest="serial_number",
        help="Serial number of the spectrometer to target",
    )
    parser.add_argument(
        "--path",
        dest="path",
        help="Device path of the spectrometer to target",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture a spectrum after connecting to verify responses",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    manager = SpectrometerManager(auto_start=False)

    exit_code = 0
    try:
        devices = manager.refresh()
        if not devices:
            logger.error("No spectrometer devices detected")
            return 1

        descriptor = _choose_descriptor(
            devices, serial_number=args.serial_number, path=args.path
        )
        if descriptor is None:
            descriptor = devices[0]
            logger.info(
                "Requested device not found; falling back to first detected: %s",
                descriptor.label(),
            )
        else:
            logger.info("Selected spectrometer: %s", descriptor.label())

        device = manager.connect(descriptor)
        if device is None or not device.is_connected():
            logger.error("Failed to connect to spectrometer %s", descriptor.label())
            return 2

        wavelengths = device.get_wavelengths()
        logger.info("Model       : %s", descriptor.model)
        logger.info("Serial      : %s", descriptor.serial_number)
        logger.info("Path        : %s", descriptor.path)
        logger.info("Vendor      : %s", descriptor.vendor)
        logger.info("Wavelengths : %d", len(wavelengths))

        if args.capture:
            intensities = device.capture()
            sample_count = min(5, len(intensities))
            sample_pairs = list(zip(wavelengths, intensities))[:sample_count]
            logger.info("Capture     : captured spectrum")
            logger.info(
                "Wavelength range: %.2f - %.2f nm",
                min(wavelengths),
                max(wavelengths),
            )
            logger.info(
                "First %d wavelength/intensity pairs (nm, counts): %s",
                sample_count,
                sample_pairs,
            )
        else:
            logger.info("Capture     : skipped (use --capture to measure)")

    except Exception:
        logger.exception("Unexpected error while checking spectrometer")
        exit_code = 3
    finally:
        try:
            manager.disconnect()
        finally:
            manager.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
