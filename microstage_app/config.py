import os

# Expected identifiers for the Marlin-based stage board.
# `EXPECTED_MACHINE_NAME` is the primary guard that ensures we only talk to
# a board flashed with the custom firmware. `EXPECTED_MACHINE_UUID` is optional
# and helps disambiguate between multiple compatible boards.
EXPECTED_MACHINE_NAME = os.getenv("MICROSTAGE_MACHINE_NAME", "MicroStageController")
EXPECTED_MACHINE_UUID = os.getenv(
    "MICROSTAGE_MACHINE_UUID",
    "a3a4637a-68c4-4340-9fda-847b4fe0d3fc",
)
