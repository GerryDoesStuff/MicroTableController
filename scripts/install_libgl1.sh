#!/usr/bin/env bash
# Install libGL.so.1 dependency for OpenCV on Debian/Ubuntu systems.
set -e
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y libgl1
else
    echo "Please install libGL.so.1 using your distribution's package manager." >&2
    exit 1
fi
