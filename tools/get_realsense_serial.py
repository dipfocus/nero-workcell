#!/usr/bin/env python3
"""Print connected Intel RealSense camera serial numbers."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

import pyrealsense2 as rs


def list_serial_numbers() -> List[str]:
    """Return serial numbers for all connected RealSense devices."""
    ctx = rs.context()
    devices = ctx.query_devices()
    serials: List[str] = []

    for dev in devices:
        serials.append(dev.get_info(rs.camera_info.serial_number))
    return serials


def main() -> int:
    # Parse CLI options.
    parser = argparse.ArgumentParser(
        description="Get connected Intel RealSense serial numbers."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print serial numbers as JSON array.",
    )
    args = parser.parse_args()

    # Query all connected RealSense devices once.
    serials = list_serial_numbers()
    if not serials:
        # Use non-zero exit code so scripts can detect "no device".
        print("No RealSense device found.", file=sys.stderr)
        return 1

    # Support plain-text and JSON output for shell usage.
    if args.json:
        print(json.dumps(serials))
    else:
        for serial in serials:
            print(serial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
