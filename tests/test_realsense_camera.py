#!/usr/bin/env python3
# coding=utf-8
"""
RealSenseCamera integration tests using a real Intel RealSense device.

Requirements:
- `pyrealsense2` must be installed.
- At least one RealSense camera must be connected and powered on.
- Optionally set `REALSENSE_SERIAL` to force a specific device.
"""

import logging
import os
import unittest

import numpy as np

from nero_workcell.core.realsense_camera import RealSenseCamera

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestRealSenseCamera")


class TestRealSenseCamera(unittest.TestCase):
    """Integration tests that exercise a real RealSense camera."""

    @classmethod
    def setUpClass(cls):
        cls.requested_serial = os.environ.get("REALSENSE_SERIAL")
        cls.discovered_serials = RealSenseCamera.discover_serial_numbers()
        if cls.requested_serial:
            if cls.requested_serial not in cls.discovered_serials:
                raise RuntimeError(
                    f"Requested RealSense camera is not connected: {cls.requested_serial}"
                )
            cls.serial_number = cls.requested_serial
        else:
            if not cls.discovered_serials:
                raise RuntimeError("No RealSense camera connected")
            cls.serial_number = cls.discovered_serials[0]

        logger.info("Using RealSense serial for integration tests: %s", cls.serial_number)

    def setUp(self):
        self.camera = None

    def tearDown(self):
        if self.camera is not None and self.camera.is_opened:
            self.camera.stop()

    def test_01_discover_serial_numbers(self):
        logger.info("=== Test 01: Discover RealSense Devices ===")
        self.assertGreaterEqual(len(self.discovered_serials), 1)
        self.assertIn(self.serial_number, self.discovered_serials)

    def test_02_setup_opens_camera_and_loads_intrinsics(self):
        logger.info("=== Test 02: Setup Camera and Load Intrinsics ===")
        self.camera = RealSenseCamera.setup(serial_number=self.serial_number)
        camera = self.camera

        self.assertTrue(camera.is_opened)
        self.assertEqual(camera.serial_number, self.serial_number)
        self.assertGreater(camera.fx, 0.0)
        self.assertGreater(camera.fy, 0.0)
        self.assertGreaterEqual(camera.cx, 0.0)
        self.assertGreaterEqual(camera.cy, 0.0)
        self.assertGreater(camera.depth_scale, 0.0)

    def test_03_read_frame_returns_color_and_depth(self):
        logger.info("=== Test 03: Read Frame ===")
        self.camera = RealSenseCamera.setup(serial_number=self.serial_number)
        camera = self.camera

        frame = camera.read_frame()
        color = frame["color"]
        depth = frame["depth"]

        self.assertIsNotNone(color, "Color frame should not be None")
        self.assertIsNotNone(depth, "Depth frame should not be None")
        self.assertEqual(color.shape, (camera.height, camera.width, 3))
        self.assertEqual(depth.shape, (camera.height, camera.width))
        self.assertEqual(color.dtype, np.uint8)
        self.assertEqual(depth.dtype, np.float32)
        self.assertGreater(frame["timestamp"], 0.0)

        valid_depth = depth[depth > 0]
        self.assertGreater(valid_depth.size, 0, "Depth image should contain valid measurements")

    def test_04_stats_increase_after_frame_read(self):
        logger.info("=== Test 04: Stats Update After Frame Read ===")
        self.camera = RealSenseCamera.setup(serial_number=self.serial_number)
        camera = self.camera
        stats_before = camera.get_stats()

        frame = camera.read_frame()
        self.assertIsNotNone(frame["color"])
        self.assertIsNotNone(frame["depth"])

        stats_after = camera.get_stats()
        self.assertEqual(stats_before["frames_captured"] + 1, stats_after["frames_captured"])
        self.assertEqual(stats_before["failed_reads"], stats_after["failed_reads"])


if __name__ == "__main__":
    unittest.main()
