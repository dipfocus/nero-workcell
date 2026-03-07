#!/usr/bin/env python3
# coding=utf-8
"""
YOLODetector integration tests using a real YOLO model and a real Intel RealSense device.

Requirements:
- `pyrealsense2` must be installed.
- `ultralytics` must be installed.
- At least one RealSense camera must be connected and powered on.
- Put a real `cup` in view for the detection assertions.

Optional environment variables:
- `REALSENSE_SERIAL`: force a specific RealSense device.
- `YOLO_MODEL_PATH`: local model path or model name resolvable by ultralytics.
- `YOLO_CONF_THRESHOLD`: detector confidence threshold. Default: `0.25`.
"""

import importlib.util
import logging
import os
import time
import unittest

import numpy as np

from nero_workcell.core.realsense_camera import RealSenseCamera
from nero_workcell.core.yolo_detector import YOLODetector


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestYOLODetector")


class TestYOLODetectorIntegration(unittest.TestCase):
    """Integration tests that exercise a real YOLO detector on live RealSense frames."""

    TARGET_CLASS = "cup"
    MAX_ATTEMPTS = 10

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("ultralytics") is None:
            raise ModuleNotFoundError("ultralytics is not installed")

        cls.requested_serial = os.environ.get("REALSENSE_SERIAL")
        cls.model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
        cls.target_class = cls.TARGET_CLASS
        cls.conf_threshold = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))
        cls.max_attempts = cls.MAX_ATTEMPTS

        discovered_serials = RealSenseCamera.discover_serial_numbers()
        if cls.requested_serial:
            if cls.requested_serial not in discovered_serials:
                raise RuntimeError(
                    f"Requested RealSense camera is not connected: {cls.requested_serial}"
                )
            cls.serial_number = cls.requested_serial
        else:
            if not discovered_serials:
                raise RuntimeError("No RealSense camera connected")
            cls.serial_number = discovered_serials[0]

        logger.info("Using RealSense serial for YOLO integration tests: %s", cls.serial_number)
        logger.info("Using YOLO model: %s", cls.model_path)
        logger.info("Target class: %s", cls.target_class)

        cls.camera = None
        try:
            cls.camera = RealSenseCamera.setup(serial_number=cls.serial_number)
            cls.detector = YOLODetector(
                target_class=cls.target_class,
                model_path=cls.model_path,
                conf_threshold=cls.conf_threshold,
            )
            cls.detector.set_intrinsics(
                fx=cls.camera.fx,
                fy=cls.camera.fy,
                cx=cls.camera.cx,
                cy=cls.camera.cy,
            )
        except Exception:
            if cls.camera is not None and cls.camera.is_opened:
                cls.camera.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        camera = getattr(cls, "camera", None)
        if camera is not None and camera.is_opened:
            camera.stop()

    def _read_valid_frame(self, retries: int = 10):
        for _ in range(retries):
            frame = self.camera.read_frame()
            color = frame["color"]
            depth = frame["depth"]
            if color is not None and depth is not None:
                return frame
            time.sleep(0.1)

        self.fail("Failed to read a valid color/depth frame from the RealSense camera")

    def _wait_for_target_detection(self):
        for _ in range(self.max_attempts):
            frame = self._read_valid_frame(retries=3)
            detected_objects = self.detector.detect_objects(frame["color"], frame["depth"])
            if detected_objects:
                return frame, detected_objects
            time.sleep(0.1)

        self.fail(
            f"No '{self.target_class}' target detected in {self.max_attempts} live frames. "
            "Place a real target in view and rerun the test."
        )

    def test_01_setup_opens_camera_and_loads_live_model(self):
        self.assertTrue(self.camera.is_opened)
        self.assertEqual(self.camera.serial_number, self.serial_number)
        self.assertEqual(self.detector.target_class, self.target_class)
        self.assertGreater(self.detector.fx, 0.0)
        self.assertGreater(self.detector.fy, 0.0)
        self.assertGreater(self.camera.depth_scale, 0.0)
        self.assertIn(
            self.target_class,
            self.detector.model.names.values(),
            msg=f"Target class '{self.target_class}' is not present in the loaded YOLO labels",
        )

    def test_02_detect_objects_runs_on_live_frame(self):
        frame = self._read_valid_frame()
        color = frame["color"]
        depth = frame["depth"]

        detected_objects = self.detector.detect_objects(color, depth)

        self.assertIsInstance(detected_objects, list)
        for detected in detected_objects:
            x1, y1, x2, y2 = detected.bbox
            center_x, center_y = detected.center

            self.assertEqual(detected.name, self.target_class)
            self.assertGreaterEqual(detected.conf, self.conf_threshold)
            self.assertEqual(detected.frame, "camera")
            self.assertTrue(np.isfinite(detected.position).all())
            self.assertGreater(detected.position[2], 0.0)
            self.assertEqual(center_x, (x1 + x2) // 2)
            self.assertEqual(center_y, (y1 + y2) // 2)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)
            self.assertLessEqual(x2, color.shape[1])
            self.assertLessEqual(y2, color.shape[0])

    def test_03_pick_best_target_uses_highest_confidence_live_detection(self):
        _, detected_objects = self._wait_for_target_detection()

        best_target = self.detector.pick_best_target(detected_objects)

        self.assertIsNotNone(best_target)
        self.assertAlmostEqual(
            best_target.conf,
            max(float(obj.conf) for obj in detected_objects),
        )
        self.assertEqual(best_target.name, self.target_class)
        self.assertEqual(best_target.frame, "camera")

    def test_04_detect_object_returns_a_live_target_when_present(self):
        frame, _ = self._wait_for_target_detection()

        best_target = self.detector.detect_object(frame["color"], frame["depth"])

        self.assertIsNotNone(best_target)
        self.assertEqual(best_target.name, self.target_class)
        self.assertGreaterEqual(best_target.conf, self.conf_threshold)
        self.assertEqual(best_target.frame, "camera")
        self.assertTrue(np.isfinite(best_target.position).all())
        self.assertGreater(best_target.position[2], 0.0)


if __name__ == "__main__":
    unittest.main()
