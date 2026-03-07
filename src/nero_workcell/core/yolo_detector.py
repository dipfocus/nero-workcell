#!/usr/bin/env python3
# coding=utf-8

import logging
from typing import List, Optional

import numpy as np

from nero_workcell.core.target_object import TargetObject

logger = logging.getLogger(__name__)


class YOLODetector:
    """Depth-aware YOLO detector that returns camera-frame TargetObject items."""

    def __init__(
        self,
        target_class: str,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        depth_window_radius: int = 5,
    ):
        if not isinstance(target_class, str) or not target_class.strip():
            raise ValueError("target_class is required and cannot be empty")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_class = target_class.strip()
        self.depth_window_radius = depth_window_radius

        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        logger.info("Loading YOLO model: %s", model_path)
        from ultralytics import YOLO

        return YOLO(model_path)

    def set_intrinsics(self, *, fx: float, fy: float, cx: float, cy: float):
        """Store camera intrinsics required for 2D to 3D projection."""
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def _ensure_intrinsics(self):
        if self.fx <= 0 or self.fy <= 0:
            raise RuntimeError("Camera intrinsics are not initialized")

    def _estimate_depth(self, depth: np.ndarray, center_x: int, center_y: int) -> float:
        h, w = depth.shape
        radius = self.depth_window_radius
        region = depth[
            max(0, center_y - radius):min(h, center_y + radius),
            max(0, center_x - radius):min(w, center_x + radius),
        ]
        valid = region[region > 0]
        return float(np.median(valid)) if valid.size > 0 else 0.0

    def detect_objects(
        self,
        color: np.ndarray,
        depth: np.ndarray,
    ) -> List[TargetObject]:
        """
        Run YOLO inference and return valid camera-frame detections.

        Args:
            color: BGR image for YOLO inference.
            depth: Depth map aligned with `color`, in meters.
        """
        if color is None or depth is None:
            missing_frames = [
                name for name, frame in (("color", color), ("depth", depth)) if frame is None
            ]
            logger.warning(
                "Skipping YOLO detection: missing frame(s): %s",
                ", ".join(missing_frames),
            )
            return []

        self._ensure_intrinsics()

        results = self.model(color, verbose=False)
        detected_objects: List[TargetObject] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                if conf < self.conf_threshold:
                    logger.debug(
                        "Skipping %s: conf %.2f < %.2f",
                        cls_name,
                        conf,
                        self.conf_threshold,
                    )
                    continue
                if cls_name != self.target_class:
                    logger.debug("Skipping %s: not target %s", cls_name, self.target_class)
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                depth_value = self._estimate_depth(depth, center_x, center_y)
                if depth_value <= 0:
                    logger.warning("Skipping %s: invalid depth %.4f", cls_name, depth_value)
                    continue

                position = np.array(
                    [
                        (center_x - self.cx) * depth_value / self.fx,
                        (center_y - self.cy) * depth_value / self.fy,
                        depth_value,
                    ],
                    dtype=float,
                )
                detected_objects.append(
                    TargetObject(
                        name=cls_name,
                        class_id=cls_id,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        position=position,
                        conf=conf,
                        frame="camera",
                    )
                )

        logger.debug(
            "YOLODetector.detect_objects: %d camera objects detected",
            len(detected_objects),
        )
        return detected_objects

    def pick_best_target(
        self,
        detected_targets: List[TargetObject],
    ) -> Optional[TargetObject]:
        """Pick the highest-confidence detection for the configured target class."""
        candidates = [
            obj for obj in detected_targets if obj.name == self.target_class
        ]

        if not candidates:
            return None
        return max(candidates, key=lambda obj: float(obj.conf))

    def detect_object(
        self,
        color: np.ndarray,
        depth: np.ndarray,
    ) -> Optional[TargetObject]:
        """Return the highest-confidence camera-frame target for one frame."""
        detected_targets = self.detect_objects(color, depth)
        return self.pick_best_target(detected_targets)
