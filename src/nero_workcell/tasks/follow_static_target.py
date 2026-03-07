#!/usr/bin/env python3
# coding=utf-8
"""
Static-target following task (visual servoing).
Uses RealSense and YOLO to control the robot arm to approach a locked target object.

Usage:
    python -m nero_workcell.tasks.follow_static_target --target cup --conf 0.5
"""

import logging
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from nero_workcell.core import NeroController, PIDController, RealSenseCamera, YOLODetector
from nero_workcell.core.target_object import TargetObject
from nero_workcell.utils.common import load_eye_in_hand_calibration, transform_to_base

logger = logging.getLogger(__name__)


class ObjectFollower:
    def __init__(self,
                 target_class: str,
                 robot_channel: str = "can0",
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.5,
                 target_distance: float = 0.3,
                 camera_serial_number: Optional[str] = None):
        
        self.target_class = target_class
        self.conf_threshold = conf_threshold
        self.target_distance = target_distance
        self.camera_serial_number = camera_serial_number
        
        # Initialize camera.
        self.width = 640
        self.height = 480
        self.camera = None
        
        # Initialize detector.
        self.detector = YOLODetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            target_class=target_class,
        )

        self.pid_x = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)  # Units: m/s
        self.pid_y = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        self.pid_z = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        self.locked_target: Optional[TargetObject] = None
        
        # Robot controller instance.
        self.robot = NeroController(robot_channel, "nero")
        self.is_running = False

    def reset_follow_state(self):
        """Clear controller state so a newly locked target starts from a clean PID state."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

    def lock_target(self, target: TargetObject):
        """Freeze the current base-frame detection for static-target following."""
        if target.frame != "base":
            raise ValueError(f"Expected base-frame target, got frame='{target.frame}'")

        self.locked_target = TargetObject(
            name=target.name,
            class_id=target.class_id,
            bbox=target.bbox,
            center=target.center,
            position=np.array(target.position, dtype=float).copy(),
            conf=target.conf,
            frame=target.frame,
        )
        self.reset_follow_state()
        logger.info("[follow] Locked static target at base position %s", self.locked_target.position)

    def clear_locked_target(self):
        """Stop following the frozen target and reset PID accumulators."""
        self.locked_target = None
        self.reset_follow_state()

    def get_follow_target(
        self,
        detected_target: Optional[TargetObject],
        *,
        follow_enabled: bool,
    ) -> Optional[TargetObject]:
        """Prefer the frozen target once static following has started."""
        if follow_enabled and self.locked_target is not None:
            return self.locked_target
        return detected_target

    def detect_object(self):
        """
        Capture one frame and detect the target in base coordinates.

        Returns:
            dict | None: A payload with color image and base-frame target object.
            Returns None when frame read fails, target is missing, or pose read fails.
        """
        # Read image and depth frame.
        frame = self.camera.read_frame()
        color, depth = frame["color"], frame["depth"]
        if color is None or depth is None:
            logger.warning("[detect] Frame read failed")
            return None
        # Run detector and keep only the best camera-frame target.
        best_target = self.detector.detect_object(color, depth)
        if best_target is None:
            logger.warning("[detect] No target detected")
            return {"color": color, "target": None}

        # Get current flange pose.
        T_gripper2base = self.robot.get_current_pose()
        if T_gripper2base is None:
            logger.warning("[detect] Failed to get flange pose: no data")
            return {"color": color, "target": None}

        T_cam2base = T_gripper2base @ self.T_cam2gripper
        detected_target = transform_to_base([best_target], T_cam2base)[0]

        logger.info("[detect] Target detected: %s", detected_target)

        return {
            "color": color,
            "target": detected_target,
        }

    def follow_target(self, target: TargetObject) -> bool:
        """
        Follow a target point in base coordinates (no grasping).

        Constraints:
        - target.position must be in the base frame.
        - Desired TCP is self.target_distance meters above the target point.
        """
        if target.frame != "base":
            logger.warning("[follow] Target frame is not base, skipping frame: %s", target.frame)
            return False

        tcp_pose = self.robot.get_tcp_pose()
        if tcp_pose is None:
            logger.warning("[follow] Failed to get TCP pose")
            return False

        tcp_pos = np.array(tcp_pose[:3], dtype=float)
        target_pos = np.array(target.position, dtype=float)

        desired_pos = target_pos.copy()
        desired_pos[2] += self.target_distance

        error = desired_pos - tcp_pos
        err_x, err_y, err_z = error.tolist()

        deadband = 0.005  # 5mm
        if abs(err_x) < deadband and abs(err_y) < deadband and abs(err_z) < deadband:
            return True

        # Reuse PID gains from 2D follower; scale error to millimeters.
        err_scale = 1000.0
        cmd_x = float(self.pid_x.compute(err_x * err_scale))
        cmd_y = float(self.pid_y.compute(err_y * err_scale))
        cmd_z = float(self.pid_z.compute(err_z * err_scale))

        max_step_z = 0.03
        cmd_z = float(np.clip(cmd_z, -max_step_z, max_step_z))

        self.robot.move_relative(dx=cmd_x, dy=cmd_y, dz=cmd_z)
        return False

    def run(self):
        # 1. Load eye-in-hand calibration.
        calib_file = Path("configs/eye_in_hand_calibration.json")
        self.T_cam2gripper = load_eye_in_hand_calibration(str(calib_file))

        # 2. Start camera and load intrinsics.
        self.camera = RealSenseCamera.setup(
            width=self.width,
            height=self.height,
            fps=30,
            serial_number=self.camera_serial_number,
        )
        self.detector.set_intrinsics(
            fx=self.camera.fx,
            fy=self.camera.fy,
            cx=self.camera.cx,
            cy=self.camera.cy,
        )

        # 3. Connect robot.
        if not self.robot.connect():
            logger.error("Robot connection failed, task aborted")
            return
        self.is_running = True
        
        logger.info(f"Starting follow task, target: {self.target_class}")
        logger.info("Press 's' to lock the current target and follow it")
        logger.info("Press 'c' to clear the locked target")
        logger.info("Press 'q' to exit")

        center_x, center_y = self.width // 2, self.height // 2
        follow_enabled = False

        try:
            while self.is_running:
                result = self.detect_object()
                if result is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                display_img = result["color"].copy()
                detected_target = result["target"]
                active_target = self.get_follow_target(
                    detected_target,
                    follow_enabled=follow_enabled,
                )

                if active_target is None:
                    self.reset_follow_state()
                    status = "SEARCHING"
                    status_color = (0, 255, 255)
                else:
                    if follow_enabled:
                        # For static targets, keep following the frozen base-frame target.
                        reached_target = self.follow_target(active_target)
                        if reached_target:
                            status = "LOCKED TARGET REACHED"
                            status_color = (0, 255, 0)
                        else:
                            status = "FOLLOWING LOCKED TARGET"
                            status_color = (0, 255, 0)
                    elif detected_target is not None:
                        status = "DETECTED: press 's' to lock"
                        status_color = (0, 255, 0)
                    else:
                        status = "SEARCHING"
                        status_color = (0, 255, 255)

                # Draw image-center crosshair.
                cv2.line(display_img, (center_x-20, center_y), (center_x+20, center_y), (255, 0, 0), 1)
                cv2.line(display_img, (center_x, center_y-20), (center_x, center_y+20), (255, 0, 0), 1)
                cv2.putText(display_img, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                cv2.imshow("Follow Static Target", display_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c'):
                    follow_enabled = False
                    self.clear_locked_target()
                    logger.info("Locked target cleared by key 'c'")
                if key == ord('s') and detected_target is not None:
                    follow_enabled = True
                    self.lock_target(detected_target)
                    logger.info("Follow triggered by key 's'")
                    locked_target = self.locked_target
                    if locked_target is not None:
                        self.follow_target(locked_target)

        finally:
            if self.camera is not None:
                self.camera.stop()
            cv2.destroyAllWindows()
            logger.info("Task finished")


def main():
    parser = argparse.ArgumentParser(description="Nero Workcell - Follow Static Target Task")
    parser.add_argument("--target", type=str, default="cup", help="Target object class name (e.g., cup, bottle)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--camera-serial", type=str, default=None, help="RealSense camera serial number")
    
    args = parser.parse_args()
    
    # Configure logging.
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )

    follower = ObjectFollower(
        target_class=args.target,
        robot_channel="can0",
        model_path=args.model,
        conf_threshold=args.conf,
        camera_serial_number=args.camera_serial,
    )
    
    follower.run()


if __name__ == "__main__":
    main()
