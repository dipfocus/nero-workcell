#!/usr/bin/env python3
# coding=utf-8
"""
Object following task (visual servoing).
Uses RealSense and YOLO to control the robot arm to follow a target object.

Usage:
    python -m nero_workcell.tasks.object_follower --target bottle --conf 0.5
"""

import logging
import argparse

import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

from nero_workcell.core import NeroController, PIDController, RealSenseCamera

logger = logging.getLogger(__name__)


class ObjectFollower:
    def __init__(self, 
                 target_class: str, 
                 robot_ip: str, 
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.5):
        
        self.target_class = target_class
        self.robot_ip = robot_ip
        self.conf_threshold = conf_threshold
        
        # Initialize camera.
        self.width = 640
        self.height = 480
        self.camera = None
        
        # Initialize YOLO.
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Initialize PID controllers for image X/Y error.
        # Gains should be tuned based on real robot dynamics.
        self.pid_x = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)  # Unit: m/s-equivalent command
        self.pid_y = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        
        # Robot controller instance.
        self.robot = NeroController(self.robot_ip)
        self.is_running = False

    def setup_camera(self):
        """Auto-discover and connect to the first available RealSense camera."""
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未找到 RealSense 相机")
        
        serial = devices[0].get_info(rs.camera_info.serial_number)
        logger.info(f"Using camera: {serial}")
        self.camera = RealSenseCamera(
            width=self.width,
            height=self.height,
            fps=30,
            serial_number=serial,
        )
        
        if not self.camera.start():
            raise RuntimeError("相机启动失败")

    def move_robot(self, vx: float, vy: float):
        """
        Move the robot using PID outputs as relative increments.

        Assumed mapping between camera frame and robot frame:
        - Image X error -> robot Y motion
        - Image Y error -> robot X motion

        Adjust axis mapping for your real installation.
        """
        if not self.robot.is_connected():
            return

        # Deadband to suppress tiny jitter.
        if abs(vx) < 0.001 and abs(vy) < 0.001:
            return

        try:
            # Example axis mapping. Tune signs per installation.
            
            scale = 0.5
            dx = -vy * scale
            dy = -vx * scale
            
            # Send relative motion command.
            self.robot.move_relative(dx=dx, dy=dy)
            
        except Exception as e:
            logger.error(f"Motion control failed: {e}")

    def run(self):
        self.setup_camera()
        if not self.robot.connect():
            logger.error("Robot connection failed, task aborted")
            return
        self.is_running = True
        
        logger.info(f"Starting follow task, target: {self.target_class}")
        logger.info("Press 's' to start following")
        logger.info("Press 'q' to exit")

        center_x, center_y = self.width // 2, self.height // 2
        follow_enabled = False

        try:
            while self.is_running:
                # 1. Read one frame.
                frame_data = self.camera.read_frame()
                color_image = frame_data['color']
                if color_image is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('s') and not follow_enabled:
                        follow_enabled = True
                        logger.info("Follow triggered by key 's'")
                    continue

                # 2. Run YOLO detection.
                results = self.model(color_image, verbose=False, conf=self.conf_threshold)
                
                target_box = None
                max_conf = 0

                # 3. Find the best matching target object.
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id]
                        conf = float(box.conf[0])
                        
                        if cls_name == self.target_class and conf > max_conf:
                            max_conf = conf
                            target_box = box.xywh[0].cpu().numpy()  # x_center, y_center, w, h

                # 4. Compute error and control command.
                display_img = color_image.copy()
                
                if target_box is not None:
                    tx, ty, tw, th = target_box
                    
                    # Draw target bounding box.
                    x1, y1 = int(tx - tw/2), int(ty - th/2)
                    x2, y2 = int(tx + tw/2), int(ty + th/2)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_img, f"{self.target_class} {max_conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Pixel-space error from image center.
                    error_x = tx - center_x
                    error_y = ty - center_y
                    
                    # Draw error vector.
                    cv2.line(display_img, (int(center_x), int(center_y)), (int(tx), int(ty)), (0, 0, 255), 2)

                    # PID control outputs.
                    vx = self.pid_x.compute(error_x)
                    vy = self.pid_y.compute(error_y)
                    
                    cv2.putText(display_img, f"Err: {error_x:.1f}, {error_y:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(display_img, f"Cmd: {vx:.4f}, {vy:.4f}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Send command to robot.
                    if follow_enabled:
                        self.move_robot(vx, vy)
                    else:
                        cv2.putText(
                            display_img,
                            "Detected: press 's' to follow",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                else:
                    # Reset PID when target is lost.
                    self.pid_x.reset()
                    self.pid_y.reset()
                    cv2.putText(display_img, "Searching...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if follow_enabled:
                        cv2.putText(display_img, "FOLLOWING ENABLED", (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw image-center crosshair.
                cv2.line(display_img, (center_x-20, center_y), (center_x+20, center_y), (255, 0, 0), 1)
                cv2.line(display_img, (center_x, center_y-20), (center_x, center_y+20), (255, 0, 0), 1)
                if follow_enabled:
                    cv2.putText(display_img, "FOLLOWING", (10, self.height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, "IDLE", (10, self.height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Object Follower", display_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s') and not follow_enabled:
                    follow_enabled = True
                    logger.info("Follow triggered by key 's'")

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            logger.info("Task finished")


def main():
    parser = argparse.ArgumentParser(description="Nero Workcell - Object Following Task")
    parser.add_argument("--target", type=str, default="bottle", help="Target object class name (e.g., bottle, cup)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    follower = ObjectFollower(
        target_class=args.target,
        robot_ip="can",
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    follower.run()


if __name__ == "__main__":
    main()
