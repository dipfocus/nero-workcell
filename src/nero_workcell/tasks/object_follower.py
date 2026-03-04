#!/usr/bin/env python3
# coding=utf-8
"""
物体跟随任务 (Visual Servoing)
结合 RealSense 相机和 YOLO 模型，控制机械臂跟随指定物体。

用法:
    python -m nero_workcell.tasks.object_follower --target bottle --conf 0.5
"""

import time
import logging
import argparse

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from nero_workcell.core import NeroController, RealSenseCamera

logger = logging.getLogger(__name__)


class PIDController:
    """简单的 PID 控制器，用于计算运动速度"""
    def __init__(self, kp: float, ki: float, kd: float, max_out: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error: float) -> float:
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0

        # 比例项
        p_term = self.kp * error

        # 积分项
        self.integral += error * dt
        i_term = self.ki * self.integral

        # 微分项
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # 计算输出
        output = p_term + i_term + d_term
        
        # 限幅
        output = np.clip(output, -self.max_out, self.max_out)

        self.prev_error = error
        self.last_time = current_time
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


class ObjectFollower:
    def __init__(self, 
                 target_class: str, 
                 robot_ip: str, 
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.5):
        
        self.target_class = target_class
        self.robot_ip = robot_ip
        self.conf_threshold = conf_threshold
        
        # 初始化相机
        self.width = 640
        self.height = 480
        self.camera = RealSenseCamera(width=self.width, height=self.height, fps=30, serial_number="") 
        # 注意：这里 serial_number 留空需要在 start 前设置，或者自动搜索
        
        # 初始化 YOLO
        logger.info(f"加载 YOLO 模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 初始化 PID 控制器 (X轴和Y轴)
        # 参数需要根据实际机械臂响应速度进行调整
        self.pid_x = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05) # 这里的单位是 m/s
        self.pid_y = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        
        # 机械臂实例
        self.robot = NeroController(self.robot_ip)
        self.is_running = False

    def setup_camera(self):
        """自动查找并连接第一个 RealSense 相机"""
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未找到 RealSense 相机")
        
        serial = devices[0].get_info(rs.camera_info.serial_number)
        self.camera.serial_number = serial
        logger.info(f"使用相机: {serial}")
        
        if not self.camera.start():
            raise RuntimeError("相机启动失败")

    def move_robot(self, vx: float, vy: float):
        """
        控制机械臂移动
        vx, vy: PID 输出的控制量，此处作为位移增量处理
        
        注意：这里假设了相机坐标系与机械臂末端坐标系的关系。
        通常 RealSense: +X 右, +Y 下, +Z 前
        机械臂末端: 需要根据实际安装确认。
        这里假设：
        - 图像 X 轴偏差 -> 控制机械臂沿 Y 轴移动 (左右)
        - 图像 Y 轴偏差 -> 控制机械臂沿 X 轴移动 (上下/前后，取决于安装)
        
        *请根据实际情况修改轴映射*
        """
        if not self.robot.is_connected():
            return

        # 简单的死区设置，避免微小抖动
        if abs(vx) < 0.001 and abs(vy) < 0.001:
            return

        try:
            # 坐标系映射 (根据实际安装调整)
            # 假设: 图像 X+ (右) -> 机械臂 Y-
            # 假设: 图像 Y+ (下) -> 机械臂 X-
            
            scale = 0.5
            dx = -vy * scale
            dy = -vx * scale
            
            # 发送相对运动指令
            self.robot.move_relative(dx=dx, dy=dy)
            
        except Exception as e:
            logger.error(f"运动控制失败: {e}")

    def run(self):
        self.setup_camera()
        if not self.robot.connect():
            logger.error("机械臂连接失败，任务终止")
            return
        self.is_running = True
        
        logger.info(f"开始跟随任务，目标: {self.target_class}")
        logger.info("按 'q' 退出")

        center_x, center_y = self.width // 2, self.height // 2

        try:
            while self.is_running:
                # 1. 读取图像
                frame_data = self.camera.read_frame()
                color_image = frame_data['color']
                if color_image is None:
                    continue

                # 2. YOLO 检测
                results = self.model(color_image, verbose=False, conf=self.conf_threshold)
                
                target_box = None
                max_conf = 0

                # 3. 寻找目标物体
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id]
                        conf = float(box.conf[0])
                        
                        if cls_name == self.target_class and conf > max_conf:
                            max_conf = conf
                            target_box = box.xywh[0].cpu().numpy() # x_center, y_center, w, h

                # 4. 计算误差并控制
                display_img = color_image.copy()
                
                if target_box is not None:
                    tx, ty, tw, th = target_box
                    
                    # 绘制目标框
                    x1, y1 = int(tx - tw/2), int(ty - th/2)
                    x2, y2 = int(tx + tw/2), int(ty + th/2)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_img, f"{self.target_class} {max_conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 计算像素误差
                    error_x = tx - center_x
                    error_y = ty - center_y
                    
                    # 绘制误差线
                    cv2.line(display_img, (int(center_x), int(center_y)), (int(tx), int(ty)), (0, 0, 255), 2)

                    # PID 计算控制量
                    vx = self.pid_x.compute(error_x)
                    vy = self.pid_y.compute(error_y)
                    
                    cv2.putText(display_img, f"Err: {error_x:.1f}, {error_y:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(display_img, f"Cmd: {vx:.4f}, {vy:.4f}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # 发送给机械臂
                    self.move_robot(vx, vy)
                else:
                    # 丢失目标，重置 PID
                    self.pid_x.reset()
                    self.pid_y.reset()
                    cv2.putText(display_img, "Searching...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 显示中心十字
                cv2.line(display_img, (center_x-20, center_y), (center_x+20, center_y), (255, 0, 0), 1)
                cv2.line(display_img, (center_x, center_y-20), (center_x, center_y+20), (255, 0, 0), 1)

                cv2.imshow("Object Follower", display_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            logger.info("任务结束")


def main():
    parser = argparse.ArgumentParser(description="Nero Workcell - Object Following Task")
    parser.add_argument("--target", type=str, default="bottle", help="Target object class name (e.g., bottle, cup)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # 配置日志
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
