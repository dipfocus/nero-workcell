#!/usr/bin/env python3
# coding=utf-8
"""
眼在手上标定 (Eye-in-Hand)
使用采集到的图片信息和机械臂位姿信息计算相机坐标系相对于机械臂末端坐标系的旋转矩阵和平移向量

与眼在手外(Eye-to-Hand)的区别:
- 眼在手上: 相机安装在机械臂末端，标定板固定在工作台
  - calibrateHandEye 输入: R_gripper2base, t_gripper2base (末端在基座下的位姿)
  - 输出: T_cam_to_gripper (相机到末端的变换)
  
- 眼在手外: 相机固定在工作台，标定板安装在机械臂末端
  - calibrateHandEye 输入: R_base2gripper, t_base2gripper (基座在末端下的位姿，即逆变换)
  - 输出: T_cam_to_base (相机到基座的变换)
"""

import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import argparse
from pathlib import Path

np.set_printoptions(precision=8, suppress=True)


def pose_to_homogeneous_matrix(pose):
    """
    位姿转齐次变换矩阵
    使用 scipy 的 from_euler，与 collect_data.py 中的欧拉角顺序一致
    """
    x, y, z, rx, ry, rz = pose
    rotation = R.from_euler('xyz', [rx, ry, rz], degrees=False)
    H = np.eye(4)
    H[:3, :3] = rotation.as_matrix()
    H[:3, 3] = [x, y, z]
    return H


def load_poses(filepath):
    """加载位姿数据"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    poses = []
    for line in lines:
        if line.strip():
            values = [float(x) for x in line.strip().split(',')]
            poses.append(values)
    return poses


def normalize_corner_order(corners):
    """
    调整角点顺序，确保起始点始终在左下角
    """
    first_point = corners[0][0]
    last_point = corners[-1][0]
    
    first_score = first_point[1] - first_point[0]
    last_score = last_point[1] - last_point[0]
    
    if last_score > first_score:
        corners = corners[::-1].copy()
    
    return corners


def calibrate(images_path, poses, corner_long, corner_short, corner_size, show_images=True):
    """
    执行眼在手上标定
    
    参考 hand-eye/1/eyeInHand/eyeInHand.py 的逻辑:
    - 直接使用末端在基座下的位姿 (R_gripper2base, t_gripper2base)
    - 不需要逆变换
    """
    print(f"标定板参数: {corner_long}x{corner_short}, 方格尺寸: {corner_size}m")
    print(f"显示图像: {'是' if show_images else '否'}")
    
    # 准备标定板角点
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    objp = np.zeros((corner_long * corner_short, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_long, 0:corner_short].T.reshape(-1, 2)
    objp = corner_size * objp
    
    obj_points = []
    img_points = []
    valid_pose_indices = []
    size = None
    
    # 遍历图片，从 1.jpg 开始（与 collect_data.py 保存的命名一致）
    for i in range(1, len(poses) + 1):
        image_path = f"{images_path}/{i}.jpg"
        if not os.path.exists(image_path):
            continue
        
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, (corner_long, corner_short), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            
            # 确保起始点在左下角
            corners2 = normalize_corner_order(corners2)
            
            img_points.append(corners2)
            valid_pose_indices.append(i - 1)  # poses 列表从 0 开始索引
            
            if show_images:
                cv2.drawChessboardCorners(img, (corner_long, corner_short), corners2, ret)
                
                start_point = tuple(corners2[0][0].astype(int))
                end_point = tuple(corners2[-1][0].astype(int))
                cv2.circle(img, start_point, 15, (0, 255, 0), 3)
                cv2.putText(img, "START", (start_point[0] + 20, start_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(img, end_point, 15, (0, 0, 255), 3)
                cv2.putText(img, "END", (end_point[0] + 20, end_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.putText(img, f"Image {i} - Press any key", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Calibration Check', img)
                cv2.waitKey(0)
            
            start_point = tuple(corners2[0][0].astype(int))
            print(f"  图像 {i}: 检测到角点 ✓ (起始点: {start_point})")
        else:
            print(f"  图像 {i}: 未检测到角点 ✗")
    
    if show_images:
        cv2.destroyAllWindows()
    
    N = len(img_points)
    print(f"\n有效图像数: {N}")
    
    if N < 3:
        print("错误: 有效图像数量不足，至少需要3张!")
        return None, None, None
    
    # 相机标定，得到标定板在相机坐标系下的位姿 (rvecs, tvecs)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(f"\n内参矩阵:\n{mtx}")
    print(f"\n畸变系数: {dist.flatten()}")
    print("-----------------------------------------------------")
    
    # 准备机械臂末端位姿
    # 眼在手上: 直接使用末端在基座下的位姿 (R_gripper2base, t_gripper2base)
    # 参考代码中: R_tool, t_tool 就是末端位姿的旋转和平移
    R_gripper2base = []
    t_gripper2base = []
    
    for idx in valid_pose_indices:
        pose = poses[idx]
        T = pose_to_homogeneous_matrix(pose)
        R_gripper2base.append(T[:3, :3])
        t_gripper2base.append(T[:3, 3])
    
    # 手眼标定 - 多种方法
    methods = [
        ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
        ("PARK", cv2.CALIB_HAND_EYE_PARK),
        ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
    ]
    
    results = {}
    for name, method in methods:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            rvecs, tvecs,
            method=method
        )
        results[name] = (R_cam2gripper, t_cam2gripper)
        print(f"\n{name} 方法计算的旋转矩阵:")
        print(R_cam2gripper)
        print(f"{name} 方法计算的平移向量:")
        print(t_cam2gripper)
    
    return results, mtx, dist


def main():
    parser = argparse.ArgumentParser(description='眼在手上标定程序 (Eye-in-Hand)')
    parser.add_argument('--no-show', '-n', action='store_true', 
                        help='不显示图像，直接计算标定结果')
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='指定数据文件夹路径')
    args = parser.parse_args()
    
    show_images = not args.no_show
    
    # 读取配置（与脚本同目录）
    config_file = Path(__file__).with_name("config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    corner_long = config['checkerboard']['corner_point_long']
    corner_short = config['checkerboard']['corner_point_short']
    corner_size = config['checkerboard']['corner_point_size']
    
    # 确定数据文件夹（在当前目录下）
    if args.data:
        images_path = args.data
    else:
        images_base = "./images"
        if not os.path.exists(images_base):
            print("错误: 未找到 images 文件夹!")
            return
            
        data_folders = [f for f in os.listdir(images_base) 
                        if f.startswith('data') and os.path.isdir(os.path.join(images_base, f))]
        if not data_folders:
            print("错误: 未找到数据文件夹!")
            return
        
        latest_folder = sorted(data_folders)[-1]
        images_path = os.path.join(images_base, latest_folder)
    
    poses_file = os.path.join(images_path, "poses.txt")
    
    print("="*60)
    print("眼在手上标定 (Eye-in-Hand)")
    print("="*60)
    print(f"数据路径: {images_path}")
    print(f"位姿文件: {poses_file}")
    
    # 加载位姿
    poses = load_poses(poses_file)
    print(f"位姿数量: {len(poses)}")
    
    # 执行标定
    results, mtx, dist = calibrate(images_path, poses, corner_long, corner_short, corner_size, show_images)
    
    if results is None:
        print("标定失败!")
        return
    
    # 使用 TSAI 方法的结果
    R_cam2gripper, t_cam2gripper = results["TSAI"]
    
    print("\n" + "="*60)
    print("默认返回 TSAI 方法计算结果")
    print("可根据实际情况自行选择合适的矩阵和平移向量")
    print("="*60)
    print("rotation_matrix:")
    print(R_cam2gripper)
    print("translation_vector:")
    print(t_cam2gripper)
    
    # 转换为四元数和欧拉角
    rotation = R.from_matrix(R_cam2gripper)
    quat = rotation.as_quat()  # [x, y, z, w]
    euler = rotation.as_euler('xyz', degrees=True)
    
    print(f"\n平移向量: x={t_cam2gripper[0,0]:.4f}, y={t_cam2gripper[1,0]:.4f}, z={t_cam2gripper[2,0]:.4f}")
    print(f"四元数: qx={quat[0]:.4f}, qy={quat[1]:.4f}, qz={quat[2]:.4f}, qw={quat[3]:.4f}")
    print(f"欧拉角(度): rx={euler[0]:.2f}, ry={euler[1]:.2f}, rz={euler[2]:.2f}")
    
    # 构建齐次变换矩阵
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    # 保存结果
    calibration_result = {
        "rotation_matrix": R_cam2gripper.tolist(),
        "translation_vector": t_cam2gripper.flatten().tolist(),
        "homogeneous_matrix": T_cam2gripper.tolist(),
        "quaternion": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3])
        },
        "euler_angles_deg": {
            "rx": float(euler[0]),
            "ry": float(euler[1]),
            "rz": float(euler[2])
        },
        "position": {
            "x": float(t_cam2gripper[0,0]),
            "y": float(t_cam2gripper[1,0]),
            "z": float(t_cam2gripper[2,0])
        },
        "calibration_type": "eye_in_hand",
        "method": "TSAI",
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "checkerboard": {
            "corner_point_long": corner_long,
            "corner_point_short": corner_short,
            "corner_point_size": corner_size,
            "unit": "meters"
        }
    }
    
    # 保存到数据目录和当前目录
    for save_path in [os.path.join(images_path, "hand_eye_calibration.json"), 
                      "./hand_eye_calibration.json"]:
        with open(save_path, 'w') as f:
            json.dump(calibration_result, f, indent=4)
        print(f"已保存: {save_path}")


if __name__ == '__main__':
    main()
