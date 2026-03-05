#!/usr/bin/env python3
# coding=utf-8
"""
眼在手外标定
使用采集到的图片信息和机械臂位姿信息计算相机坐标系相对于机械臂基座标的旋转矩阵和平移向量
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
    """位姿转齐次变换矩阵 (使用scipy，和collect_data.py一致)"""
    x, y, z, rx, ry, rz = pose
    rotation = R.from_euler('xyz', [rx, ry, rz], degrees=False)
    H = np.eye(4)
    H[:3, :3] = rotation.as_matrix()
    H[:3, 3] = [x, y, z]
    return H


def inverse_matrix(T):
    """计算齐次变换矩阵的逆"""
    R_mat = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t
    return T_inv


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
    
    逻辑：
    - 比较第一个点和最后一个点的位置
    - 如果第一个点不在左下角（y值较大，x值较小），则翻转整个数组
    """
    first_point = corners[0][0]
    last_point = corners[-1][0]
    
    # 判断哪个点更接近左下角
    # 左下角的特征：x值小，y值大（图像坐标系y向下）
    # 用 (y - x) 作为判断依据，值越大越接近左下角
    first_score = first_point[1] - first_point[0]  # y - x
    last_score = last_point[1] - last_point[0]
    
    # 如果最后一个点更接近左下角，翻转数组
    if last_score > first_score:
        corners = corners[::-1].copy()
    
    return corners


def calibrate(images_path, poses, corner_long, corner_short, corner_size, show_images=True):
    """执行手眼标定"""
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
    
    for i in range(1, len(poses) + 1):
        image_path = f"{images_path}/{i}.jpg"
        if not os.path.exists(image_path):
            continue
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, (corner_long, corner_short), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            
            # 确保起始点在左下角
            corners2 = normalize_corner_order(corners2)
            
            img_points.append(corners2)
            valid_pose_indices.append(i - 1)
            
            # 可视化校对：绘制检测到的角点，并标记起始点和终点
            if show_images:
                cv2.drawChessboardCorners(img, (corner_long, corner_short), corners2, ret)
                
                # 标记起始点（绿色）和终点（红色）
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
                cv2.imshow(f'Calibration Check', img)
                cv2.waitKey(0)  # 等待按键继续
            
            start_point = tuple(corners2[0][0].astype(int))
            print(f"  图像 {i}: 检测到角点 ✓ (起始点: {start_point})")
        else:
            print(f"  图像 {i}: 未检测到角点 ✗")
    
    cv2.destroyAllWindows() if show_images else None
    
    N = len(img_points)
    print(f"\n有效图像数: {N}")
    
    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(f"\n内参矩阵:\n{mtx}")
    print(f"\n畸变系数: {dist.flatten()}")
    
    # 准备机械臂位姿 (使用逆变换)
    R_gripper = []
    t_gripper = []
    
    for idx in valid_pose_indices:
        pose = poses[idx]
        T = pose_to_homogeneous_matrix(pose)
        T_inv = inverse_matrix(T)
        R_gripper.append(T_inv[:3, :3])
        t_gripper.append(T_inv[:3, 3])
    
    # 手眼标定 - 多种方法
    methods = [
        ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
        ("PARK", cv2.CALIB_HAND_EYE_PARK),
        ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
    ]
    
    results = {}
    for name, method in methods:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper, t_gripper,
            rvecs, tvecs,
            method=method
        )
        results[name] = (R_cam2base, t_cam2base)
        print(f"\n{name} 方法:")
        print(f"  旋转矩阵:\n{R_cam2base}")
        print(f"  平移向量: {t_cam2base.flatten()}")
    
    return results, mtx, dist


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='眼在手外标定程序')
    parser.add_argument('--no-show', '-n', action='store_true', 
                        help='不显示图像，直接计算标定结果')
    parser.add_argument('--show', '-s', action='store_true', default=True,
                        help='显示图像进行校对（默认）')
    args = parser.parse_args()
    
    show_images = not args.no_show
    
    # 读取配置（与脚本同目录）
    config_file = Path(__file__).with_name("config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    corner_long = config['checkerboard']['corner_point_long']
    corner_short = config['checkerboard']['corner_point_short']
    corner_size = config['checkerboard']['corner_point_size']
    
    # 找最新数据文件夹（在当前目录下）
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
    print("眼在手外标定")
    print("="*60)
    print(f"数据路径: {images_path}")
    
    # 加载位姿
    poses = load_poses(poses_file)
    print(f"位姿数量: {len(poses)}")
    
    # 执行标定
    results, mtx, dist = calibrate(images_path, poses, corner_long, corner_short, corner_size, show_images)
    
    # 使用 TSAI 方法的结果
    R_cam2base, t_cam2base = results["TSAI"]
    
    # 转换为四元数和欧拉角
    rotation = R.from_matrix(R_cam2base)
    quat = rotation.as_quat()  # [x, y, z, w]
    euler = rotation.as_euler('xyz', degrees=True)
    
    print("\n" + "="*60)
    print("最终结果 (TSAI)")
    print("="*60)
    print(f"平移向量: x={t_cam2base[0,0]:.4f}, y={t_cam2base[1,0]:.4f}, z={t_cam2base[2,0]:.4f}")
    print(f"四元数: qx={quat[0]:.4f}, qy={quat[1]:.4f}, qz={quat[2]:.4f}, qw={quat[3]:.4f}")
    print(f"欧拉角(度): rx={euler[0]:.2f}, ry={euler[1]:.2f}, rz={euler[2]:.2f}")
    
    # 保存结果
    calibration_result = {
        "rotation_matrix": R_cam2base.tolist(),
        "translation_vector": t_cam2base.flatten().tolist(),
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
            "x": float(t_cam2base[0,0]),
            "y": float(t_cam2base[1,0]),
            "z": float(t_cam2base[2,0])
        },
        "calibration_type": "eye_to_hand",
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
