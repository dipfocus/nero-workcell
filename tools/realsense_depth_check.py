#!/usr/bin/env python3
# coding=utf-8
"""
最简单的深度测试 - 不做任何对齐，直接读取原始深度
"""

import cv2
import numpy as np
import pyrealsense2 as rs

print("=== 最简单的深度测试 ===\n")

# 创建管道
pipeline = rs.pipeline()
config = rs.config()

# 只启用深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动
print("启动相机...")
profile = pipeline.start(config)

# 获取深度比例
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"深度比例: {depth_scale}")

print("\n按 'q' 退出")

try:
    frame_count = 0
    while True:
        # 等待帧 - 不做任何对齐
        frames = pipeline.wait_for_frames()
        
        # 直接获取深度帧
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame:
            print("没有深度帧!")
            continue
        
        # 转换为numpy - 原始uint16数据
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
        
        frame_count += 1
        
        # 每30帧打印一次统计
        if frame_count % 30 == 1:
            valid_mask = depth_image > 0
            valid_count = np.sum(valid_mask)
            print(f"帧 {frame_count}: 有效像素 {valid_count}/{depth_image.size} ({100*valid_count/depth_image.size:.1f}%)")
            if valid_count > 0:
                print(f"  深度范围: {np.min(depth_image[valid_mask])}-{np.max(depth_image[valid_mask])} mm")
                # 中心点
                h, w = depth_image.shape
                center_val = depth_image[h//2, w//2]
                center_dist = depth_frame.get_distance(w//2, h//2)
                print(f"  中心点: {center_val} mm / {center_dist:.3f} m")
        
        # 显示深度图 - 使用动态范围，类似realsense-viewer
        valid_mask = depth_image > 0
        if np.any(valid_mask):
            min_depth = np.min(depth_image[valid_mask])
            max_depth = np.percentile(depth_image[valid_mask], 95)  # 用95%分位数避免极端值
            # 归一化到0-255
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
            depth_normalized[valid_mask] = np.clip(
                255 * (depth_image[valid_mask] - min_depth) / (max_depth - min_depth + 1),
                0, 255
            ).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            # 无效区域显示为黑色
            depth_colormap[~valid_mask] = [0, 0, 0]
        else:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
        
        # 显示有效像素比例
        valid_count = np.sum(depth_image > 0)
        valid_ratio = 100 * valid_count / depth_image.size
        cv2.putText(depth_colormap, f"Valid: {valid_ratio:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Depth (No Align)', depth_colormap)
        if color_image is not None:
            cv2.imshow('Color', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("已停止")
