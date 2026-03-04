#!/usr/bin/env python3
# coding=utf-8
"""
NeroController 真机测试用例

说明:
- 此脚本用于在真实机械臂上测试 NeroController 的核心功能。
- 包含连接、读取状态和简单的运动测试。

警告:
- 运行 test_03_relative_movement 会驱动机械臂运动！
- 请确保机械臂周围没有障碍物。
- 建议手放在急停按钮上。
"""

import unittest
import time
import logging
import sys
import os

# 尝试导入 NeroController
# 自动处理路径，以便在不同目录下运行
try:
    from nero_workcell.core.nero_controller import NeroController
except ImportError:
    # 如果作为脚本直接运行，将 src 目录添加到路径
    # 假设目录结构: project_root/tests/test_nero_controller_real.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    src_dir = os.path.join(project_root, "src")
    
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from nero_workcell.core.nero_controller import NeroController

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestNeroReal")


class TestNeroControllerReal(unittest.TestCase):
    """NeroController 真机集成测试"""

    def setUp(self):
        """每个测试前初始化控制器"""
        self.controller = NeroController(channel="can0", robot_type="nero")
        
    def tearDown(self):
        """每个测试后断开连接"""
        if self.controller.is_connected():
            self.controller.disconnect()

    def test_01_connection(self):
        """测试：连接和断开机械臂"""
        logger.info("=== 测试 01: 连接测试 ===")
        success = self.controller.connect()
        self.assertTrue(success, "连接机械臂失败，请检查 CAN 连接或电源")
        self.assertTrue(self.controller.is_connected(), "连接状态应为 True")
        
        # 断开连接
        self.controller.disconnect()
        self.assertFalse(self.controller.is_connected(), "断开后连接状态应为 False")
        logger.info("连接测试通过")

    def test_02_read_poses(self):
        """测试：读取法兰和TCP位姿"""
        logger.info("=== 测试 02: 位姿读取测试 ===")
        if not self.controller.connect():
            self.skipTest("无法连接机械臂，跳过测试")
            
        # 1. 获取法兰位姿
        flange_pose = self.controller.get_flange_pose()
        logger.info(f"法兰位姿: {flange_pose}")
        self.assertIsNotNone(flange_pose, "法兰位姿不应为空")
        self.assertEqual(len(flange_pose), 6, "位姿数据长度应为 6 [x,y,z,r,p,y]")
        
        # 2. 获取 TCP 位姿
        tcp_pose = self.controller.get_tcp_pose()
        logger.info(f"TCP 位姿: {tcp_pose}")
        self.assertIsNotNone(tcp_pose, "TCP 位姿不应为空")
        self.assertEqual(len(tcp_pose), 6, "位姿数据长度应为 6")
        
        logger.info("位姿读取测试通过")

    def test_03_relative_movement(self):
        """测试：相对运动 (Z轴移动 1cm)"""
        logger.info("=== 测试 03: 相对运动测试 (Z轴 +1cm) ===")
        if not self.controller.connect():
            self.skipTest("无法连接机械臂，跳过测试")
            
        # 获取初始位姿
        start_pose = self.controller.get_tcp_pose()
        self.assertIsNotNone(start_pose)
        initial_z = start_pose[2]
        
        move_dist = 0.01  # 1cm
        logger.info(f"当前 Z: {initial_z:.4f}, 准备向上移动 {move_dist}m")
        
        # 执行移动
        self.controller.move_relative(dz=move_dist)
        time.sleep(2.0)  # 等待运动完成
        
        # 检查新位置
        mid_pose = self.controller.get_tcp_pose()
        self.assertIsNotNone(mid_pose)
        mid_z = mid_pose[2]
        logger.info(f"移动后 Z: {mid_z:.4f}")
        
        # 验证 Z 轴变化 (允许 2mm 误差)
        self.assertAlmostEqual(mid_z, initial_z + move_dist, delta=0.002, 
                             msg="Z轴移动距离不符合预期")
        
        # 恢复位置
        logger.info("恢复原位...")
        self.controller.move_relative(dz=-move_dist)
        time.sleep(2.0)
        
        end_pose = self.controller.get_tcp_pose()
        end_z = end_pose[2]
        logger.info(f"恢复后 Z: {end_z:.4f}")
        
        self.assertAlmostEqual(end_z, initial_z, delta=0.002,
                             msg="未能准确回到原位")
        logger.info("相对运动测试通过")

if __name__ == '__main__':
    unittest.main()