#!/usr/bin/env python3
# coding=utf-8
"""
Object follower implementation for the static-target follow task.
"""

import logging
from typing import Optional

import numpy as np

from .nero_controller import NeroController
from .pid_controller import PIDController
from .target_object import TargetObject

logger = logging.getLogger(__name__)


class ObjectFollower:
    def __init__(
        self,
        robot_channel: str = "can0",
        target_distance: float = 0.3,
    ):
        self.target_distance = target_distance

        self.pid_x = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        self.pid_y = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        self.pid_z = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        self.locked_target: Optional[TargetObject] = None

        self.robot = NeroController(robot_channel, "nero")

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

    def follow_target(self, target: TargetObject) -> bool:
        """
        Follow a target point in base coordinates (no grasping).

        Constraints:
        - target.position must be in the base frame.
        - Desired TCP is self.target_distance meters above the target point.
        """
        if target.frame != "base":
            raise ValueError(f"Expected base-frame target, got frame='{target.frame}'")

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

        deadband = 0.005
        if abs(err_x) < deadband and abs(err_y) < deadband and abs(err_z) < deadband:
            return True

        err_scale = 1000.0
        cmd_x = float(self.pid_x.compute(err_x * err_scale))
        cmd_y = float(self.pid_y.compute(err_y * err_scale))
        cmd_z = float(self.pid_z.compute(err_z * err_scale))

        max_step_z = 0.03
        cmd_z = float(np.clip(cmd_z, -max_step_z, max_step_z))

        self.robot.move_relative(dx=cmd_x, dy=cmd_y, dz=cmd_z)
        return False
