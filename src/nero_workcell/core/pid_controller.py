"""
PID controller utility.
"""

import time

import numpy as np


class PIDController:
    """Simple PID controller for motion control."""

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

        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        output = np.clip(output, -self.max_out, self.max_out)

        self.prev_error = error
        self.last_time = current_time
        return float(output)

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
