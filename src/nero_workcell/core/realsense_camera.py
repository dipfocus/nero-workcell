"""
Intel RealSense D435i camera implementation.
Supports RGB + depth streams.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """Intel RealSense D435i camera."""

    @staticmethod
    def _validate_intrinsics_data(intrinsics: Dict[str, Any]) -> None:
        """Reject invalid camera intrinsics instead of propagating bad values."""
        if intrinsics["width"] <= 0 or intrinsics["height"] <= 0:
            raise RuntimeError(
                f"Invalid camera intrinsics image size: {intrinsics['width']}x{intrinsics['height']}"
            )

        for field in ("fx", "fy", "cx", "cy"):
            value = intrinsics[field]
            if not np.isfinite(value):
                raise RuntimeError(f"Invalid camera intrinsic {field}: {value}")

        if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
            raise RuntimeError(
                f"Invalid camera focal lengths: fx={intrinsics['fx']}, fy={intrinsics['fy']}"
            )

        if not (0.0 <= intrinsics["cx"] < intrinsics["width"]):
            raise RuntimeError(
                f"Invalid principal point cx={intrinsics['cx']} for width={intrinsics['width']}"
            )

        if not (0.0 <= intrinsics["cy"] < intrinsics["height"]):
            raise RuntimeError(
                f"Invalid principal point cy={intrinsics['cy']} for height={intrinsics['height']}"
            )

        distortion_coeffs = intrinsics["distortion_coeffs"]
        if any(not np.isfinite(coeff) for coeff in distortion_coeffs):
            raise RuntimeError(f"Invalid distortion coefficients: {distortion_coeffs}")

    @classmethod
    def discover_serial_numbers(cls) -> List[str]:
        """Return serial numbers for all connected RealSense devices."""
        ctx = rs.context()
        devices = ctx.query_devices()
        serial_numbers: List[str] = []

        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            serial_numbers.append(serial)
            logger.info("Discovered device: %s (serial: %s)", name, serial)

        logger.info("Discovered %d RealSense camera(s)", len(serial_numbers))
        return serial_numbers

    @classmethod
    def setup(
        cls,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        serial_number: Optional[str] = None,
    ) -> "RealSenseCamera":
        """Create and start a RealSense camera, optionally by explicit serial number."""
        requested_serial = serial_number.strip() if isinstance(serial_number, str) else None
        available_serials = cls.discover_serial_numbers()

        if requested_serial:
            if requested_serial not in available_serials:
                raise RuntimeError(
                    f"Requested RealSense camera not found: {requested_serial}"
                )
            selected_serial = requested_serial
        else:
            if not available_serials:
                raise RuntimeError("No RealSense camera found")
            selected_serial = available_serials[0]

        camera = cls(
            width=width,
            height=height,
            fps=fps,
            serial_number=selected_serial,
        )
        logger.info("Using camera: %s", camera.serial_number)
        camera.start()
        return camera
    
    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 *,
                 serial_number: str):
        """
        Initialize the RealSense camera.

        Args:
            width: Image width.
            height: Image height.
            fps: Frame rate.
            serial_number: Camera serial number (required).
        """
        if not isinstance(serial_number, str) or not serial_number.strip():
            raise ValueError("serial_number is required and cannot be empty")

        self.width = width
        self.height = height
        self.fps = fps
        self._is_opened = False

        self.frames_captured = 0
        self.failed_reads = 0
        self.start_time = 0

        self.serial_number = serial_number.strip()
        
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self.depth_scale = 1.0
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
    
    def start(self) -> None:
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Bind to the target device by serial number.
            self.config.enable_device(self.serial_number)
            
            # Always enable depth and color streams.
            self.config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps
            )
            self.config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps
            )
            
            # Start the RealSense pipeline.
            self.profile = self.pipeline.start(self.config)
            
            # Always align depth to color.
            self.align = rs.align(rs.stream.color)

            # Read depth scale.
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Read device info.
            device = self.profile.get_device()
            device_name = device.get_info(rs.camera_info.name)

            self._warm_up(frame_count=self.fps)
            
            self.start_time = time.time()
            self._is_opened = True
            self.get_intrinsics()
            logger.info(
                "Camera is ready: device=%s, serial=%s, resolution=%sx%s, fps=%s, depth_scale=%s",
                device_name,
                self.serial_number,
                self.width,
                self.height,
                self.fps,
                self.depth_scale,
            )
            
        except Exception as e:
            logger.error("Camera start failed: %s", e)
            try:
                if self.pipeline is not None:
                    self.pipeline.stop()
            except Exception as stop_error:
                logger.debug("Cleanup after start failure failed: %s", stop_error)
            self.pipeline = None
            self.config = None
            self.align = None
            self.profile = None
            self._is_opened = False
            raise RuntimeError(f"Failed to start camera: {self.serial_number}") from e

    def _warm_up(self, frame_count: int, timeout_ms: int = 1000):
        """Wait for a small number of frames so exposure and streams stabilize."""
        if self.pipeline is None:
            raise RuntimeError("Camera pipeline is not initialized")
        if frame_count < 1:
            raise ValueError("frame_count must be greater than 0")

        for _ in range(frame_count):
            self.pipeline.wait_for_frames(timeout_ms=timeout_ms)

        logger.info("Camera warm-up completed with %d frames", frame_count)

    def read_frame(self) -> Dict[str, Any]:
        """
        Read one frame and return a unified payload.

        Returns:
            dict: {
                'color': np.ndarray | None,  # BGR image (H, W, 3)
                'depth': np.ndarray | None,  # Depth map (H, W), in meters
                'timestamp': float,          # Unix timestamp
            }

        Raises:
            RuntimeError: Raised when camera is not started.
        """
        if not self._is_opened:
            raise RuntimeError("Camera is not opened. Call start() first.")

        try:
            result = self._read_frame_raw()
            result["timestamp"] = time.time()
        except AssertionError:
            raise
        except Exception as e:
            logger.error("Read frame failed: %s", e)
            result = {"color": None, "depth": None, "timestamp": time.time()}

        has_color = result.get("color") is not None
        has_depth = result.get("depth") is not None

        if has_color and has_depth:
            self.frames_captured += 1
        else:
            self.failed_reads += 1

        logger.debug(
            "Read frame completed: color=%s, depth=%s, timestamp=%.3f",
            has_color,
            has_depth,
            result["timestamp"],
        )
        return result
    
    def _read_frame_raw(self) -> Dict[str, Any]:
        assert self.pipeline is not None, "Camera pipeline is not initialized"
        assert self.align is not None, "Depth alignment is not initialized"
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align depth frame to color frame.
            frames = self.align.process(frames)
            
            result = {'color': None, 'depth': None}
            
            # Read color frame.
            color_frame = frames.get_color_frame()
            if color_frame:
                result['color'] = np.asanyarray(color_frame.get_data())

            # Read depth frame and convert to meters.
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                result['depth'] = depth_image.astype(np.float32) * self.depth_scale
            
            return result
            
        except Exception as e:
            logger.error("Read raw frame failed: %s", e)
            return {'color': None, 'depth': None}
    
    @property
    def is_opened(self) -> bool:
        return self._is_opened

    def get_stats(self) -> Dict[str, Any]:
        total = self.frames_captured + self.failed_reads
        success_rate = (self.frames_captured / total * 100) if total > 0 else 0
        return {
            "frames_captured": self.frames_captured,
            "failed_reads": self.failed_reads,
            "success_rate": success_rate,
        }

    def print_stats(self):
        stats = self.get_stats()
        logger.info("Camera statistics")
        logger.info("  Captured frames: %s", stats["frames_captured"])
        logger.info("  Failed reads: %s", stats["failed_reads"])
        logger.info("  Success rate: %.1f%%", stats["success_rate"])
    
    def get_intrinsics(self) -> Dict[str, Any]:
        if not self._is_opened:
            raise RuntimeError("Camera is not opened. Call start() first.")
        assert self.profile is not None, "Camera profile is not initialized"
        
        try:
            stream = self.profile.get_stream(rs.stream.color)
            intrinsics = stream.as_video_stream_profile().get_intrinsics()
            intrinsics_data = {
                'width': int(intrinsics.width),
                'height': int(intrinsics.height),
                'fx': float(intrinsics.fx),
                'fy': float(intrinsics.fy),
                'cx': float(intrinsics.ppx),
                'cy': float(intrinsics.ppy),
                'distortion_model': str(intrinsics.model),
                'distortion_coeffs': [float(coeff) for coeff in intrinsics.coeffs],
            }
        except Exception as e:
            logger.error("Get intrinsics failed: %s", e)
            raise RuntimeError("Failed to get camera intrinsics") from e

        if (
            intrinsics_data["width"] != self.width
            or intrinsics_data["height"] != self.height
        ):
            logger.warning(
                "Requested color resolution %sx%s, but camera intrinsics report %sx%s",
                self.width,
                self.height,
                intrinsics_data["width"],
                intrinsics_data["height"],
            )

        self._validate_intrinsics_data(intrinsics_data)
        self.fx = intrinsics_data["fx"]
        self.fy = intrinsics_data["fy"]
        self.cx = intrinsics_data["cx"]
        self.cy = intrinsics_data["cy"]
        return intrinsics_data

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        
        self._is_opened = False
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.print_stats()
        logger.info("Camera stopped")
