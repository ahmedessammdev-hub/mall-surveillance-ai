"""
Multi-Camera Stream Manager.

Manages RTSP streams, video files, and webcam sources with:
- Thread-per-camera reading
- Frame sampling to target FPS
- Health monitoring
- Automatic reconnection
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    RTSP = "rtsp"
    FILE = "file"
    WEBCAM = "webcam"


@dataclass
class CameraStatus:
    """Runtime status of a camera stream."""
    camera_id: str
    source: str
    source_type: SourceType
    is_connected: bool = False
    fps_actual: float = 0.0
    frame_count: int = 0
    last_frame_time: float = 0.0
    reconnect_count: int = 0
    error: str = ""
    resolution: tuple[int, int] = (0, 0)


class CameraStream:
    """A single camera stream running in its own thread."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        target_fps: int = 5,
        reconnect_interval: float = 5.0,
        max_reconnects: int = 10,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.reconnect_interval = reconnect_interval
        self.max_reconnects = max_reconnects

        self.source_type = self._detect_source_type(source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._status = CameraStatus(
            camera_id=camera_id,
            source=source,
            source_type=self.source_type,
        )

    @staticmethod
    def _detect_source_type(source: str) -> SourceType:
        """Auto-detect whether source is RTSP, file, or webcam."""
        if source.startswith("rtsp://") or source.startswith("rtsps://"):
            return SourceType.RTSP
        try:
            idx = int(source)
            return SourceType.WEBCAM
        except (ValueError, TypeError):
            return SourceType.FILE

    def start(self) -> bool:
        """Start reading from the camera source."""
        if self._running:
            return True

        if not self._connect():
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"cam-{self.camera_id}",
        )
        self._thread.start()
        logger.info(f"Camera {self.camera_id} stream started ({self.source_type.value})")
        return True

    def stop(self) -> None:
        """Stop reading from the camera source."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._disconnect()
        logger.info(f"Camera {self.camera_id} stream stopped")

    def get_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame from the camera.

        Returns:
            (success, frame) tuple. Frame is None if no frame is available.
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def get_status(self) -> CameraStatus:
        """Get current camera status."""
        return self._status

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _connect(self) -> bool:
        """Open VideoCapture connection."""
        try:
            if self.source_type == SourceType.WEBCAM:
                source = int(self.source)
            else:
                source = self.source

            self._cap = cv2.VideoCapture(source)

            # Set buffer size for RTSP (reduce latency)
            if self.source_type == SourceType.RTSP:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._cap.isOpened():
                self._status.error = "Failed to open stream"
                self._status.is_connected = False
                logger.error(f"Camera {self.camera_id}: Failed to open {source}")
                return False

            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._status.resolution = (w, h)
            self._status.is_connected = True
            self._status.error = ""
            logger.info(f"Camera {self.camera_id}: Connected ({w}x{h})")
            return True

        except Exception as e:
            self._status.error = str(e)
            self._status.is_connected = False
            logger.error(f"Camera {self.camera_id}: Connection error: {e}")
            return False

    def _disconnect(self) -> None:
        """Release VideoCapture."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._status.is_connected = False

    def _read_loop(self) -> None:
        """Main read loop running in a dedicated thread."""
        frame_interval = 1.0 / self.target_fps
        reconnect_attempts = 0

        while self._running:
            t_start = time.time()

            if not self._cap or not self._cap.isOpened():
                # Attempt reconnection
                if reconnect_attempts >= self.max_reconnects:
                    logger.error(
                        f"Camera {self.camera_id}: Max reconnection attempts reached"
                    )
                    self._status.error = "Max reconnections exceeded"
                    self._running = False
                    break

                reconnect_attempts += 1
                self._status.reconnect_count += 1
                logger.info(
                    f"Camera {self.camera_id}: Reconnecting "
                    f"(attempt {reconnect_attempts}/{self.max_reconnects})"
                )
                self._disconnect()
                time.sleep(self.reconnect_interval)
                if self._connect():
                    reconnect_attempts = 0
                continue

            ret, frame = self._cap.read()

            if not ret or frame is None:
                if self.source_type == SourceType.FILE:
                    # Loop video files
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Stream error — trigger reconnection
                    self._status.is_connected = False
                    self._status.error = "Frame read failed"
                    logger.warning(f"Camera {self.camera_id}: Frame read failed")
                    continue

            # Update latest frame
            with self._lock:
                self._latest_frame = frame

            self._status.frame_count += 1
            self._status.last_frame_time = time.time()
            self._status.is_connected = True
            self._status.error = ""
            reconnect_attempts = 0

            # Frame rate limiting
            elapsed = time.time() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Update actual FPS
            actual_elapsed = time.time() - t_start
            self._status.fps_actual = 1.0 / actual_elapsed if actual_elapsed > 0 else 0.0


class StreamManager:
    """Manages multiple camera streams.

    Provides a unified interface to add/remove cameras and retrieve frames.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._cameras: dict[str, CameraStream] = {}
        self._lock = threading.Lock()

    def add_camera(
        self,
        camera_id: str,
        source: str,
        target_fps: int | None = None,
    ) -> bool:
        """Add and start a new camera stream.

        Args:
            camera_id: Unique identifier for the camera.
            source: RTSP URL, file path, or webcam index (as string).
            target_fps: Override default target FPS.

        Returns:
            True if camera was added and started successfully.
        """
        with self._lock:
            if camera_id in self._cameras:
                logger.warning(f"Camera {camera_id} already exists, removing first")
                self._cameras[camera_id].stop()

            stream = CameraStream(
                camera_id=camera_id,
                source=source,
                target_fps=target_fps or self.config.target_fps,
                reconnect_interval=self.config.reconnect_interval,
                max_reconnects=self.config.max_reconnect_attempts,
            )

            success = stream.start()
            if success:
                self._cameras[camera_id] = stream
            return success

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera stream."""
        with self._lock:
            stream = self._cameras.pop(camera_id, None)
            if stream:
                stream.stop()
                return True
            return False

    def get_frame(self, camera_id: str) -> tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame from a specific camera."""
        stream = self._cameras.get(camera_id)
        if not stream:
            return False, None
        return stream.get_frame()

    def get_all_frames(self) -> dict[str, Optional[np.ndarray]]:
        """Get the latest frame from all cameras.

        Returns:
            Dict mapping camera_id to frame (or None if unavailable).
        """
        frames = {}
        for cam_id, stream in self._cameras.items():
            success, frame = stream.get_frame()
            if success:
                frames[cam_id] = frame
        return frames

    def get_status(self) -> dict[str, CameraStatus]:
        """Get status of all cameras."""
        return {cam_id: stream.get_status() for cam_id, stream in self._cameras.items()}

    def get_camera_ids(self) -> list[str]:
        """Return list of all camera IDs."""
        return list(self._cameras.keys())

    def stop_all(self) -> None:
        """Stop all camera streams."""
        with self._lock:
            for stream in self._cameras.values():
                stream.stop()
            self._cameras.clear()
        logger.info("All camera streams stopped")

    @property
    def camera_count(self) -> int:
        return len(self._cameras)
