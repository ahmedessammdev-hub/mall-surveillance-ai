"""
ByteTrack Multi-Person Tracker.

Maintains persistent track IDs across frames with trajectory history
and lifecycle management (active → lost → removed).
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ai.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """A single point in a track's trajectory."""
    x: float
    y: float
    timestamp: float
    frame_id: int = 0


@dataclass
class TrackedPerson:
    """A tracked person with trajectory history and metadata."""
    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    center: tuple[float, float] = field(init=False)
    status: str = "active"  # active, lost, removed
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_frames: int = 0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "center": list(self.center),
            "status": self.status,
            "total_frames": self.total_frames,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class TrackHistory:
    """Complete history for a single track."""
    track_id: int
    trajectory: list[TrajectoryPoint]
    first_seen: float
    last_seen: float
    total_frames: int
    status: str
    avg_speed: float = 0.0
    max_speed: float = 0.0


class MultiPersonTracker:
    """Multi-person tracker using ByteTrack via Ultralytics.

    Wraps the Ultralytics tracking interface and maintains additional
    trajectory history and lifecycle state per track.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._frame_count = 0
        self._track_histories: dict[int, list[TrajectoryPoint]] = defaultdict(list)
        self._track_metadata: dict[int, dict] = {}
        self._active_track_ids: set[int] = set()
        self._lost_track_ids: set[int] = set()
        self._max_traj_len = config.max_trajectory_length

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        """Update tracker with new detections and return tracked persons.

        This method processes raw detections through ByteTrack logic to assign
        persistent IDs. Since we use the Ultralytics built-in tracker in the
        main pipeline via model.track(), here we provide a simplified matching
        based on IoU for standalone usage.

        Args:
            detections: Person detections for the current frame.

        Returns:
            List of TrackedPerson with assigned track IDs.
        """
        self._frame_count += 1
        now = time.time()
        tracked: list[TrackedPerson] = []

        if not detections:
            # Mark all active tracks as potentially lost
            for tid in list(self._active_track_ids):
                meta = self._track_metadata.get(tid, {})
                frames_since = self._frame_count - meta.get("last_frame", 0)
                if frames_since > self.config.track_buffer:
                    self._active_track_ids.discard(tid)
                    self._lost_track_ids.add(tid)
            return tracked

        # Simple IoU-based matching with existing tracks
        det_bboxes = np.array([d.bbox for d in detections])
        matched_det_indices: set[int] = set()
        matched_track_ids: set[int] = set()

        # Try to match detections to existing tracks
        for tid in list(self._active_track_ids):
            if tid not in self._track_metadata:
                continue
            prev_bbox = self._track_metadata[tid].get("bbox")
            if prev_bbox is None:
                continue

            best_iou = 0.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue
                iou = self._compute_iou(prev_bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self.config.match_thresh and best_idx >= 0:
                matched_det_indices.add(best_idx)
                matched_track_ids.add(tid)
                det = detections[best_idx]

                # Update trajectory
                point = TrajectoryPoint(
                    x=det.center[0], y=det.center[1],
                    timestamp=now, frame_id=self._frame_count,
                )
                self._track_histories[tid].append(point)
                if len(self._track_histories[tid]) > self._max_traj_len:
                    self._track_histories[tid] = self._track_histories[tid][-self._max_traj_len:]

                self._track_metadata[tid].update({
                    "bbox": det.bbox,
                    "last_frame": self._frame_count,
                    "last_seen": now,
                    "total_frames": self._track_metadata[tid].get("total_frames", 0) + 1,
                })

                person = TrackedPerson(
                    track_id=tid,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    status="active",
                    trajectory=list(self._track_histories[tid]),
                    first_seen=self._track_metadata[tid].get("first_seen", now),
                    last_seen=now,
                    total_frames=self._track_metadata[tid]["total_frames"],
                )
                tracked.append(person)

        # Also try matching with lost tracks (re-identification)
        for tid in list(self._lost_track_ids):
            if tid not in self._track_metadata:
                continue
            prev_bbox = self._track_metadata[tid].get("bbox")
            if prev_bbox is None:
                continue

            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue
                iou = self._compute_iou(prev_bbox, det.bbox)
                if iou >= self.config.match_thresh * 0.7:  # Lower threshold for re-id
                    matched_det_indices.add(i)
                    self._lost_track_ids.discard(tid)
                    self._active_track_ids.add(tid)

                    point = TrajectoryPoint(
                        x=det.center[0], y=det.center[1],
                        timestamp=now, frame_id=self._frame_count,
                    )
                    self._track_histories[tid].append(point)

                    self._track_metadata[tid].update({
                        "bbox": det.bbox,
                        "last_frame": self._frame_count,
                        "last_seen": now,
                        "total_frames": self._track_metadata[tid].get("total_frames", 0) + 1,
                    })

                    person = TrackedPerson(
                        track_id=tid,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        status="active",
                        trajectory=list(self._track_histories[tid]),
                        first_seen=self._track_metadata[tid].get("first_seen", now),
                        last_seen=now,
                        total_frames=self._track_metadata[tid]["total_frames"],
                    )
                    tracked.append(person)
                    break

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in matched_det_indices:
                continue
            tid = self._next_track_id()
            self._active_track_ids.add(tid)

            point = TrajectoryPoint(
                x=det.center[0], y=det.center[1],
                timestamp=now, frame_id=self._frame_count,
            )
            self._track_histories[tid] = [point]
            self._track_metadata[tid] = {
                "bbox": det.bbox,
                "first_seen": now,
                "last_seen": now,
                "last_frame": self._frame_count,
                "total_frames": 1,
            }

            person = TrackedPerson(
                track_id=tid,
                bbox=det.bbox,
                confidence=det.confidence,
                status="active",
                trajectory=[point],
                first_seen=now,
                last_seen=now,
                total_frames=1,
            )
            tracked.append(person)

        # Age out lost tracks
        for tid in list(self._active_track_ids):
            if tid in matched_track_ids:
                continue
            if tid not in self._track_metadata:
                continue
            frames_since = self._frame_count - self._track_metadata[tid].get("last_frame", 0)
            if frames_since > self.config.track_buffer:
                self._active_track_ids.discard(tid)
                self._lost_track_ids.add(tid)

        return tracked

    def get_track_history(self, track_id: int) -> TrackHistory | None:
        """Get complete history for a specific track."""
        if track_id not in self._track_metadata:
            return None
        meta = self._track_metadata[track_id]
        trajectory = self._track_histories.get(track_id, [])

        # Compute speeds
        speeds = self._compute_speeds(trajectory)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0

        status = "active" if track_id in self._active_track_ids else "lost"
        return TrackHistory(
            track_id=track_id,
            trajectory=trajectory,
            first_seen=meta.get("first_seen", 0),
            last_seen=meta.get("last_seen", 0),
            total_frames=meta.get("total_frames", 0),
            status=status,
            avg_speed=avg_speed,
            max_speed=max_speed,
        )

    def get_active_tracks(self) -> list[int]:
        """Return list of currently active track IDs."""
        return list(self._active_track_ids)

    def get_all_trajectories(self) -> dict[int, list[TrajectoryPoint]]:
        """Return all trajectory histories."""
        return dict(self._track_histories)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    _next_id_counter = 0

    def _next_track_id(self) -> int:
        MultiPersonTracker._next_id_counter += 1
        return MultiPersonTracker._next_id_counter

    @staticmethod
    def _compute_iou(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def _compute_speeds(trajectory: list[TrajectoryPoint]) -> list[float]:
        """Compute speeds between consecutive trajectory points (px/sec)."""
        speeds: list[float] = []
        for i in range(1, len(trajectory)):
            p0, p1 = trajectory[i - 1], trajectory[i]
            dt = p1.timestamp - p0.timestamp
            if dt <= 0:
                continue
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            speeds.append(dist / dt)
        return speeds
