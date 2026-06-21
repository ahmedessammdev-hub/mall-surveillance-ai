"""
Feature Extractor for tracked persons.

Extracts motion, behavior, and interaction features from tracking data
for use in event detection and embedding generation.

Improvements over v1:
- KDTree-based O(n·log n) proximity computation
- EMA temporal smoothing for speed/acceleration/direction
- Proper dwell time tracking with continuous state
- Resolution normalization (thresholds relative to 720p)
- Bbox aspect ratio + posture estimation
- Zone-based features
- Movement trend features
- Incremental path length accumulation
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ai.tracker import TrackedPerson, TrajectoryPoint

logger = logging.getLogger(__name__)

# 720p reference resolution
BASE_HEIGHT = 720
BASE_WIDTH = 1280


@dataclass
class Zone:
    """A named polygonal zone in the camera frame."""
    name: str
    polygon: list[tuple[float, float]]  # list of (x, y) vertices

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon test using ray casting."""
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


@dataclass
class PersonFeatures:
    """Extracted features for a single tracked person."""
    track_id: int
    speed: float = 0.0  # pixels/second (EMA smoothed)
    acceleration: float = 0.0  # pixels/second² (EMA smoothed)
    direction: float = 0.0  # radians (EMA smoothed)
    direction_changes: int = 0  # number of direction changes in recent window
    dwell_time: float = 0.0  # continuous seconds in current area
    displacement: float = 0.0  # total displacement from first to current position
    path_length: float = 0.0  # total path length traveled
    straightness: float = 0.0  # displacement / path_length (1.0 = straight line)
    vertical_displacement: float = 0.0  # vertical change (for fall detection)
    speed_variance: float = 0.0  # variance in speed (erratic motion indicator)

    # Bbox / posture features
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    bbox_aspect_ratio: float = 0.0  # width / height
    posture_state: str = "unknown"  # standing, sitting, fallen, unknown
    posture_confidence: float = 0.0

    # Proximity / interaction features
    nearest_person_distance: float = float("inf")
    nearest_person_id: int = -1
    people_within_radius: int = 0  # count of people within interaction radius
    interaction_duration: float = 0.0  # seconds of sustained proximity
    proximity_score: float = 0.0  # 0-1, higher = closer interactions

    # Zone features
    current_zone: str = ""
    is_in_restricted_zone: bool = False

    # Movement trends
    speed_trend: float = 0.0  # EMA of acceleration (positive = speeding up)
    direction_trend: float = 0.0  # EMA of direction change rate

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "speed": round(self.speed, 2),
            "acceleration": round(self.acceleration, 2),
            "direction": round(self.direction, 4),
            "direction_changes": self.direction_changes,
            "dwell_time": round(self.dwell_time, 2),
            "displacement": round(self.displacement, 2),
            "path_length": round(self.path_length, 2),
            "straightness": round(self.straightness, 4),
            "vertical_displacement": round(self.vertical_displacement, 2),
            "speed_variance": round(self.speed_variance, 4),
            "bbox_width": round(self.bbox_width, 2),
            "bbox_height": round(self.bbox_height, 2),
            "bbox_aspect_ratio": round(self.bbox_aspect_ratio, 4),
            "posture_state": self.posture_state,
            "posture_confidence": round(self.posture_confidence, 4),
            "nearest_person_distance": round(self.nearest_person_distance, 2),
            "nearest_person_id": self.nearest_person_id,
            "people_within_radius": self.people_within_radius,
            "interaction_duration": round(self.interaction_duration, 2),
            "proximity_score": round(self.proximity_score, 4),
            "current_zone": self.current_zone,
            "is_in_restricted_zone": self.is_in_restricted_zone,
            "speed_trend": round(self.speed_trend, 2),
            "direction_trend": round(self.direction_trend, 4),
        }


@dataclass
class SceneFeatures:
    """Scene-level aggregated features."""
    person_count: int = 0
    crowd_density: float = 0.0  # people per unit area
    avg_speed: float = 0.0
    max_speed: float = 0.0
    avg_acceleration: float = 0.0
    motion_energy: float = 0.0  # sum of all speeds squared
    motion_dispersion: float = 0.0  # variance of motion directions (panic indicator)
    avg_nearest_distance: float = 0.0
    min_nearest_distance: float = float("inf")
    interaction_count: int = 0  # pairs within interaction radius
    fallen_count: int = 0  # number of people in fallen posture
    restricted_zone_violations: int = 0

    def to_dict(self) -> dict:
        return {
            "person_count": self.person_count,
            "crowd_density": round(self.crowd_density, 6),
            "avg_speed": round(self.avg_speed, 2),
            "max_speed": round(self.max_speed, 2),
            "avg_acceleration": round(self.avg_acceleration, 2),
            "motion_energy": round(self.motion_energy, 2),
            "motion_dispersion": round(self.motion_dispersion, 4),
            "avg_nearest_distance": round(self.avg_nearest_distance, 2),
            "min_nearest_distance": round(self.min_nearest_distance, 2),
            "interaction_count": self.interaction_count,
            "fallen_count": self.fallen_count,
            "restricted_zone_violations": self.restricted_zone_violations,
        }


class FeatureExtractor:
    """Extracts motion, behavior, and interaction features from tracked persons.

    Uses KDTree for efficient proximity computation, EMA for temporal smoothing,
    and resolution normalization for cross-camera consistency.
    """

    # Base thresholds (scaled to 720p via normalization)
    INTERACTION_RADIUS = 120.0  # pixels at 720p
    DIRECTION_CHANGE_WINDOW = 30  # trajectory points
    DWELL_RADIUS = 50.0  # pixels at 720p

    def __init__(
        self,
        ema_alpha: float = 0.3,
        zones: Optional[list[dict]] = None,
        restricted_zones: Optional[list[str]] = None,
    ) -> None:
        self._ema_alpha = ema_alpha

        # Zone configuration
        self._zones: list[Zone] = []
        if zones:
            for z in zones:
                self._zones.append(Zone(
                    name=z["name"],
                    polygon=[(p[0], p[1]) for p in z["polygon"]],
                ))
        self._restricted_zone_names: set[str] = set(restricted_zones or [])

        # Per-track state
        self._prev_speeds: dict[int, list[float]] = {}
        self._prev_speed_times: dict[int, list[float]] = {}
        self._smoothed_speed: dict[int, float] = {}
        self._smoothed_accel: dict[int, float] = {}
        self._smoothed_direction: dict[int, float] = {}
        self._speed_trend: dict[int, float] = {}
        self._direction_trend: dict[int, float] = {}
        self._path_lengths: dict[int, float] = {}
        self._aspect_ratios: dict[int, list[float]] = {}

        # Dwell tracking
        self._dwell_start: dict[int, float] = {}
        self._dwell_anchor: dict[int, tuple[float, float]] = {}

        # Interaction tracking
        self._interaction_start: dict[tuple[int, int], float] = {}
        self._interaction_history: dict[tuple[int, int], float] = {}

    def extract(
        self,
        tracks: list[TrackedPerson],
        frame_shape: tuple[int, int],
    ) -> dict[str, object]:
        """Extract per-person and scene-level features.

        Args:
            tracks: List of tracked persons from the current frame.
            frame_shape: (height, width) of the frame.

        Returns:
            Dict with keys "persons" (dict[int, PersonFeatures]) and
            "scene" (SceneFeatures).
        """
        person_features: dict[int, PersonFeatures] = {}

        for track in tracks:
            pf = self._extract_person_features(track, frame_shape)
            person_features[track.track_id] = pf

        self._compute_interactions(tracks, person_features, frame_shape)
        scene = self._compute_scene_features(person_features, frame_shape)

        return {
            "persons": person_features,
            "scene": scene,
        }

    def _normalize(self, value: float, frame_shape: tuple[int, int]) -> float:
        """Normalize a pixel value to 720p base resolution."""
        h, w = frame_shape
        scale = BASE_HEIGHT / h
        return value * scale

    def _ema(self, prev: float, new: float, alpha: float = 0.0) -> float:
        """Exponential moving average."""
        a = alpha if alpha > 0 else self._ema_alpha
        return a * new + (1 - a) * prev

    def _extract_person_features(
        self,
        track: TrackedPerson,
        frame_shape: tuple[int, int],
    ) -> PersonFeatures:
        """Extract motion features for a single person."""
        pf = PersonFeatures(track_id=track.track_id)
        traj = track.trajectory
        tid = track.track_id
        now = time.time()

        # Bbox features
        x1, y1, x2, y2 = track.bbox
        pf.bbox_width = x2 - x1
        pf.bbox_height = y2 - y1
        pf.bbox_aspect_ratio = pf.bbox_width / pf.bbox_height if pf.bbox_height > 0 else 0.0

        # Track aspect ratio history for posture estimation
        self._aspect_ratios.setdefault(tid, [])
        self._aspect_ratios[tid].append(pf.bbox_aspect_ratio)
        if len(self._aspect_ratios[tid]) > 30:
            self._aspect_ratios[tid] = self._aspect_ratios[tid][-30:]

        # Posture estimation from aspect ratio
        pf.posture_state, pf.posture_confidence = self._estimate_posture(tid)

        if len(traj) < 2:
            pf.dwell_time = now - track.first_seen
            return pf

        # --- Speed computation ---
        speeds = []
        speed_times = []
        for i in range(1, len(traj)):
            dt = traj[i].timestamp - traj[i - 1].timestamp
            if dt <= 0:
                continue
            dx = traj[i].x - traj[i - 1].x
            dy = traj[i].y - traj[i - 1].y
            speed = math.sqrt(dx ** 2 + dy ** 2) / dt
            speeds.append(speed)
            speed_times.append(traj[i].timestamp)

        if speeds:
            raw_speed = speeds[-1]
            pf.speed = self._ema(
                self._smoothed_speed.get(tid, raw_speed),
                raw_speed,
            )
            self._smoothed_speed[tid] = pf.speed

            pf.speed_variance = float(np.var(speeds[-20:])) if len(speeds) > 1 else 0.0

            # Store speed history for acceleration
            self._prev_speeds.setdefault(tid, [])
            self._prev_speed_times.setdefault(tid, [])
            self._prev_speeds[tid].append(pf.speed)
            self._prev_speed_times[tid].append(speed_times[-1])
            if len(self._prev_speeds[tid]) > 30:
                self._prev_speeds[tid] = self._prev_speeds[tid][-30:]
                self._prev_speed_times[tid] = self._prev_speed_times[tid][-30:]

        # --- Acceleration (proper px/sec²) ---
        speed_hist = self._prev_speeds.get(tid, [])
        time_hist = self._prev_speed_times.get(tid, [])
        if len(speed_hist) >= 2:
            dt = time_hist[-1] - time_hist[-2]
            if dt > 0:
                raw_accel = (speed_hist[-1] - speed_hist[-2]) / dt
                pf.acceleration = self._ema(
                    self._smoothed_accel.get(tid, raw_accel),
                    raw_accel,
                )
                self._smoothed_accel[tid] = pf.acceleration

                # Speed trend (EMA of acceleration)
                self._speed_trend[tid] = self._ema(
                    self._speed_trend.get(tid, 0.0),
                    pf.acceleration,
                    alpha=0.2,
                )
                pf.speed_trend = self._speed_trend[tid]

        # --- Direction (EMA smoothed) ---
        if len(traj) >= 2:
            dx = traj[-1].x - traj[-2].x
            dy = traj[-1].y - traj[-2].y
            raw_direction = math.atan2(dy, dx)
            pf.direction = self._ema(
                self._smoothed_direction.get(tid, raw_direction),
                raw_direction,
            )
            self._smoothed_direction[tid] = pf.direction

        # --- Direction changes ---
        if len(traj) >= 3:
            window = traj[-self.DIRECTION_CHANGE_WINDOW:]
            changes = 0
            direction_deltas = []
            for i in range(2, len(window)):
                d1 = math.atan2(
                    window[i - 1].y - window[i - 2].y,
                    window[i - 1].x - window[i - 2].x,
                )
                d2 = math.atan2(
                    window[i].y - window[i - 1].y,
                    window[i].x - window[i - 1].x,
                )
                angle_diff = abs(d2 - d1)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                direction_deltas.append(angle_diff)
                if angle_diff > math.pi / 4:
                    changes += 1
            pf.direction_changes = changes

            # Direction trend (EMA of direction change rate)
            if direction_deltas:
                avg_delta = float(np.mean(direction_deltas))
                self._direction_trend[tid] = self._ema(
                    self._direction_trend.get(tid, 0.0),
                    avg_delta,
                    alpha=0.2,
                )
                pf.direction_trend = self._direction_trend[tid]

        # --- Displacement & incremental path length ---
        pf.displacement = math.sqrt(
            (traj[-1].x - traj[0].x) ** 2 + (traj[-1].y - traj[0].y) ** 2
        )

        # Incremental path length
        prev_len = self._path_lengths.get(tid, 0.0)
        if len(traj) >= 2:
            dx = traj[-1].x - traj[-2].x
            dy = traj[-1].y - traj[-2].y
            segment = math.sqrt(dx ** 2 + dy ** 2)
            self._path_lengths[tid] = prev_len + segment
        pf.path_length = self._path_lengths.get(tid, 0.0)
        pf.straightness = pf.displacement / pf.path_length if pf.path_length > 0 else 0.0

        # --- Vertical displacement (for fall detection) ---
        if len(traj) >= 5:
            recent_window = traj[-5:]
            pf.vertical_displacement = recent_window[-1].y - recent_window[0].y

        # --- Dwell time (proper continuous tracking) ---
        dwell_radius_norm = self._normalize(self.DWELL_RADIUS, frame_shape)
        anchor = self._dwell_anchor.get(tid)

        if anchor is None:
            # First time — set anchor
            self._dwell_anchor[tid] = (traj[-1].x, traj[-1].y)
            self._dwell_start[tid] = traj[0].timestamp
            pf.dwell_time = 0.0
        else:
            dist_from_anchor = math.sqrt(
                (traj[-1].x - anchor[0]) ** 2 + (traj[-1].y - anchor[1]) ** 2
            )
            if dist_from_anchor < dwell_radius_norm:
                # Still dwelling — accumulate time
                pf.dwell_time = now - self._dwell_start[tid]
            else:
                # Moved away — reset dwell
                self._dwell_anchor[tid] = (traj[-1].x, traj[-1].y)
                self._dwell_start[tid] = now
                pf.dwell_time = 0.0

        # --- Zone features ---
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        for zone in self._zones:
            if zone.contains(center_x, center_y):
                pf.current_zone = zone.name
                if zone.name in self._restricted_zone_names:
                    pf.is_in_restricted_zone = True
                break

        return pf

    def _estimate_posture(self, tid: int) -> tuple[str, float]:
        """Estimate posture from bbox aspect ratio history."""
        ratios = self._aspect_ratios.get(tid, [])
        if len(ratios) < 3:
            return "unknown", 0.0

        recent = ratios[-5:]
        avg_ratio = float(np.mean(recent))
        ratio_variance = float(np.var(recent))

        # Heuristic thresholds for surveillance camera perspective
        if avg_ratio > 1.5:
            return "standing", min(1.0, avg_ratio / 2.0)
        elif avg_ratio > 1.0:
            # Could be sitting or crouching — check if ratio is changing
            if ratio_variance > 0.1:
                return "unknown", 0.3  # Transitioning
            return "sitting", 0.6
        else:
            # Very wide bbox = person horizontal
            return "fallen", min(1.0, (1.0 - avg_ratio) * 2.0)

    def _compute_interactions(
        self,
        tracks: list[TrackedPerson],
        features: dict[int, PersonFeatures],
        frame_shape: tuple[int, int],
    ) -> None:
        """Compute pairwise proximity and interaction features using KDTree."""
        if len(tracks) < 2:
            if tracks:
                pf = features.get(tracks[0].track_id)
                if pf:
                    pf.nearest_person_distance = float("inf")
                    pf.nearest_person_id = -1
            return

        now = time.time()
        interaction_radius_norm = self._normalize(self.INTERACTION_RADIUS, frame_shape)

        # Build KDTree from track centers
        centers = np.array([t.center for t in tracks])
        tree = cKDTree(centers)

        # Query all pairs within interaction radius
        pairs = tree.query_pairs(r=interaction_radius_norm)

        # Build per-track nearest neighbor and nearby counts
        nearest_dist: dict[int, float] = {t.track_id: float("inf") for t in tracks}
        nearest_id: dict[int, int] = {t.track_id: -1 for t in tracks}
        nearby_count: dict[int, int] = {t.track_id: 0 for t in tracks}

        # Also compute exact nearest neighbor for each track
        for i, t in enumerate(tracks):
            tid = t.track_id
            # Query k=2 (first is the point itself)
            dists, idxs = tree.query(centers[i], k=min(2, len(tracks)))
            if len(dists) > 1:
                nearest_dist[tid] = dists[1]
                nearest_id[tid] = tracks[idxs[1]].track_id

        # Process interaction pairs
        active_pairs: set[tuple[int, int]] = set()
        for i, j in pairs:
            tid_i = tracks[i].track_id
            tid_j = tracks[j].track_id
            pair = (min(tid_i, tid_j), max(tid_i, tid_j))
            active_pairs.add(pair)

            dist = math.sqrt(
                (tracks[i].center[0] - tracks[j].center[0]) ** 2
                + (tracks[i].center[1] - tracks[j].center[1]) ** 2
            )

            nearby_count[tid_i] += 1
            nearby_count[tid_j] += 1

            # Track interaction duration
            if pair not in self._interaction_start:
                self._interaction_start[pair] = now

        # Update person features
        for t in tracks:
            tid = t.track_id
            pf = features.get(tid)
            if not pf:
                continue
            pf.nearest_person_distance = nearest_dist[tid]
            pf.nearest_person_id = nearest_id[tid]
            pf.people_within_radius = nearby_count[tid]

            if pf.nearest_person_distance < float("inf"):
                pf.proximity_score = max(
                    0, 1.0 - pf.nearest_person_distance / (interaction_radius_norm * 2)
                )
            else:
                pf.proximity_score = 0.0

        # Compute interaction durations
        for pair in active_pairs:
            start = self._interaction_start.get(pair, now)
            duration = now - start
            tid1, tid2 = pair
            if tid1 in features:
                features[tid1].interaction_duration = max(
                    features[tid1].interaction_duration, duration
                )
            if tid2 in features:
                features[tid2].interaction_duration = max(
                    features[tid2].interaction_duration, duration
                )

        # Move ended interactions to history (keep for dedup window)
        for pair in list(self._interaction_start.keys()):
            if pair not in active_pairs:
                self._interaction_history[pair] = self._interaction_start.pop(pair)

        # Clean old history entries (keep for 2x dedup window)
        cutoff = now - 30.0  # 30 seconds retention
        self._interaction_history = {
            k: v for k, v in self._interaction_history.items() if v > cutoff
        }

    def _compute_scene_features(
        self,
        features: dict[int, PersonFeatures],
        frame_shape: tuple[int, int],
    ) -> SceneFeatures:
        """Compute scene-level aggregated features."""
        scene = SceneFeatures()
        scene.person_count = len(features)

        if scene.person_count == 0:
            return scene

        h, w = frame_shape
        frame_area = h * w
        scene.crowd_density = scene.person_count / frame_area if frame_area > 0 else 0.0

        speeds = [pf.speed for pf in features.values()]
        accels = [pf.acceleration for pf in features.values()]
        distances = [
            pf.nearest_person_distance
            for pf in features.values()
            if pf.nearest_person_distance < float("inf")
        ]
        directions = [pf.direction for pf in features.values()]

        scene.avg_speed = float(np.mean(speeds)) if speeds else 0.0
        scene.max_speed = float(np.max(speeds)) if speeds else 0.0
        scene.avg_acceleration = float(np.mean(accels)) if accels else 0.0
        scene.motion_energy = float(np.sum(np.array(speeds) ** 2)) if speeds else 0.0

        if distances:
            scene.avg_nearest_distance = float(np.mean(distances))
            scene.min_nearest_distance = float(np.min(distances))

        # Motion dispersion: circular variance of directions
        if len(directions) > 1:
            sin_sum = sum(math.sin(d) for d in directions)
            cos_sum = sum(math.cos(d) for d in directions)
            r_bar = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(directions)
            scene.motion_dispersion = 1.0 - r_bar

        # Count interaction pairs
        scene.interaction_count = sum(
            1 for pf in features.values() if pf.people_within_radius > 0
        ) // 2

        # Count fallen persons
        scene.fallen_count = sum(
            1 for pf in features.values() if pf.posture_state == "fallen"
        )

        # Count restricted zone violations
        scene.restricted_zone_violations = sum(
            1 for pf in features.values() if pf.is_in_restricted_zone
        )

        return scene

    def cleanup_track(self, track_id: int) -> None:
        """Remove all state for a track that has been lost/removed."""
        for store in (
            self._prev_speeds,
            self._prev_speed_times,
            self._smoothed_speed,
            self._smoothed_accel,
            self._smoothed_direction,
            self._speed_trend,
            self._direction_trend,
            self._path_lengths,
            self._aspect_ratios,
            self._dwell_start,
            self._dwell_anchor,
        ):
            store.pop(track_id, None)
