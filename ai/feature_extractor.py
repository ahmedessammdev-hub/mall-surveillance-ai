"""
Feature Extractor for tracked persons.

Extracts motion, behavior, and interaction features from tracking data
for use in event detection and embedding generation.
"""

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ai.tracker import TrackedPerson, TrajectoryPoint

logger = logging.getLogger(__name__)


@dataclass
class PersonFeatures:
    """Extracted features for a single tracked person."""
    track_id: int
    speed: float = 0.0  # pixels/second
    acceleration: float = 0.0  # pixels/second²
    direction: float = 0.0  # radians (0 = right, π/2 = down)
    direction_changes: int = 0  # number of direction changes in recent window
    dwell_time: float = 0.0  # seconds in current area
    displacement: float = 0.0  # total displacement from first to current position
    path_length: float = 0.0  # total path length traveled
    straightness: float = 0.0  # displacement / path_length (1.0 = straight line)
    vertical_displacement: float = 0.0  # vertical change (for fall detection)
    speed_variance: float = 0.0  # variance in speed (erratic motion indicator)

    # Proximity / interaction features
    nearest_person_distance: float = float("inf")
    nearest_person_id: int = -1
    people_within_radius: int = 0  # count of people within interaction radius
    interaction_duration: float = 0.0  # seconds of sustained proximity
    proximity_score: float = 0.0  # 0-1, higher = closer interactions

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
            "nearest_person_distance": round(self.nearest_person_distance, 2),
            "nearest_person_id": self.nearest_person_id,
            "people_within_radius": self.people_within_radius,
            "interaction_duration": round(self.interaction_duration, 2),
            "proximity_score": round(self.proximity_score, 4),
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
        }


class FeatureExtractor:
    """Extracts motion, behavior, and interaction features from tracked persons.

    Maintains a small state of recent speeds/positions per track for computing
    acceleration, direction changes, dwell time, and interaction durations.
    """

    INTERACTION_RADIUS = 120.0  # pixels — threshold for "interacting"
    DIRECTION_CHANGE_WINDOW = 30  # trajectory points to consider
    DWELL_RADIUS = 50.0  # pixels — if displacement < this, person is "dwelling"

    def __init__(self) -> None:
        # Per-track state for acceleration/interaction tracking
        self._prev_speeds: dict[int, list[float]] = {}
        self._interaction_start: dict[tuple[int, int], float] = {}

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

        # --- Per-person feature extraction ---
        for track in tracks:
            pf = self._extract_person_features(track)
            person_features[track.track_id] = pf

        # --- Pairwise interaction features ---
        self._compute_interactions(tracks, person_features)

        # --- Scene-level features ---
        scene = self._compute_scene_features(person_features, frame_shape)

        return {
            "persons": person_features,
            "scene": scene,
        }

    def _extract_person_features(self, track: TrackedPerson) -> PersonFeatures:
        """Extract motion features for a single person."""
        pf = PersonFeatures(track_id=track.track_id)
        traj = track.trajectory

        if len(traj) < 2:
            pf.dwell_time = time.time() - track.first_seen
            return pf

        # --- Speed ---
        speeds = []
        for i in range(1, len(traj)):
            dt = traj[i].timestamp - traj[i - 1].timestamp
            if dt <= 0:
                continue
            dx = traj[i].x - traj[i - 1].x
            dy = traj[i].y - traj[i - 1].y
            speed = math.sqrt(dx ** 2 + dy ** 2) / dt
            speeds.append(speed)

        if speeds:
            pf.speed = speeds[-1]
            pf.speed_variance = float(np.var(speeds[-20:])) if len(speeds) > 1 else 0.0

            # Track speed history for acceleration
            self._prev_speeds.setdefault(track.track_id, [])
            self._prev_speeds[track.track_id].append(pf.speed)
            if len(self._prev_speeds[track.track_id]) > 30:
                self._prev_speeds[track.track_id] = self._prev_speeds[track.track_id][-30:]

        # --- Acceleration ---
        speed_hist = self._prev_speeds.get(track.track_id, [])
        if len(speed_hist) >= 2:
            # Use last two speed measurements
            pf.acceleration = speed_hist[-1] - speed_hist[-2]

        # --- Direction ---
        if len(traj) >= 2:
            dx = traj[-1].x - traj[-2].x
            dy = traj[-1].y - traj[-2].y
            pf.direction = math.atan2(dy, dx)

        # --- Direction changes ---
        if len(traj) >= 3:
            window = traj[-self.DIRECTION_CHANGE_WINDOW:]
            changes = 0
            for i in range(2, len(window)):
                d1 = math.atan2(window[i - 1].y - window[i - 2].y, window[i - 1].x - window[i - 2].x)
                d2 = math.atan2(window[i].y - window[i - 1].y, window[i].x - window[i - 1].x)
                angle_diff = abs(d2 - d1)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                if angle_diff > math.pi / 4:  # > 45 degrees
                    changes += 1
            pf.direction_changes = changes

        # --- Displacement & path length ---
        pf.displacement = math.sqrt(
            (traj[-1].x - traj[0].x) ** 2 + (traj[-1].y - traj[0].y) ** 2
        )
        path_len = 0.0
        for i in range(1, len(traj)):
            dx = traj[i].x - traj[i - 1].x
            dy = traj[i].y - traj[i - 1].y
            path_len += math.sqrt(dx ** 2 + dy ** 2)
        pf.path_length = path_len
        pf.straightness = pf.displacement / pf.path_length if pf.path_length > 0 else 0.0

        # --- Vertical displacement (for fall detection) ---
        if len(traj) >= 5:
            recent_window = traj[-5:]
            pf.vertical_displacement = recent_window[-1].y - recent_window[0].y

        # --- Dwell time ---
        pf.dwell_time = traj[-1].timestamp - traj[0].timestamp
        if pf.displacement < self.DWELL_RADIUS:
            pf.dwell_time = traj[-1].timestamp - traj[0].timestamp
        else:
            pf.dwell_time = 0.0  # Person is moving, not dwelling

        return pf

    def _compute_interactions(
        self,
        tracks: list[TrackedPerson],
        features: dict[int, PersonFeatures],
    ) -> None:
        """Compute pairwise proximity and interaction features."""
        now = time.time()
        active_pairs: set[tuple[int, int]] = set()

        for i, t1 in enumerate(tracks):
            pf1 = features.get(t1.track_id)
            if not pf1:
                continue

            nearest_dist = float("inf")
            nearest_id = -1
            nearby_count = 0

            for j, t2 in enumerate(tracks):
                if i == j:
                    continue
                dist = math.sqrt(
                    (t1.center[0] - t2.center[0]) ** 2
                    + (t1.center[1] - t2.center[1]) ** 2
                )

                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = t2.track_id

                if dist < self.INTERACTION_RADIUS:
                    nearby_count += 1
                    pair = (min(t1.track_id, t2.track_id), max(t1.track_id, t2.track_id))
                    active_pairs.add(pair)

                    # Track interaction duration
                    if pair not in self._interaction_start:
                        self._interaction_start[pair] = now

            pf1.nearest_person_distance = nearest_dist
            pf1.nearest_person_id = nearest_id
            pf1.people_within_radius = nearby_count

            # Proximity score: 0 = far, 1 = very close
            if nearest_dist < float("inf"):
                pf1.proximity_score = max(0, 1.0 - nearest_dist / (self.INTERACTION_RADIUS * 2))
            else:
                pf1.proximity_score = 0.0

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

        # Clean up old interaction pairs
        for pair in list(self._interaction_start.keys()):
            if pair not in active_pairs:
                del self._interaction_start[pair]

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
            scene.motion_dispersion = 1.0 - r_bar  # 0 = all same direction, 1 = random

        # Count interaction pairs
        scene.interaction_count = sum(
            1 for pf in features.values() if pf.people_within_radius > 0
        ) // 2  # Avoid double-counting

        return scene
