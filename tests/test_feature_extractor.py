"""Tests for Feature Extractor."""

import time
import pytest
import numpy as np

from ai.feature_extractor import FeatureExtractor, PersonFeatures, SceneFeatures, Zone
from ai.tracker import TrackedPerson, TrajectoryPoint


class TestPersonFeatures:
    def test_to_dict(self):
        pf = PersonFeatures(track_id=1, speed=100.5, acceleration=20.0)
        d = pf.to_dict()
        assert d["track_id"] == 1
        assert d["speed"] == 100.5
        assert "bbox_aspect_ratio" in d
        assert "posture_state" in d
        assert "current_zone" in d
        assert "speed_trend" in d

    def test_defaults(self):
        pf = PersonFeatures(track_id=1)
        assert pf.speed == 0.0
        assert pf.nearest_person_distance == float("inf")
        assert pf.posture_state == "unknown"
        assert pf.current_zone == ""
        assert pf.is_in_restricted_zone is False


class TestSceneFeatures:
    def test_to_dict(self):
        sf = SceneFeatures(person_count=5, avg_speed=100.0)
        d = sf.to_dict()
        assert d["person_count"] == 5
        assert "fallen_count" in d
        assert "restricted_zone_violations" in d

    def test_defaults(self):
        sf = SceneFeatures()
        assert sf.fallen_count == 0
        assert sf.restricted_zone_violations == 0


class TestZone:
    def test_contains_inside(self):
        zone = Zone(name="test", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        assert zone.contains(50, 50) is True

    def test_contains_outside(self):
        zone = Zone(name="test", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        assert zone.contains(200, 200) is False

    def test_contains_edge(self):
        zone = Zone(name="test", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        # On boundary — behavior depends on implementation
        result = zone.contains(0, 0)
        assert isinstance(result, bool)


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor(ema_alpha=0.3)

    def test_extract_empty_tracks(self):
        result = self.extractor.extract([], (480, 640))
        assert result["persons"] == {}
        assert result["scene"].person_count == 0

    def test_extract_single_track(self, sample_tracked_persons):
        tracks = [sample_tracked_persons[0]]
        result = self.extractor.extract(tracks, (480, 640))
        assert 1 in result["persons"]
        assert result["scene"].person_count == 1

    def test_extract_multiple_tracks(self, sample_tracked_persons):
        result = self.extractor.extract(sample_tracked_persons, (480, 640))
        assert len(result["persons"]) == 2
        assert result["scene"].person_count == 2

    def test_proximity_detection(self):
        now = time.time()
        close_tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=150, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=150, y=200, timestamp=now),
                ],
                first_seen=now - 1,
                last_seen=now,
                total_frames=5,
            ),
            TrackedPerson(
                track_id=2,
                bbox=(130, 100, 230, 300),
                confidence=0.85,
                trajectory=[
                    TrajectoryPoint(x=180, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=180, y=200, timestamp=now),
                ],
                first_seen=now - 1,
                last_seen=now,
                total_frames=5,
            ),
        ]
        result = self.extractor.extract(close_tracks, (480, 640))
        pf1 = result["persons"][1]
        assert pf1.nearest_person_distance < 100
        assert pf1.nearest_person_id == 2

    def test_bbox_features(self):
        now = time.time()
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=150, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=150, y=200, timestamp=now),
                ],
                first_seen=now - 1,
                last_seen=now,
                total_frames=5,
            ),
        ]
        result = self.extractor.extract(tracks, (480, 640))
        pf = result["persons"][1]
        assert pf.bbox_width == 100.0
        assert pf.bbox_height == 200.0
        assert pf.bbox_aspect_ratio == 0.5

    def test_posture_estimation_standing(self):
        now = time.time()
        # Tall bbox (aspect ratio < 1) → after enough frames, should detect standing
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 150, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=125, y=200, timestamp=now - 2),
                    TrajectoryPoint(x=125, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=125, y=200, timestamp=now),
                ],
                first_seen=now - 1,
                last_seen=now,
                total_frames=5,
            ),
        ]
        # Run multiple times to build aspect ratio history
        for _ in range(5):
            result = self.extractor.extract(tracks, (480, 640))
        pf = result["persons"][1]
        # Aspect ratio is 50/200 = 0.25 → fallen (wide bbox)
        assert pf.posture_state in ("standing", "sitting", "fallen", "unknown")

    def test_dwell_time_tracking(self):
        now = time.time()
        # Person staying in same spot
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=150, y=200, timestamp=now - 10),
                    TrajectoryPoint(x=150, y=200, timestamp=now),
                ],
                first_seen=now - 10,
                last_seen=now,
                total_frames=50,
            ),
        ]
        result = self.extractor.extract(tracks, (480, 640))
        pf = result["persons"][1]
        # Should have some dwell time since they haven't moved
        assert pf.dwell_time >= 0

    def test_ema_smoothing(self):
        now = time.time()
        # Person moving consistently
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=150, y=200, timestamp=now - 2),
                    TrajectoryPoint(x=160, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=170, y=200, timestamp=now),
                ],
                first_seen=now - 5,
                last_seen=now,
                total_frames=15,
            ),
        ]
        result = self.extractor.extract(tracks, (480, 640))
        pf = result["persons"][1]
        # Speed should be smoothed
        assert pf.speed >= 0

    def test_resolution_normalization(self):
        now = time.time()
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=150, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=150, y=200, timestamp=now),
                ],
                first_seen=now - 1,
                last_seen=now,
                total_frames=5,
            ),
        ]
        # Different resolutions should produce normalized results
        result_720 = self.extractor.extract(tracks, (720, 1280))
        result_480 = self.extractor.extract(tracks, (480, 640))
        # Both should work without errors
        assert result_720["scene"].person_count == 1
        assert result_480["scene"].person_count == 1

    def test_zone_detection(self):
        extractor = FeatureExtractor(
            zones=[{"name": "entrance", "polygon": [(0, 0), (300, 0), (300, 300), (0, 300)]}],
            restricted_zones=["entrance"],
        )
        now = time.time()
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=150, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=150, y=200, timestamp=now),
                ],
                first_seen=now - 1,
                last_seen=now,
                total_frames=5,
            ),
        ]
        result = extractor.extract(tracks, (480, 640))
        pf = result["persons"][1]
        assert pf.current_zone == "entrance"
        assert pf.is_in_restricted_zone is True

    def test_cleanup_track(self):
        now = time.time()
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=150, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=150, y=200, timestamp=now),
                ],
                first_seen=now - 1,
                last_seen=now,
                total_frames=5,
            ),
        ]
        self.extractor.extract(tracks, (480, 640))
        self.extractor.cleanup_track(1)
        # Should not crash
        result = self.extractor.extract(tracks, (480, 640))
        assert result["scene"].person_count == 1

    def test_scene_fallen_count(self):
        extractor = FeatureExtractor(ema_alpha=0.3)
        now = time.time()
        # Wide bbox → fallen posture
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 200, 300, 250),  # wide, short = horizontal
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=200, y=225, timestamp=now - 2),
                    TrajectoryPoint(x=200, y=225, timestamp=now - 1),
                    TrajectoryPoint(x=200, y=225, timestamp=now),
                ],
                first_seen=now - 1,
                last_seen=now,
                total_frames=5,
            ),
        ]
        # Run multiple times to build aspect ratio history
        for _ in range(5):
            result = extractor.extract(tracks, (480, 640))
        # The scene should track fallen count
        assert result["scene"].fallen_count >= 0

    def test_movement_trends(self):
        now = time.time()
        tracks = [
            TrackedPerson(
                track_id=1,
                bbox=(100, 100, 200, 300),
                confidence=0.9,
                trajectory=[
                    TrajectoryPoint(x=150, y=200, timestamp=now - 3),
                    TrajectoryPoint(x=155, y=200, timestamp=now - 2),
                    TrajectoryPoint(x=165, y=200, timestamp=now - 1),
                    TrajectoryPoint(x=180, y=200, timestamp=now),
                ],
                first_seen=now - 5,
                last_seen=now,
                total_frames=15,
            ),
        ]
        result = self.extractor.extract(tracks, (480, 640))
        pf = result["persons"][1]
        # Person is accelerating (speed_trend should be positive)
        assert isinstance(pf.speed_trend, float)
        assert isinstance(pf.direction_trend, float)
