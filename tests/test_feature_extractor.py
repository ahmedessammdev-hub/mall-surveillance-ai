"""Tests for Feature Extractor."""

import time
import pytest
import numpy as np

from ai.feature_extractor import FeatureExtractor, PersonFeatures, SceneFeatures
from ai.tracker import TrackedPerson, TrajectoryPoint


class TestPersonFeatures:
    def test_to_dict(self):
        pf = PersonFeatures(track_id=1, speed=100.5, acceleration=20.0)
        d = pf.to_dict()
        assert d["track_id"] == 1
        assert d["speed"] == 100.5

    def test_defaults(self):
        pf = PersonFeatures(track_id=1)
        assert pf.speed == 0.0
        assert pf.nearest_person_distance == float("inf")


class TestSceneFeatures:
    def test_to_dict(self):
        sf = SceneFeatures(person_count=5, avg_speed=100.0)
        d = sf.to_dict()
        assert d["person_count"] == 5


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

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
        # Two persons very close together
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
        assert pf1.nearest_person_distance < 100  # Close
        assert pf1.nearest_person_id == 2
