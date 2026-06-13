"""Tests for ByteTrack Multi-Person Tracker."""

import time
import pytest
import numpy as np

from ai.detector import Detection
from ai.tracker import MultiPersonTracker, TrackedPerson, TrajectoryPoint


class TestTrajectoryPoint:
    def test_creation(self):
        tp = TrajectoryPoint(x=100.0, y=200.0, timestamp=time.time())
        assert tp.x == 100.0
        assert tp.y == 200.0


class TestTrackedPerson:
    def test_creation(self):
        tp = TrackedPerson(track_id=1, bbox=(10, 20, 100, 200), confidence=0.9)
        assert tp.track_id == 1
        assert tp.center == (55.0, 110.0)

    def test_to_dict(self):
        tp = TrackedPerson(track_id=1, bbox=(10, 20, 100, 200), confidence=0.9)
        d = tp.to_dict()
        assert d["track_id"] == 1
        assert "center" in d


class TestMultiPersonTracker:
    def setup_method(self):
        from config import settings
        self.tracker = MultiPersonTracker(settings.tracker)

    def test_new_tracks_created(self, sample_detections):
        tracks = self.tracker.update(sample_detections)
        assert len(tracks) == 3
        for t in tracks:
            assert isinstance(t, TrackedPerson)
            assert t.status == "active"

    def test_track_persistence(self, sample_detections):
        tracks1 = self.tracker.update(sample_detections)
        ids1 = {t.track_id for t in tracks1}

        # Same detections should match to same tracks
        tracks2 = self.tracker.update(sample_detections)
        ids2 = {t.track_id for t in tracks2}

        assert len(ids1.intersection(ids2)) > 0  # At least some tracks persist

    def test_trajectory_growth(self, sample_detections):
        self.tracker.update(sample_detections)
        tracks = self.tracker.update(sample_detections)

        for t in tracks:
            assert len(t.trajectory) >= 1

    def test_empty_detections(self):
        tracks = self.tracker.update([])
        assert tracks == []

    def test_get_active_tracks(self, sample_detections):
        self.tracker.update(sample_detections)
        active = self.tracker.get_active_tracks()
        assert len(active) >= 1

    def test_iou_computation(self):
        iou = MultiPersonTracker._compute_iou(
            (0, 0, 10, 10),
            (5, 5, 15, 15),
        )
        # Intersection = 5x5 = 25, Union = 100+100-25 = 175
        assert abs(iou - 25 / 175) < 0.01

    def test_iou_no_overlap(self):
        iou = MultiPersonTracker._compute_iou(
            (0, 0, 10, 10),
            (20, 20, 30, 30),
        )
        assert iou == 0.0

    def test_speed_computation(self):
        now = time.time()
        traj = [
            TrajectoryPoint(x=0, y=0, timestamp=now),
            TrajectoryPoint(x=3, y=4, timestamp=now + 1),
        ]
        speeds = MultiPersonTracker._compute_speeds(traj)
        assert len(speeds) == 1
        assert abs(speeds[0] - 5.0) < 0.01  # 3-4-5 triangle, 5 px/sec
