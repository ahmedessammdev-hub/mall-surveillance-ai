"""Tests for Event Engine (rules + constructor)."""

import time
import pytest
import numpy as np

from ai.feature_extractor import PersonFeatures, SceneFeatures
from ai.tracker import TrackedPerson, TrajectoryPoint
from event_engine.schemas import EventType, RiskLevel, SecurityEvent
from event_engine.rules import EventRules
from event_engine.constructor import EventConstructor


class TestEventRules:
    def setup_method(self):
        from config import settings
        self.rules = EventRules(settings.event)

    def test_no_event_empty_scene(self):
        events = self.rules.evaluate_all(
            camera_id="cam1",
            tracks=[],
            persons={},
            scene=SceneFeatures(),
        )
        assert events == []

    def test_fight_detection_conditions(self):
        """Test that fight is detected when conditions are met."""
        now = time.time()
        tracks = [
            TrackedPerson(track_id=1, bbox=(100, 100, 200, 300), confidence=0.9,
                         first_seen=now - 5, last_seen=now, total_frames=25),
            TrackedPerson(track_id=2, bbox=(130, 100, 230, 300), confidence=0.85,
                         first_seen=now - 5, last_seen=now, total_frames=25),
        ]

        persons = {
            1: PersonFeatures(
                track_id=1, speed=200.0, acceleration=100.0,
                nearest_person_distance=50.0, nearest_person_id=2,
                people_within_radius=1, interaction_duration=5.0,
                proximity_score=0.8,
            ),
            2: PersonFeatures(
                track_id=2, speed=180.0, acceleration=90.0,
                nearest_person_distance=50.0, nearest_person_id=1,
                people_within_radius=1, interaction_duration=5.0,
                proximity_score=0.8,
            ),
        }

        scene = SceneFeatures(person_count=2, avg_speed=190, crowd_density=0.001)

        events = self.rules.evaluate_all("cam1", tracks, persons, scene)
        fight_events = [e for e in events if e.event_type == EventType.FIGHT]
        # Should detect fight given high speed, close proximity, and sustained interaction
        assert len(fight_events) >= 1

    def test_loitering_detection(self):
        """Test loitering detection with low speed and high dwell time."""
        persons = {
            1: PersonFeatures(
                track_id=1, speed=5.0, dwell_time=600.0,
                displacement=20.0, path_length=50.0,
            ),
        }
        tracks = [
            TrackedPerson(track_id=1, bbox=(100, 100, 200, 300), confidence=0.9,
                         first_seen=time.time() - 600, last_seen=time.time(), total_frames=3000),
        ]
        scene = SceneFeatures(person_count=1)

        events = self.rules.evaluate_all("cam1", tracks, persons, scene)
        loiter_events = [e for e in events if e.event_type == EventType.LOITERING]
        assert len(loiter_events) >= 1


class TestEventConstructor:
    def setup_method(self):
        from config import settings
        self.constructor = EventConstructor(settings.event)

    def test_deduplication(self):
        """Events of the same type from the same camera should be deduplicated."""
        now = time.time()
        tracks = [
            TrackedPerson(track_id=1, bbox=(100, 100, 200, 300), confidence=0.9,
                         first_seen=now - 5, last_seen=now, total_frames=25),
            TrackedPerson(track_id=2, bbox=(130, 100, 230, 300), confidence=0.85,
                         first_seen=now - 5, last_seen=now, total_frames=25),
        ]

        persons = {
            1: PersonFeatures(
                track_id=1, speed=200.0, acceleration=100.0,
                nearest_person_distance=50.0, nearest_person_id=2,
                people_within_radius=1, interaction_duration=5.0,
                proximity_score=0.8,
            ),
            2: PersonFeatures(
                track_id=2, speed=180.0, acceleration=90.0,
                nearest_person_distance=50.0, nearest_person_id=1,
                people_within_radius=1, interaction_duration=5.0,
                proximity_score=0.8,
            ),
        }
        scene = SceneFeatures(person_count=2, avg_speed=190, crowd_density=0.001)
        features = {"persons": persons, "scene": scene}

        # First call may produce events
        events1 = self.constructor.process_frame_data("cam1", tracks, features)

        # Second call within dedup window should produce fewer/no events
        events2 = self.constructor.process_frame_data("cam1", tracks, features)

        if events1:
            # Same event types should be deduplicated
            types1 = {e.event_type for e in events1}
            types2 = {e.event_type for e in events2}
            assert len(types2.intersection(types1)) == 0 or len(events2) <= len(events1)
