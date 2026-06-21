"""
Event Detection Rules.

Each rule function takes person/scene features and returns a potential
SecurityEvent if the rule conditions are met.
"""

import logging
from datetime import datetime
from typing import Optional

from ai.feature_extractor import PersonFeatures, SceneFeatures
from ai.tracker import TrackedPerson
from event_engine.schemas import (
    AlertPriority,
    CrowdPanicEvent,
    EventType,
    FallEvent,
    FightEvent,
    InvolvedTrack,
    LoiteringEvent,
    RiskLevel,
    SecurityEvent,
    SuspiciousBehaviorEvent,
    VandalismEvent,
)

logger = logging.getLogger(__name__)


class EventRules:
    """Rule-based event detection engine.

    Applies heuristic rules to tracking/feature data to produce candidate events.
    Each rule outputs a SecurityEvent with a confidence score.
    """

    def __init__(self, config) -> None:
        self.cfg = config

    def evaluate_all(
        self,
        camera_id: str,
        tracks: list[TrackedPerson],
        persons: dict[int, PersonFeatures],
        scene: SceneFeatures,
    ) -> list[SecurityEvent]:
        """Run all rules and return detected events."""
        events: list[SecurityEvent] = []

        fight = self.detect_fight(camera_id, tracks, persons, scene)
        if fight:
            events.append(fight)

        fall = self.detect_fall(camera_id, tracks, persons)
        if fall:
            events.append(fall)

        panic = self.detect_crowd_panic(camera_id, tracks, persons, scene)
        if panic:
            events.append(panic)

        loitering_events = self.detect_loitering(camera_id, tracks, persons)
        events.extend(loitering_events)

        suspicious_events = self.detect_suspicious(camera_id, tracks, persons)
        events.extend(suspicious_events)

        vandalism = self.detect_vandalism(camera_id, tracks, persons)
        if vandalism:
            events.append(vandalism)

        return events

    # -------------------------------------------------------------------
    # Fight Detection
    # -------------------------------------------------------------------
    def detect_fight(
        self,
        camera_id: str,
        tracks: list[TrackedPerson],
        persons: dict[int, PersonFeatures],
        scene: SceneFeatures,
    ) -> Optional[FightEvent]:
        """Detect physical fights based on proximity, speed, and interaction."""
        if scene.person_count < 2:
            return None

        # Find interacting pairs with aggressive indicators
        fight_pairs: list[tuple[int, int]] = []
        fight_scores: list[float] = []

        for tid, pf in persons.items():
            if (
                pf.speed > self.cfg.fight_speed_threshold * 0.5
                and pf.nearest_person_distance < self.cfg.fight_proximity_threshold
                and pf.interaction_duration > self.cfg.fight_interaction_duration
                and pf.people_within_radius >= 1
            ):
                pair = (tid, pf.nearest_person_id)
                rev_pair = (pf.nearest_person_id, tid)
                if pair not in fight_pairs and rev_pair not in fight_pairs:
                    fight_pairs.append(pair)

                    # Compute fight score with trend awareness
                    speed_score = min(pf.speed / self.cfg.fight_speed_threshold, 1.0)
                    prox_score = 1.0 - (pf.nearest_person_distance / self.cfg.fight_proximity_threshold)
                    interaction_score = min(pf.interaction_duration / (self.cfg.fight_interaction_duration * 3), 1.0)
                    accel_score = min(abs(pf.acceleration) / 200.0, 1.0)
                    # Boost if speed is increasing (aggressive trend)
                    trend_boost = 0.1 if pf.speed_trend > 50 else 0.0

                    score = (speed_score * 0.3 + prox_score * 0.3 + interaction_score * 0.25 + accel_score * 0.15 + trend_boost)
                    fight_scores.append(score)

        if not fight_pairs:
            return None

        max_score = max(fight_scores)
        if max_score < self.cfg.event_confidence_threshold:
            return None

        # Build involved tracks
        involved_ids = set()
        for a, b in fight_pairs:
            involved_ids.add(a)
            involved_ids.add(b)

        involved = []
        for tid in involved_ids:
            t = next((t for t in tracks if t.track_id == tid), None)
            pf = persons.get(tid)
            if t:
                involved.append(InvolvedTrack(
                    track_id=tid,
                    bbox=list(t.bbox),
                    speed=pf.speed if pf else 0.0,
                    role="participant",
                    posture_state=pf.posture_state if pf else "unknown",
                    zone=pf.current_zone if pf else "",
                ))

        risk = self._score_to_risk(max_score)

        # Determine dominant zone
        zones = [pf.current_zone for pf in persons.values() if pf.current_zone]
        event_zone = max(set(zones), key=zones.count) if zones else ""

        return FightEvent(
            camera_id=camera_id,
            person_count=scene.person_count,
            crowd_density=scene.crowd_density,
            confidence=round(max_score, 3),
            risk_level=risk,
            involved_tracks=involved,
            aggression_score=max_score,
            interaction_intensity=max(
                (persons.get(tid, PersonFeatures(track_id=0)).interaction_duration for tid in involved_ids),
                default=0.0,
            ),
            involved_count=len(involved_ids),
            posture_state="fighting",
            zone=event_zone,
            behavior_scores={"fight_score": max_score},
            motion_features=scene.to_dict(),
        )

    # -------------------------------------------------------------------
    # Fall Detection
    # -------------------------------------------------------------------
    def detect_fall(
        self,
        camera_id: str,
        tracks: list[TrackedPerson],
        persons: dict[int, PersonFeatures],
    ) -> Optional[FallEvent]:
        """Detect falls based on vertical displacement, speed drop, and posture."""
        for tid, pf in persons.items():
            vertical_ok = pf.vertical_displacement > self.cfg.fall_vertical_displacement
            speed_drop_ok = pf.speed < pf.speed_variance * self.cfg.fall_speed_drop_ratio + 5.0
            posture_ok = pf.posture_state == "fallen"

            # Trigger on vertical displacement + speed drop, OR posture-based detection
            if not (vertical_ok and speed_drop_ok) and not posture_ok:
                continue

            # Compute confidence with posture boost
            base_confidence = min(
                (pf.vertical_displacement / (self.cfg.fall_vertical_displacement * 2))
                * 0.5
                + (1.0 - min(pf.speed / 50.0, 1.0)) * 0.3,
                1.0,
            )

            # Boost confidence if posture confirms fall
            posture_boost = 0.2 if posture_ok else 0.0
            confidence = min(base_confidence + posture_boost, 1.0)

            if confidence < self.cfg.event_confidence_threshold:
                continue

            t = next((t for t in tracks if t.track_id == tid), None)
            involved = []
            if t:
                involved.append(InvolvedTrack(
                    track_id=tid,
                    bbox=list(t.bbox),
                    speed=pf.speed,
                    role="victim",
                    posture_state=pf.posture_state,
                    zone=pf.current_zone,
                ))

            return FallEvent(
                camera_id=camera_id,
                person_count=1,
                confidence=round(confidence, 3),
                risk_level=self._score_to_risk(confidence),
                involved_tracks=involved,
                vertical_drop=pf.vertical_displacement,
                recovery_detected=False,
                posture_state=pf.posture_state,
                zone=pf.current_zone,
                behavior_scores={"fall_score": confidence},
                motion_features=pf.to_dict(),
            )

        return None

    # -------------------------------------------------------------------
    # Crowd Panic Detection
    # -------------------------------------------------------------------
    def detect_crowd_panic(
        self,
        camera_id: str,
        tracks: list[TrackedPerson],
        persons: dict[int, PersonFeatures],
        scene: SceneFeatures,
    ) -> Optional[CrowdPanicEvent]:
        """Detect crowd panic from high avg speed + high dispersion + crowd size."""
        if scene.person_count < self.cfg.panic_min_crowd_size:
            return None

        speed_ok = scene.avg_speed > self.cfg.panic_avg_speed_threshold
        dispersion_ok = scene.motion_dispersion > self.cfg.panic_dispersion_threshold

        if not (speed_ok and dispersion_ok):
            return None

        speed_score = min(scene.avg_speed / (self.cfg.panic_avg_speed_threshold * 2), 1.0)
        disp_score = scene.motion_dispersion
        crowd_score = min(scene.person_count / (self.cfg.panic_min_crowd_size * 3), 1.0)

        confidence = speed_score * 0.4 + disp_score * 0.35 + crowd_score * 0.25

        if confidence < self.cfg.event_confidence_threshold:
            return None

        involved = [
            InvolvedTrack(
                track_id=t.track_id,
                bbox=list(t.bbox),
                speed=persons.get(t.track_id, PersonFeatures(track_id=0)).speed,
                role="affected",
                posture_state=persons.get(t.track_id, PersonFeatures(track_id=0)).posture_state,
                zone=persons.get(t.track_id, PersonFeatures(track_id=0)).current_zone,
            )
            for t in tracks
        ]

        return CrowdPanicEvent(
            camera_id=camera_id,
            person_count=scene.person_count,
            crowd_density=scene.crowd_density,
            confidence=round(confidence, 3),
            risk_level=self._score_to_risk(confidence),
            involved_tracks=involved,
            dispersion_score=scene.motion_dispersion,
            avg_crowd_speed=scene.avg_speed,
            affected_count=scene.person_count,
            behavior_scores={"panic_score": confidence},
            motion_features=scene.to_dict(),
        )

    # -------------------------------------------------------------------
    # Loitering Detection
    # -------------------------------------------------------------------
    def detect_loitering(
        self,
        camera_id: str,
        tracks: list[TrackedPerson],
        persons: dict[int, PersonFeatures],
    ) -> list[LoiteringEvent]:
        """Detect loitering: low speed + high dwell time."""
        events: list[LoiteringEvent] = []

        for tid, pf in persons.items():
            if (
                pf.dwell_time > self.cfg.loitering_dwell_time
                and pf.speed < self.cfg.loitering_max_speed
            ):
                dwell_ratio = min(pf.dwell_time / (self.cfg.loitering_dwell_time * 3), 1.0)
                speed_ratio = 1.0 - min(pf.speed / self.cfg.loitering_max_speed, 1.0)
                confidence = dwell_ratio * 0.6 + speed_ratio * 0.4

                if confidence < self.cfg.event_confidence_threshold:
                    continue

                t = next((t for t in tracks if t.track_id == tid), None)
                involved = []
                if t:
                    involved.append(InvolvedTrack(
                        track_id=tid,
                        bbox=list(t.bbox),
                        speed=pf.speed,
                        role="subject",
                    ))

                events.append(LoiteringEvent(
                    camera_id=camera_id,
                    person_count=1,
                    confidence=round(confidence, 3),
                    risk_level=RiskLevel.LOW if confidence < 0.7 else RiskLevel.MEDIUM,
                    involved_tracks=involved,
                    dwell_duration=pf.dwell_time,
                    zone=pf.current_zone,
                    posture_state=pf.posture_state,
                    behavior_scores={"loitering_score": confidence},
                    motion_features=pf.to_dict(),
                ))

        return events

    # -------------------------------------------------------------------
    # Suspicious Behavior Detection
    # -------------------------------------------------------------------
    def detect_suspicious(
        self,
        camera_id: str,
        tracks: list[TrackedPerson],
        persons: dict[int, PersonFeatures],
    ) -> list[SuspiciousBehaviorEvent]:
        """Detect suspicious behavior: erratic movement, direction changes, zone violations."""
        events: list[SuspiciousBehaviorEvent] = []

        for tid, pf in persons.items():
            dir_changes_ok = pf.direction_changes > self.cfg.suspicious_direction_changes
            erratic_ok = pf.straightness < 0.3 and pf.path_length > 100
            speed_var_ok = pf.speed_variance > 1000
            zone_violation_ok = pf.is_in_restricted_zone

            indicators = sum([dir_changes_ok, erratic_ok, speed_var_ok, zone_violation_ok])
            if indicators < 2:
                continue

            confidence = min(indicators / 4.0, 1.0) * 0.8
            # Boost for restricted zone violations
            if zone_violation_ok:
                confidence = min(confidence + 0.15, 1.0)

            if confidence < self.cfg.event_confidence_threshold:
                continue

            t = next((t for t in tracks if t.track_id == tid), None)
            involved = []
            if t:
                involved.append(InvolvedTrack(
                    track_id=tid,
                    bbox=list(t.bbox),
                    speed=pf.speed,
                    role="subject",
                    posture_state=pf.posture_state,
                    zone=pf.current_zone,
                ))

            events.append(SuspiciousBehaviorEvent(
                camera_id=camera_id,
                person_count=1,
                confidence=round(confidence, 3),
                risk_level=RiskLevel.MEDIUM,
                involved_tracks=involved,
                direction_change_count=pf.direction_changes,
                zone_violation_count=1 if zone_violation_ok else 0,
                pattern_anomaly_score=1.0 - pf.straightness,
                posture_state=pf.posture_state,
                zone=pf.current_zone,
                behavior_scores={"suspicious_score": confidence},
                motion_features=pf.to_dict(),
            ))

        return events

    # -------------------------------------------------------------------
    # Vandalism Detection
    # -------------------------------------------------------------------
    def detect_vandalism(
        self,
        camera_id: str,
        tracks: list[TrackedPerson],
        persons: dict[int, PersonFeatures],
    ) -> Optional[VandalismEvent]:
        """Detect vandalism: sudden high acceleration + isolated person."""
        for tid, pf in persons.items():
            if (
                abs(pf.acceleration) > self.cfg.vandalism_acceleration_threshold
                and pf.people_within_radius == 0  # Acting alone
                and pf.speed > 50.0
            ):
                accel_score = min(abs(pf.acceleration) / (self.cfg.vandalism_acceleration_threshold * 2), 1.0)
                confidence = accel_score * 0.7 + (1.0 if pf.people_within_radius == 0 else 0.5) * 0.3

                if confidence < self.cfg.event_confidence_threshold:
                    continue

                t = next((t for t in tracks if t.track_id == tid), None)
                involved = []
                if t:
                    involved.append(InvolvedTrack(
                        track_id=tid,
                        bbox=list(t.bbox),
                        speed=pf.speed,
                        role="perpetrator",
                        posture_state=pf.posture_state,
                        zone=pf.current_zone,
                    ))

                return VandalismEvent(
                    camera_id=camera_id,
                    person_count=1,
                    confidence=round(confidence, 3),
                    risk_level=self._score_to_risk(confidence),
                    involved_tracks=involved,
                    impact_force_estimate=abs(pf.acceleration),
                    target_area=pf.current_zone,
                    posture_state=pf.posture_state,
                    zone=pf.current_zone,
                    behavior_scores={"vandalism_score": confidence},
                    motion_features=pf.to_dict(),
                )

        return None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _score_to_risk(score: float) -> RiskLevel:
        if score >= 0.85:
            return RiskLevel.CRITICAL
        elif score >= 0.7:
            return RiskLevel.HIGH
        elif score >= 0.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
