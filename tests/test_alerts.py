"""Tests for Alert Engine."""

import pytest

from alerts.engine import AlertEngine, PRIORITY_MAP
from config import AlertPriority
from event_engine.schemas import (
    AlertSchema,
    EventType,
    ReasoningResult,
    RiskLevel,
    SecurityEvent,
)


class TestAlertEngine:
    def setup_method(self):
        self.engine = AlertEngine()

    def test_fight_is_p1(self):
        event = SecurityEvent(
            event_type=EventType.FIGHT,
            confidence=0.85,
            risk_level=RiskLevel.HIGH,
            person_count=2,
        )
        alert = self.engine.process_event(event)
        assert alert is not None
        assert alert.priority == AlertPriority.P1

    def test_fall_is_p1(self):
        event = SecurityEvent(
            event_type=EventType.FALL,
            confidence=0.9,
            risk_level=RiskLevel.CRITICAL,
            person_count=1,
        )
        alert = self.engine.process_event(event)
        assert alert.priority == AlertPriority.P1

    def test_crowd_panic_is_p2(self):
        event = SecurityEvent(
            event_type=EventType.CROWD_PANIC,
            confidence=0.75,
            risk_level=RiskLevel.HIGH,
            person_count=10,
        )
        alert = self.engine.process_event(event)
        assert alert.priority == AlertPriority.P2

    def test_loitering_is_p3(self):
        event = SecurityEvent(
            event_type=EventType.LOITERING,
            confidence=0.65,
            risk_level=RiskLevel.LOW,
            person_count=1,
        )
        alert = self.engine.process_event(event)
        assert alert.priority == AlertPriority.P3

    def test_critical_risk_elevates_to_p1(self):
        event = SecurityEvent(
            event_type=EventType.LOITERING,  # Normally P3
            confidence=0.95,
            risk_level=RiskLevel.CRITICAL,  # But critical risk
            person_count=1,
        )
        alert = self.engine.process_event(event)
        assert alert.priority == AlertPriority.P1  # Elevated

    def test_with_reasoning(self):
        event = SecurityEvent(
            event_type=EventType.FIGHT,
            confidence=0.85,
            person_count=2,
        )
        reasoning = ReasoningResult(
            event_type="fight",
            risk_level="critical",
            requires_intervention=True,
            reasoning="Two individuals engaged in physical altercation.",
            recommended_action="Dispatch security immediately.",
        )
        alert = self.engine.process_event(event, reasoning)
        assert alert.reasoning == "Two individuals engaged in physical altercation."
        assert "Dispatch" in alert.recommended_action

    def test_alert_count(self):
        initial = self.engine.total_alerts
        event = SecurityEvent(
            event_type=EventType.FIGHT,
            confidence=0.8,
            person_count=2,
        )
        self.engine.process_event(event)
        assert self.engine.total_alerts == initial + 1


class TestPriorityMapping:
    def test_all_event_types_mapped(self):
        for et in EventType:
            assert et in PRIORITY_MAP
