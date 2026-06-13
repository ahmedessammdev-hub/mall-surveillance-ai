"""Tests for LLM Reasoning Engine."""

import pytest

from event_engine.schemas import EventType, RiskLevel, SecurityEvent, ReasoningResult
from reasoning.engine import ReasoningEngine
from reasoning.prompts import format_event_analysis_prompt, SYSTEM_PROMPT


class TestReasoningPrompts:
    def test_format_prompt(self):
        event = SecurityEvent(
            event_type=EventType.FIGHT,
            camera_id="cam1",
            confidence=0.85,
            person_count=2,
            crowd_density=0.001,
            behavior_scores={"fight_score": 0.85},
            motion_features={"avg_speed": 150.0},
        )

        prompt = format_event_analysis_prompt(event, [], {})
        assert "fight" in prompt.lower()
        assert "cam1" in prompt
        assert "0.85" in prompt

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 50


class TestReasoningEngine:
    def setup_method(self):
        from config import settings
        self.engine = ReasoningEngine(settings.llm)

    def test_default_reasoning_fight(self):
        event = SecurityEvent(
            event_type=EventType.FIGHT,
            camera_id="cam1",
            confidence=0.85,
            risk_level=RiskLevel.HIGH,
            person_count=2,
        )

        result = ReasoningEngine._default_reasoning(event)
        assert isinstance(result, ReasoningResult)
        assert result.requires_intervention is True
        assert "fight" in result.reasoning.lower()
        assert result.risk_level == "high"

    def test_default_reasoning_loitering(self):
        event = SecurityEvent(
            event_type=EventType.LOITERING,
            camera_id="cam1",
            confidence=0.65,
            risk_level=RiskLevel.LOW,
            person_count=1,
        )

        result = ReasoningEngine._default_reasoning(event)
        assert result.requires_intervention is False
        assert "loitering" in result.reasoning.lower()

    def test_default_reasoning_fall(self):
        event = SecurityEvent(
            event_type=EventType.FALL,
            confidence=0.9,
            risk_level=RiskLevel.CRITICAL,
            person_count=1,
        )
        result = ReasoningEngine._default_reasoning(event)
        assert result.requires_intervention is True
