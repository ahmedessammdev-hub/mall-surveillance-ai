"""
LLM Reasoning Engine.

Orchestrates event analysis by combining event data, retrieved
similar events, and security rules into LLM prompts.
Only activated when event confidence exceeds threshold.
"""

import logging
from typing import Optional

from event_engine.schemas import ReasoningResult, SecurityEvent
from reasoning.llm_client import LLMClient
from reasoning.prompts import SYSTEM_PROMPT, format_event_analysis_prompt
from vector_db.faiss_store import SimilarEvent

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """LLM-powered reasoning engine for event analysis.

    The LLM is activated ONLY when event confidence > threshold.
    It provides structured analysis including event confirmation,
    risk assessment, and recommended actions.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.llm = LLMClient(config)

    def analyze_event(
        self,
        event: SecurityEvent,
        similar_events: list[SimilarEvent] | None = None,
        camera_meta: dict | None = None,
    ) -> ReasoningResult:
        """Analyze a security event using the LLM.

        Args:
            event: The detected security event.
            similar_events: Retrieved similar historical events.
            camera_meta: Camera metadata for context.

        Returns:
            ReasoningResult with LLM analysis.
        """
        if not self.llm.is_available:
            logger.debug("LLM not available, returning default reasoning")
            return self._default_reasoning(event)

        try:
            # Format prompt
            prompt = format_event_analysis_prompt(
                event=event,
                similar_events=similar_events or [],
                camera_context=camera_meta or {},
            )

            # Call LLM
            result = self.llm.generate_json(prompt, system_prompt=SYSTEM_PROMPT)

            if not result:
                return self._default_reasoning(event)

            return ReasoningResult(
                event_type=result.get("event_type", event.event_type.value),
                risk_level=result.get("risk_level", event.risk_level.value),
                requires_intervention=result.get("requires_intervention", False),
                reasoning=result.get("reasoning", "LLM analysis completed"),
                recommended_action=result.get("recommended_action", "Monitor the situation"),
            )

        except Exception as e:
            logger.error(f"Reasoning engine error: {e}")
            return self._default_reasoning(event)

    @staticmethod
    def _default_reasoning(event: SecurityEvent) -> ReasoningResult:
        """Produce a rule-based reasoning result when LLM is unavailable."""
        intervention_types = {"fight", "fall", "crowd_panic", "vandalism"}
        requires_intervention = event.event_type.value in intervention_types

        risk_actions = {
            "critical": "Dispatch security immediately. Notify mall management.",
            "high": "Send nearest security officer to investigate.",
            "medium": "Monitor the situation via cameras. Stand by for escalation.",
            "low": "Log the event. No immediate action required.",
        }

        return ReasoningResult(
            event_type=event.event_type.value,
            risk_level=event.risk_level.value,
            requires_intervention=requires_intervention,
            reasoning=(
                f"Automated detection: {event.event_type.value} event detected "
                f"with {event.confidence:.1%} confidence. "
                f"{event.person_count} person(s) involved. "
                f"Risk level assessed as {event.risk_level.value} based on motion and behavior analysis."
            ),
            recommended_action=risk_actions.get(
                event.risk_level.value,
                "Monitor the situation.",
            ),
        )

    @property
    def is_available(self) -> bool:
        return self.llm.is_available
