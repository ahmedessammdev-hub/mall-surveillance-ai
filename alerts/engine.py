"""
Alert Engine.

Processes security events into prioritized alerts.
Priority mapping:
  P1 (Critical): Fight, Medical Emergency (Fall)
  P2 (High):     Crowd Panic, Vandalism
  P3 (Medium):   Loitering, Suspicious Activity
"""

import logging
from datetime import datetime
from typing import Optional

from config import AlertPriority
from event_engine.schemas import (
    AlertSchema,
    EventType,
    ReasoningResult,
    RiskLevel,
    SecurityEvent,
)

logger = logging.getLogger(__name__)

# Event type → alert priority mapping
PRIORITY_MAP: dict[EventType, AlertPriority] = {
    EventType.FIGHT: AlertPriority.P1,
    EventType.FALL: AlertPriority.P1,
    EventType.CROWD_PANIC: AlertPriority.P2,
    EventType.VANDALISM: AlertPriority.P2,
    EventType.LOITERING: AlertPriority.P3,
    EventType.SUSPICIOUS_BEHAVIOR: AlertPriority.P3,
}


class AlertEngine:
    """Processes events into prioritized alerts.

    Applies priority mapping, incorporates LLM reasoning,
    and produces AlertSchema objects for storage and display.
    """

    def __init__(self) -> None:
        self._alert_count = 0

    def process_event(
        self,
        event: SecurityEvent,
        reasoning: Optional[ReasoningResult] = None,
    ) -> Optional[AlertSchema]:
        """Process a security event into an alert.

        Args:
            event: Detected security event.
            reasoning: Optional LLM reasoning result.

        Returns:
            AlertSchema if the event warrants an alert, None otherwise.
        """
        # Determine priority
        priority = PRIORITY_MAP.get(event.event_type, AlertPriority.P3)

        # Elevate priority if risk is critical
        if event.risk_level == RiskLevel.CRITICAL and priority != AlertPriority.P1:
            priority = AlertPriority.P1

        # Build reasoning text
        if reasoning:
            reasoning_text = reasoning.reasoning
            action_text = reasoning.recommended_action

            # Override risk level from LLM if it escalates
            if reasoning.risk_level == "critical" and priority != AlertPriority.P1:
                priority = AlertPriority.P1
        else:
            reasoning_text = self._generate_default_reasoning(event)
            action_text = self._generate_default_action(event, priority)

        # Determine final risk level
        risk_level = event.risk_level
        if reasoning and reasoning.risk_level:
            try:
                risk_level = RiskLevel(reasoning.risk_level)
            except ValueError:
                pass

        self._alert_count += 1

        alert = AlertSchema(
            event_id=event.event_id,
            event_type=event.event_type,
            priority=priority,
            confidence=event.confidence,
            risk_level=risk_level,
            timestamp=datetime.utcnow(),
            reasoning=reasoning_text,
            recommended_action=action_text,
        )

        logger.info(
            f"Alert generated: [{priority.value}] {event.event_type.value} "
            f"(confidence={event.confidence:.2f}, risk={risk_level.value})"
        )

        return alert

    @staticmethod
    def _generate_default_reasoning(event: SecurityEvent) -> str:
        """Generate default reasoning when LLM is not available."""
        type_descriptions = {
            EventType.FIGHT: "Physical altercation detected between multiple individuals",
            EventType.FALL: "Person fall detected — possible medical emergency",
            EventType.CROWD_PANIC: "Abnormal crowd dispersal pattern detected",
            EventType.LOITERING: "Extended loitering behavior detected in monitored zone",
            EventType.SUSPICIOUS_BEHAVIOR: "Unusual movement pattern detected",
            EventType.VANDALISM: "Aggressive physical action detected against property",
        }

        desc = type_descriptions.get(event.event_type, "Security event detected")
        return (
            f"{desc}. "
            f"Detection confidence: {event.confidence:.1%}. "
            f"Persons involved: {event.person_count}. "
            f"Camera: {event.camera_id}."
        )

    @staticmethod
    def _generate_default_action(event: SecurityEvent, priority: AlertPriority) -> str:
        """Generate default recommended action."""
        actions = {
            AlertPriority.P1: (
                "IMMEDIATE: Dispatch security team to camera location. "
                "Notify management. Prepare to contact emergency services if needed."
            ),
            AlertPriority.P2: (
                "URGENT: Send nearest security officer to investigate. "
                "Monitor via live camera feed. Stand by for escalation."
            ),
            AlertPriority.P3: (
                "MONITOR: Continue observation via camera feed. "
                "Log incident details. Dispatch officer if behavior escalates."
            ),
        }
        return actions.get(priority, "Monitor the situation.")

    @property
    def total_alerts(self) -> int:
        return self._alert_count
