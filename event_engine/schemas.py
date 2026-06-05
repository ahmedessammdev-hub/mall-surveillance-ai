"""
Pydantic event schemas for the surveillance system.

Defines all event types and the unified SecurityEvent model
used throughout the pipeline.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class EventType(str, Enum):
    FIGHT = "fight"
    FALL = "fall"
    CROWD_PANIC = "crowd_panic"
    LOITERING = "loitering"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    VANDALISM = "vandalism"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InvolvedTrack(BaseModel):
    """A tracked person involved in an event."""
    track_id: int
    bbox: list[float] = Field(default_factory=list)
    speed: float = 0.0
    role: str = ""  # e.g., "aggressor", "victim", "bystander"

    class Config:
        from_attributes = True


class SecurityEvent(BaseModel):
    """Unified event schema for all security incident types.

    This is the primary data structure passed through the pipeline:
    Event Engine → FAISS → Reasoning → Alerts.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    camera_id: str = ""
    location: str = ""
    event_type: EventType
    involved_tracks: list[InvolvedTrack] = Field(default_factory=list)
    person_count: int = 0
    crowd_density: float = 0.0

    # Behavior & interaction scores
    behavior_scores: dict[str, float] = Field(default_factory=dict)
    interaction_scores: dict[str, float] = Field(default_factory=dict)
    motion_features: dict[str, float] = Field(default_factory=dict)

    # Embedding & retrieval
    embedding_id: Optional[str] = None
    embedding: Optional[Any] = Field(default=None, exclude=True)  # np.ndarray, excluded from JSON
    similar_events: list[dict] = Field(default_factory=list)

    # Scoring
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = RiskLevel.LOW

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True

    def to_summary(self) -> dict:
        """Return a JSON-safe summary for logging/API."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "camera_id": self.camera_id,
            "event_type": self.event_type.value,
            "person_count": self.person_count,
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level.value,
            "involved_tracks": [t.track_id for t in self.involved_tracks],
        }


# ---------------------------------------------------------------------------
# Specialized event types (extend SecurityEvent with type-specific fields)
# ---------------------------------------------------------------------------

class FightEvent(SecurityEvent):
    """Physical altercation between two or more persons."""
    event_type: EventType = EventType.FIGHT
    aggression_score: float = 0.0
    interaction_intensity: float = 0.0
    involved_count: int = 0


class FallEvent(SecurityEvent):
    """Person fall — potential medical emergency."""
    event_type: EventType = EventType.FALL
    vertical_drop: float = 0.0
    recovery_detected: bool = False
    time_on_ground: float = 0.0


class CrowdPanicEvent(SecurityEvent):
    """Sudden crowd dispersal or panic movement."""
    event_type: EventType = EventType.CROWD_PANIC
    dispersion_score: float = 0.0
    avg_crowd_speed: float = 0.0
    affected_count: int = 0


class LoiteringEvent(SecurityEvent):
    """Person lingering in an area for an extended duration."""
    event_type: EventType = EventType.LOITERING
    dwell_duration: float = 0.0  # seconds
    zone: str = ""


class SuspiciousBehaviorEvent(SecurityEvent):
    """Unusual movement patterns or zone violations."""
    event_type: EventType = EventType.SUSPICIOUS_BEHAVIOR
    direction_change_count: int = 0
    zone_violation_count: int = 0
    pattern_anomaly_score: float = 0.0


class VandalismEvent(SecurityEvent):
    """Aggressive physical action against objects/infrastructure."""
    event_type: EventType = EventType.VANDALISM
    impact_force_estimate: float = 0.0
    target_area: str = ""


# ---------------------------------------------------------------------------
# Reasoning / Alert schemas
# ---------------------------------------------------------------------------

class ReasoningResult(BaseModel):
    """Output from the LLM reasoning engine."""
    event_type: str = ""
    risk_level: str = ""
    requires_intervention: bool = False
    reasoning: str = ""
    recommended_action: str = ""
    confidence_adjustment: float = 0.0  # LLM may adjust confidence

    class Config:
        from_attributes = True


class AlertSchema(BaseModel):
    """Alert generated from an event + reasoning."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""
    event_type: EventType = EventType.FIGHT
    priority: "AlertPriority" = "P3"
    confidence: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reasoning: str = ""
    recommended_action: str = ""

    class Config:
        from_attributes = True


# Import here to avoid circular dependency
from config import AlertPriority  # noqa: E402

AlertSchema.model_rebuild()
