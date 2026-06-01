"""Event repository — CRUD operations for the Event table."""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from database.models import Event


class EventRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        camera_id: int,
        event_type: str,
        confidence: float = 0.0,
        risk_level: str = "low",
        location: str = "",
        involved_tracks: list | None = None,
        person_count: int = 0,
        crowd_density: float = 0.0,
        behavior_scores: dict | None = None,
        interaction_scores: dict | None = None,
        motion_features: dict | None = None,
        similar_events: list | None = None,
        reasoning: dict | None = None,
    ) -> Event:
        event = Event(
            camera_id=camera_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            location=location,
            involved_tracks_json=json.dumps(involved_tracks or []),
            person_count=person_count,
            crowd_density=crowd_density,
            behavior_scores_json=json.dumps(behavior_scores or {}),
            interaction_scores_json=json.dumps(interaction_scores or {}),
            motion_features_json=json.dumps(motion_features or {}),
            similar_events_json=json.dumps(similar_events or []),
            confidence=confidence,
            risk_level=risk_level,
            reasoning_json=json.dumps(reasoning or {}),
        )
        self.session.add(event)
        self.session.flush()
        return event

    def create_from_schema(self, event_schema) -> Event:
        """Create an Event from a Pydantic SecurityEvent schema."""
        return self.create(
            camera_id=int(event_schema.camera_id) if event_schema.camera_id.isdigit() else 1,
            event_type=event_schema.event_type.value,
            confidence=event_schema.confidence,
            risk_level=event_schema.risk_level.value,
            location=event_schema.location,
            involved_tracks=[t.dict() if hasattr(t, "dict") else t for t in event_schema.involved_tracks],
            person_count=event_schema.person_count,
            crowd_density=event_schema.crowd_density,
            behavior_scores=event_schema.behavior_scores,
            interaction_scores=event_schema.interaction_scores,
            motion_features=event_schema.motion_features,
            similar_events=event_schema.similar_events,
        )

    def get_by_id(self, event_id: int) -> Optional[Event]:
        return self.session.query(Event).filter(Event.id == event_id).first()

    def get_by_uid(self, uid: str) -> Optional[Event]:
        return self.session.query(Event).filter(Event.uid == uid).first()

    def get_all(
        self,
        camera_id: int | None = None,
        event_type: str | None = None,
        risk_level: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        q = self.session.query(Event)
        if camera_id is not None:
            q = q.filter(Event.camera_id == camera_id)
        if event_type is not None:
            q = q.filter(Event.event_type == event_type)
        if risk_level is not None:
            q = q.filter(Event.risk_level == risk_level)
        if start_date is not None:
            q = q.filter(Event.timestamp >= start_date)
        if end_date is not None:
            q = q.filter(Event.timestamp <= end_date)
        return q.order_by(desc(Event.timestamp)).limit(limit).offset(offset).all()

    def get_recent(self, limit: int = 20) -> list[Event]:
        return (
            self.session.query(Event)
            .order_by(desc(Event.timestamp))
            .limit(limit)
            .all()
        )

    def count(
        self,
        camera_id: int | None = None,
        event_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        q = self.session.query(func.count(Event.id))
        if camera_id is not None:
            q = q.filter(Event.camera_id == camera_id)
        if event_type is not None:
            q = q.filter(Event.event_type == event_type)
        if start_date is not None:
            q = q.filter(Event.timestamp >= start_date)
        if end_date is not None:
            q = q.filter(Event.timestamp <= end_date)
        return q.scalar() or 0

    def count_by_type(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[tuple[str, int]]:
        q = self.session.query(Event.event_type, func.count(Event.id))
        if start_date:
            q = q.filter(Event.timestamp >= start_date)
        if end_date:
            q = q.filter(Event.timestamp <= end_date)
        return q.group_by(Event.event_type).all()

    def count_by_risk(self) -> list[tuple[str, int]]:
        return (
            self.session.query(Event.risk_level, func.count(Event.id))
            .group_by(Event.risk_level)
            .all()
        )

    def update(self, event_id: int, **kwargs) -> Optional[Event]:
        event = self.get_by_id(event_id)
        if not event:
            return None
        for field in ("involved_tracks", "behavior_scores", "interaction_scores",
                      "motion_features", "similar_events", "reasoning"):
            if field in kwargs:
                kwargs[f"{field}_json"] = json.dumps(kwargs.pop(field))
        for key, value in kwargs.items():
            if hasattr(event, key):
                setattr(event, key, value)
        self.session.flush()
        return event

    def delete(self, event_id: int) -> bool:
        event = self.get_by_id(event_id)
        if not event:
            return False
        self.session.delete(event)
        self.session.flush()
        return True
