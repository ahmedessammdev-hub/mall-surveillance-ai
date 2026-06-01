"""Alert repository — CRUD operations for the Alert table."""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from database.models import Alert


class AlertRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        event_id: int,
        priority: str,
        event_type: str,
        confidence: float = 0.0,
        risk_level: str = "low",
        reasoning: str = "",
        recommended_action: str = "",
    ) -> Alert:
        alert = Alert(
            event_id=event_id,
            priority=priority,
            event_type=event_type,
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.utcnow(),
            reasoning=reasoning,
            recommended_action=recommended_action,
        )
        self.session.add(alert)
        self.session.flush()
        return alert

    def create_from_schema(self, alert_schema) -> Alert:
        """Create an Alert from a Pydantic AlertSchema."""
        return self.create(
            event_id=alert_schema.event_id if isinstance(alert_schema.event_id, int) else 0,
            priority=alert_schema.priority.value,
            event_type=alert_schema.event_type.value,
            confidence=alert_schema.confidence,
            risk_level=alert_schema.risk_level.value,
            reasoning=alert_schema.reasoning,
            recommended_action=alert_schema.recommended_action,
        )

    def get_by_id(self, alert_id: int) -> Optional[Alert]:
        return self.session.query(Alert).filter(Alert.id == alert_id).first()

    def get_by_uid(self, uid: str) -> Optional[Alert]:
        return self.session.query(Alert).filter(Alert.uid == uid).first()

    def get_all(
        self,
        priority: str | None = None,
        acknowledged: bool | None = None,
        event_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Alert]:
        q = self.session.query(Alert)
        if priority is not None:
            q = q.filter(Alert.priority == priority)
        if acknowledged is not None:
            q = q.filter(Alert.acknowledged == acknowledged)
        if event_type is not None:
            q = q.filter(Alert.event_type == event_type)
        if start_date is not None:
            q = q.filter(Alert.timestamp >= start_date)
        if end_date is not None:
            q = q.filter(Alert.timestamp <= end_date)
        return q.order_by(desc(Alert.timestamp)).limit(limit).offset(offset).all()

    def get_active(self, limit: int = 50) -> list[Alert]:
        return self.get_all(acknowledged=False, limit=limit)

    def get_recent(self, limit: int = 20) -> list[Alert]:
        return (
            self.session.query(Alert)
            .order_by(desc(Alert.timestamp))
            .limit(limit)
            .all()
        )

    def acknowledge(self, alert_id: int, user: str = "operator") -> Optional[Alert]:
        alert = self.get_by_id(alert_id)
        if not alert:
            return None
        alert.acknowledged = True
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.utcnow()
        self.session.flush()
        return alert

    def count(
        self,
        priority: str | None = None,
        acknowledged: bool | None = None,
    ) -> int:
        q = self.session.query(func.count(Alert.id))
        if priority is not None:
            q = q.filter(Alert.priority == priority)
        if acknowledged is not None:
            q = q.filter(Alert.acknowledged == acknowledged)
        return q.scalar() or 0

    def count_by_priority(self) -> list[tuple[str, int]]:
        return (
            self.session.query(Alert.priority, func.count(Alert.id))
            .group_by(Alert.priority)
            .all()
        )

    def count_by_type(self) -> list[tuple[str, int]]:
        return (
            self.session.query(Alert.event_type, func.count(Alert.id))
            .group_by(Alert.event_type)
            .all()
        )

    def delete(self, alert_id: int) -> bool:
        alert = self.get_by_id(alert_id)
        if not alert:
            return False
        self.session.delete(alert)
        self.session.flush()
        return True
