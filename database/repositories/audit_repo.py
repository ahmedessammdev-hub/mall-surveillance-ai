"""AuditLog repository — CRUD operations for the AuditLog table."""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import AuditLog


class AuditRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def log_action(
        self,
        action: str,
        entity_type: str,
        entity_id: str = "",
        details: dict | None = None,
        user: str = "system",
    ) -> AuditLog:
        log = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details_json=json.dumps(details or {}),
            user=user,
            timestamp=datetime.utcnow(),
        )
        self.session.add(log)
        self.session.flush()
        return log

    def get_by_id(self, log_id: int) -> Optional[AuditLog]:
        return self.session.query(AuditLog).filter(AuditLog.id == log_id).first()

    def get_all(
        self,
        entity_type: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        q = self.session.query(AuditLog)
        if entity_type is not None:
            q = q.filter(AuditLog.entity_type == entity_type)
        if action is not None:
            q = q.filter(AuditLog.action == action)
        return q.order_by(desc(AuditLog.timestamp)).limit(limit).offset(offset).all()

    def get_recent(self, limit: int = 50) -> list[AuditLog]:
        return (
            self.session.query(AuditLog)
            .order_by(desc(AuditLog.timestamp))
            .limit(limit)
            .all()
        )
