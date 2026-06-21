"""
SQLAlchemy ORM models for the surveillance system.

Tables: Camera, Track, Event, Alert, Embedding, AuditLog
"""

import datetime
import uuid
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _generate_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(36), default=_generate_uuid, unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    rtsp_url = Column(String(512), nullable=True)
    location = Column(String(255), nullable=True)
    zone = Column(String(100), nullable=True)
    status = Column(String(20), default="offline", nullable=False)
    resolution_w = Column(Integer, default=1280)
    resolution_h = Column(Integer, default=720)
    fps = Column(Integer, default=5)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    tracks = relationship("Track", back_populates="camera", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="camera", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, name='{self.name}', status='{self.status}')>"


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(36), default=_generate_uuid, unique=True, nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    track_id = Column(Integer, nullable=False, doc="ByteTrack assigned ID")
    first_seen = Column(DateTime, default=func.now(), nullable=False)
    last_seen = Column(DateTime, default=func.now(), nullable=False)
    trajectory_json = Column(Text, default="[]", doc="List of (x,y,t) points")
    status = Column(String(20), default="active", doc="active|lost|removed")
    total_frames = Column(Integer, default=0)
    avg_speed = Column(Float, default=0.0)
    max_speed = Column(Float, default=0.0)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    camera = relationship("Camera", back_populates="tracks")

    def __repr__(self) -> str:
        return f"<Track(id={self.id}, track_id={self.track_id}, status='{self.status}')>"


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------
class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(36), default=_generate_uuid, unique=True, nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    event_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    location = Column(String(255), nullable=True)
    involved_tracks_json = Column(Text, default="[]")
    person_count = Column(Integer, default=0)
    crowd_density = Column(Float, default=0.0)
    behavior_scores_json = Column(Text, default="{}")
    interaction_scores_json = Column(Text, default="{}")
    motion_features_json = Column(Text, default="{}")
    embedding_id = Column(Integer, ForeignKey("embeddings.id"), nullable=True)
    similar_events_json = Column(Text, default="[]")
    confidence = Column(Float, default=0.0)
    risk_level = Column(String(20), default="low")
    reasoning_json = Column(Text, default="{}", doc="LLM reasoning result")
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    camera = relationship("Camera", back_populates="events")
    alerts = relationship("Alert", back_populates="event", cascade="all, delete-orphan")
    embedding = relationship("Embedding", uselist=False, foreign_keys=[embedding_id])

    def __repr__(self) -> str:
        return (
            f"<Event(id={self.id}, type='{self.event_type}', "
            f"confidence={self.confidence:.2f}, risk='{self.risk_level}')>"
        )


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(36), default=_generate_uuid, unique=True, nullable=False)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    priority = Column(String(5), nullable=False, doc="P1, P2, P3")
    event_type = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.0)
    risk_level = Column(String(20), default="low")
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    reasoning = Column(Text, default="")
    recommended_action = Column(Text, default="")
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    event = relationship("Event", back_populates="alerts")

    def __repr__(self) -> str:
        return (
            f"<Alert(id={self.id}, priority='{self.priority}', "
            f"type='{self.event_type}', ack={self.acknowledged})>"
        )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(36), default=_generate_uuid, unique=True, nullable=False)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=True)
    vector_blob = Column(LargeBinary, nullable=False, doc="Raw float32 bytes")
    model_name = Column(String(255), default="videomae-base")
    dimension = Column(Integer, default=768)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    event = relationship("Event", foreign_keys=[event_id], viewonly=True)

    def __repr__(self) -> str:
        return f"<Embedding(id={self.id}, dim={self.dimension}, model='{self.model_name}')>"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(50), nullable=True)
    details_json = Column(Text, default="{}")
    user = Column(String(100), default="system")
    timestamp = Column(DateTime, default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action='{self.action}', entity='{self.entity_type}')>"
