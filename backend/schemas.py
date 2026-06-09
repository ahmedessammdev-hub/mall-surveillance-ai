"""
API Request/Response Pydantic schemas.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Camera Schemas
# ---------------------------------------------------------------------------

class CameraCreate(BaseModel):
    name: str
    rtsp_url: str = ""
    location: str = ""
    zone: str = ""
    resolution_w: int = 1280
    resolution_h: int = 720
    fps: int = 5


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    location: Optional[str] = None
    zone: Optional[str] = None
    status: Optional[str] = None
    resolution_w: Optional[int] = None
    resolution_h: Optional[int] = None
    fps: Optional[int] = None


class CameraResponse(BaseModel):
    id: int
    uid: str
    name: str
    rtsp_url: str = ""
    location: str = ""
    zone: str = ""
    status: str = "offline"
    resolution_w: int = 1280
    resolution_h: int = 720
    fps: int = 5
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Event Schemas
# ---------------------------------------------------------------------------

class EventResponse(BaseModel):
    id: int
    uid: str
    camera_id: int
    event_type: str
    timestamp: datetime
    location: str = ""
    person_count: int = 0
    crowd_density: float = 0.0
    confidence: float = 0.0
    risk_level: str = "low"
    involved_tracks_json: str = "[]"
    behavior_scores_json: str = "{}"
    motion_features_json: str = "{}"
    similar_events_json: str = "[]"
    reasoning_json: str = "{}"
    created_at: datetime

    class Config:
        from_attributes = True


class EventFilter(BaseModel):
    camera_id: Optional[int] = None
    event_type: Optional[str] = None
    risk_level: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=50, le=200)
    offset: int = 0


# ---------------------------------------------------------------------------
# Alert Schemas
# ---------------------------------------------------------------------------

class AlertResponse(BaseModel):
    id: int
    uid: str
    event_id: int
    priority: str
    event_type: str
    confidence: float = 0.0
    risk_level: str = "low"
    timestamp: datetime
    reasoning: str = ""
    recommended_action: str = ""
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AlertAcknowledge(BaseModel):
    user: str = "operator"


class AlertFilter(BaseModel):
    priority: Optional[str] = None
    acknowledged: Optional[bool] = None
    event_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=50, le=200)
    offset: int = 0


# ---------------------------------------------------------------------------
# Track Schemas
# ---------------------------------------------------------------------------

class TrackResponse(BaseModel):
    id: int
    uid: str
    camera_id: int
    track_id: int
    first_seen: datetime
    last_seen: datetime
    status: str = "active"
    total_frames: int = 0
    avg_speed: float = 0.0
    max_speed: float = 0.0

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Search Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query_event_id: Optional[str] = None
    event_type: Optional[str] = None
    top_k: int = Field(default=10, le=50)


class SearchResult(BaseModel):
    event_id: str
    score: float
    event_type: str = ""
    camera_id: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Analytics Schemas
# ---------------------------------------------------------------------------

class AnalyticsResponse(BaseModel):
    total_events: int = 0
    total_alerts: int = 0
    active_alerts: int = 0
    events_by_type: dict[str, int] = Field(default_factory=dict)
    events_by_risk: dict[str, int] = Field(default_factory=dict)
    alerts_by_priority: dict[str, int] = Field(default_factory=dict)
    active_cameras: int = 0
    total_cameras: int = 0


# ---------------------------------------------------------------------------
# System Health Schemas
# ---------------------------------------------------------------------------

class SystemHealthResponse(BaseModel):
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu_name: str = ""
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization: float = 0.0
    active_streams: int = 0
    processing_fps: dict[str, float] = Field(default_factory=dict)
    faiss_vectors: int = 0
    uptime_seconds: float = 0.0
