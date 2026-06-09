"""Analytics aggregation API routes."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.schemas import AnalyticsResponse
from database.connection import get_session_dependency
from database.repositories.alert_repo import AlertRepository
from database.repositories.camera_repo import CameraRepository
from database.repositories.event_repo import EventRepository

router = APIRouter()


@router.get("/", response_model=AnalyticsResponse)
def get_analytics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    session: Session = Depends(get_session_dependency),
):
    event_repo = EventRepository(session)
    alert_repo = AlertRepository(session)
    camera_repo = CameraRepository(session)

    # Event counts
    total_events = event_repo.count(start_date=start_date, end_date=end_date)
    events_by_type = dict(event_repo.count_by_type(start_date=start_date, end_date=end_date))
    events_by_risk = dict(event_repo.count_by_risk())

    # Alert counts
    total_alerts = alert_repo.count()
    active_alerts = alert_repo.count(acknowledged=False)
    alerts_by_priority = dict(alert_repo.count_by_priority())

    # Camera counts
    all_cameras = camera_repo.get_all()
    online_cameras = camera_repo.get_online()

    return AnalyticsResponse(
        total_events=total_events,
        total_alerts=total_alerts,
        active_alerts=active_alerts,
        events_by_type=events_by_type,
        events_by_risk=events_by_risk,
        alerts_by_priority=alerts_by_priority,
        active_cameras=len(online_cameras),
        total_cameras=len(all_cameras),
    )
