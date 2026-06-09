"""Alert listing, filtering, and acknowledgement API routes."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.schemas import AlertAcknowledge, AlertResponse
from database.connection import get_session_dependency
from database.repositories.alert_repo import AlertRepository

router = APIRouter()


@router.get("/", response_model=list[AlertResponse])
def list_alerts(
    priority: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    event_type: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    session: Session = Depends(get_session_dependency),
):
    repo = AlertRepository(session)
    return repo.get_all(
        priority=priority,
        acknowledged=acknowledged,
        event_type=event_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )


@router.get("/active", response_model=list[AlertResponse])
def active_alerts(
    limit: int = Query(50, le=200),
    session: Session = Depends(get_session_dependency),
):
    repo = AlertRepository(session)
    return repo.get_active(limit=limit)


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: int, session: Session = Depends(get_session_dependency)):
    repo = AlertRepository(session)
    alert = repo.get_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(
    alert_id: int,
    data: AlertAcknowledge,
    session: Session = Depends(get_session_dependency),
):
    repo = AlertRepository(session)
    alert = repo.acknowledge(alert_id, user=data.user)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert
