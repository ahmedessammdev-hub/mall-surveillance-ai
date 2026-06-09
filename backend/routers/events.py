"""Event listing and detail API routes."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.schemas import EventResponse
from database.connection import get_session_dependency
from database.repositories.event_repo import EventRepository

router = APIRouter()


@router.get("/", response_model=list[EventResponse])
def list_events(
    camera_id: Optional[int] = Query(None),
    event_type: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    session: Session = Depends(get_session_dependency),
):
    repo = EventRepository(session)
    return repo.get_all(
        camera_id=camera_id,
        event_type=event_type,
        risk_level=risk_level,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )


@router.get("/recent", response_model=list[EventResponse])
def recent_events(
    limit: int = Query(20, le=100),
    session: Session = Depends(get_session_dependency),
):
    repo = EventRepository(session)
    return repo.get_recent(limit=limit)


@router.get("/{event_id}", response_model=EventResponse)
def get_event(event_id: int, session: Session = Depends(get_session_dependency)):
    repo = EventRepository(session)
    event = repo.get_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event
