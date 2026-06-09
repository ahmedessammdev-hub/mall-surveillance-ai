"""Track listing API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.schemas import TrackResponse
from database.connection import get_session_dependency
from database.repositories.track_repo import TrackRepository

router = APIRouter()


@router.get("/", response_model=list[TrackResponse])
def list_tracks(
    camera_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    session: Session = Depends(get_session_dependency),
):
    repo = TrackRepository(session)
    return repo.get_all(camera_id=camera_id, status=status)


@router.get("/active", response_model=list[TrackResponse])
def active_tracks(
    camera_id: Optional[int] = Query(None),
    session: Session = Depends(get_session_dependency),
):
    repo = TrackRepository(session)
    return repo.get_active(camera_id=camera_id)


@router.get("/count")
def track_count(
    camera_id: Optional[int] = Query(None),
    session: Session = Depends(get_session_dependency),
):
    repo = TrackRepository(session)
    return {"active_tracks": repo.count_active(camera_id=camera_id)}
