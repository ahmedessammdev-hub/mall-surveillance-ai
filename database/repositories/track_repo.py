"""Track repository — CRUD operations for the Track table."""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from database.models import Track


class TrackRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        camera_id: int,
        track_id: int,
        trajectory: list | None = None,
        status: str = "active",
    ) -> Track:
        track = Track(
            camera_id=camera_id,
            track_id=track_id,
            trajectory_json=json.dumps(trajectory or []),
            status=status,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
        )
        self.session.add(track)
        self.session.flush()
        return track

    def get_by_id(self, id: int) -> Optional[Track]:
        return self.session.query(Track).filter(Track.id == id).first()

    def get_by_track_id(self, camera_id: int, track_id: int) -> Optional[Track]:
        return (
            self.session.query(Track)
            .filter(Track.camera_id == camera_id, Track.track_id == track_id)
            .first()
        )

    def get_all(self, camera_id: int | None = None, status: str | None = None) -> list[Track]:
        q = self.session.query(Track)
        if camera_id is not None:
            q = q.filter(Track.camera_id == camera_id)
        if status is not None:
            q = q.filter(Track.status == status)
        return q.order_by(Track.last_seen.desc()).all()

    def get_active(self, camera_id: int | None = None) -> list[Track]:
        return self.get_all(camera_id=camera_id, status="active")

    def update(self, id: int, **kwargs) -> Optional[Track]:
        track = self.get_by_id(id)
        if not track:
            return None
        if "trajectory" in kwargs:
            kwargs["trajectory_json"] = json.dumps(kwargs.pop("trajectory"))
        for key, value in kwargs.items():
            if hasattr(track, key):
                setattr(track, key, value)
        self.session.flush()
        return track

    def update_or_create(
        self,
        camera_id: int,
        track_id: int,
        trajectory: list | None = None,
        avg_speed: float = 0.0,
        max_speed: float = 0.0,
    ) -> Track:
        existing = self.get_by_track_id(camera_id, track_id)
        if existing:
            existing.last_seen = datetime.utcnow()
            existing.total_frames += 1
            existing.avg_speed = avg_speed
            existing.max_speed = max_speed
            if trajectory:
                existing.trajectory_json = json.dumps(trajectory)
            self.session.flush()
            return existing
        return self.create(camera_id=camera_id, track_id=track_id, trajectory=trajectory)

    def mark_lost(self, id: int) -> Optional[Track]:
        return self.update(id, status="lost")

    def delete(self, id: int) -> bool:
        track = self.get_by_id(id)
        if not track:
            return False
        self.session.delete(track)
        self.session.flush()
        return True

    def count_active(self, camera_id: int | None = None) -> int:
        q = self.session.query(func.count(Track.id)).filter(Track.status == "active")
        if camera_id is not None:
            q = q.filter(Track.camera_id == camera_id)
        return q.scalar() or 0
