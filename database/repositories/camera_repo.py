"""Camera repository — CRUD operations for the Camera table."""

import json
from typing import Optional

from sqlalchemy.orm import Session

from database.models import Camera


class CameraRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        name: str,
        rtsp_url: str = "",
        location: str = "",
        zone: str = "",
        status: str = "offline",
        resolution_w: int = 1280,
        resolution_h: int = 720,
        fps: int = 5,
        metadata: dict | None = None,
    ) -> Camera:
        camera = Camera(
            name=name,
            rtsp_url=rtsp_url,
            location=location,
            zone=zone,
            status=status,
            resolution_w=resolution_w,
            resolution_h=resolution_h,
            fps=fps,
            metadata_json=json.dumps(metadata or {}),
        )
        self.session.add(camera)
        self.session.flush()
        return camera

    def get_by_id(self, camera_id: int) -> Optional[Camera]:
        return self.session.query(Camera).filter(Camera.id == camera_id).first()

    def get_by_uid(self, uid: str) -> Optional[Camera]:
        return self.session.query(Camera).filter(Camera.uid == uid).first()

    def get_all(self) -> list[Camera]:
        return self.session.query(Camera).order_by(Camera.id).all()

    def get_online(self) -> list[Camera]:
        return (
            self.session.query(Camera)
            .filter(Camera.status == "online")
            .order_by(Camera.id)
            .all()
        )

    def update(self, camera_id: int, **kwargs) -> Optional[Camera]:
        camera = self.get_by_id(camera_id)
        if not camera:
            return None
        if "metadata" in kwargs:
            kwargs["metadata_json"] = json.dumps(kwargs.pop("metadata"))
        for key, value in kwargs.items():
            if hasattr(camera, key):
                setattr(camera, key, value)
        self.session.flush()
        return camera

    def delete(self, camera_id: int) -> bool:
        camera = self.get_by_id(camera_id)
        if not camera:
            return False
        self.session.delete(camera)
        self.session.flush()
        return True

    def update_status(self, camera_id: int, status: str) -> Optional[Camera]:
        return self.update(camera_id, status=status)
