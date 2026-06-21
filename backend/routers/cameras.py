"""Camera management API routes."""

import io
import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.schemas import CameraCreate, CameraResponse, CameraUpdate
from database.connection import get_session_dependency
from database.repositories.camera_repo import CameraRepository

router = APIRouter()


@router.get("/", response_model=list[CameraResponse])
def list_cameras(session: Session = Depends(get_session_dependency)):
    repo = CameraRepository(session)
    return repo.get_all()


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: int, session: Session = Depends(get_session_dependency)):
    repo = CameraRepository(session)
    camera = repo.get_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.post("/", response_model=CameraResponse, status_code=201)
def create_camera(
    data: CameraCreate,
    request: Request,
    session: Session = Depends(get_session_dependency),
):
    repo = CameraRepository(session)
    camera = repo.create(
        name=data.name,
        rtsp_url=data.rtsp_url,
        location=data.location,
        zone=data.zone,
        resolution_w=data.resolution_w,
        resolution_h=data.resolution_h,
        fps=data.fps,
    )

    # Try to start the stream
    app_state = request.app.state.app_state
    if app_state and app_state.stream_manager and data.rtsp_url:
        success = app_state.stream_manager.add_camera(
            camera_id=str(camera.id),
            source=data.rtsp_url,
            target_fps=data.fps,
        )
        if success:
            repo.update_status(camera.id, "online")
            camera.status = "online"

    return camera


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: int,
    data: CameraUpdate,
    session: Session = Depends(get_session_dependency),
):
    repo = CameraRepository(session)
    update_data = data.model_dump(exclude_unset=True)
    camera = repo.update(camera_id, **update_data)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.delete("/{camera_id}", status_code=204)
def delete_camera(
    camera_id: int,
    request: Request,
    session: Session = Depends(get_session_dependency),
):
    repo = CameraRepository(session)
    if not repo.delete(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")

    # Stop the stream
    app_state = request.app.state.app_state
    if app_state and app_state.stream_manager:
        app_state.stream_manager.remove_camera(str(camera_id))


@router.get("/{camera_id}/frame")
def get_camera_frame(
    camera_id: int,
    request: Request,
):
    """Return the latest frame from a camera as JPEG with detection overlays."""
    app_state = request.app.state.app_state
    if not app_state:
        raise HTTPException(status_code=503, detail="App not ready")

    # Get latest frame
    frame = app_state.latest_frames.get(str(camera_id))
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")

    # Draw detection overlays
    tracks = app_state.latest_tracks.get(str(camera_id), [])
    for track in tracks:
        x1, y1, x2, y2 = [int(c) for c in track.bbox]
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (139, 92, 246), 2)
        # Draw track ID and confidence
        label = f"ID:{track.track_id} {track.confidence:.0%}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 92, 246), 2)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    io_buf = io.BytesIO(buffer.tobytes())

    return StreamingResponse(io_buf, media_type="image/jpeg")
