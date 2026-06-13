"""Tests for Database layer (repositories)."""

import pytest
from database.repositories.camera_repo import CameraRepository
from database.repositories.event_repo import EventRepository
from database.repositories.alert_repo import AlertRepository
from database.repositories.audit_repo import AuditRepository


class TestCameraRepository:
    def test_create_camera(self, in_memory_db):
        repo = CameraRepository(in_memory_db)
        cam = repo.create(name="Test Camera", rtsp_url="rtsp://test")
        assert cam.id is not None
        assert cam.name == "Test Camera"

    def test_get_by_id(self, in_memory_db):
        repo = CameraRepository(in_memory_db)
        cam = repo.create(name="Camera 1")
        assert repo.get_by_id(cam.id).name == "Camera 1"

    def test_get_all(self, in_memory_db):
        repo = CameraRepository(in_memory_db)
        repo.create(name="Cam 1")
        repo.create(name="Cam 2")
        assert len(repo.get_all()) == 2

    def test_update(self, in_memory_db):
        repo = CameraRepository(in_memory_db)
        cam = repo.create(name="Old")
        assert repo.update(cam.id, name="New").name == "New"

    def test_delete(self, in_memory_db):
        repo = CameraRepository(in_memory_db)
        cam = repo.create(name="Del")
        assert repo.delete(cam.id) is True
        assert repo.get_by_id(cam.id) is None


class TestEventRepository:
    def test_create_event(self, in_memory_db):
        cam_repo = CameraRepository(in_memory_db)
        cam = cam_repo.create(name="Cam")
        repo = EventRepository(in_memory_db)
        event = repo.create(camera_id=cam.id, event_type="fight", confidence=0.85)
        assert event.event_type == "fight"

    def test_count_by_type(self, in_memory_db):
        cam_repo = CameraRepository(in_memory_db)
        cam = cam_repo.create(name="Cam")
        repo = EventRepository(in_memory_db)
        repo.create(camera_id=cam.id, event_type="fight")
        repo.create(camera_id=cam.id, event_type="fight")
        repo.create(camera_id=cam.id, event_type="loitering")
        counts = dict(repo.count_by_type())
        assert counts["fight"] == 2
        assert counts["loitering"] == 1


class TestAlertRepository:
    def test_create_and_acknowledge(self, in_memory_db):
        cam_repo = CameraRepository(in_memory_db)
        cam = cam_repo.create(name="Cam")
        event_repo = EventRepository(in_memory_db)
        event = event_repo.create(camera_id=cam.id, event_type="fight")
        repo = AlertRepository(in_memory_db)
        alert = repo.create(event_id=event.id, priority="P1", event_type="fight")
        assert alert.acknowledged is False
        acked = repo.acknowledge(alert.id, user="admin")
        assert acked.acknowledged is True


class TestAuditRepository:
    def test_log_action(self, in_memory_db):
        repo = AuditRepository(in_memory_db)
        log = repo.log_action(action="test", entity_type="event", entity_id="1")
        assert log.action == "test"
