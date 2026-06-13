"""Tests for FastAPI API endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.app import create_app


@pytest.fixture
def client():
    app = create_app(app_state=None)
    return TestClient(app)


class TestRootEndpoint:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"


class TestCameraEndpoints:
    def test_list_cameras(self, client):
        resp = client.get("/api/cameras/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_camera(self, client):
        resp = client.post("/api/cameras/", json={"name": "Test Cam", "rtsp_url": ""})
        assert resp.status_code == 201
        assert resp.json()["name"] == "Test Cam"

    def test_get_camera_not_found(self, client):
        resp = client.get("/api/cameras/9999")
        assert resp.status_code == 404


class TestEventEndpoints:
    def test_list_events(self, client):
        resp = client.get("/api/events/")
        assert resp.status_code == 200

    def test_recent_events(self, client):
        resp = client.get("/api/events/recent")
        assert resp.status_code == 200


class TestAlertEndpoints:
    def test_list_alerts(self, client):
        resp = client.get("/api/alerts/")
        assert resp.status_code == 200

    def test_active_alerts(self, client):
        resp = client.get("/api/alerts/active")
        assert resp.status_code == 200


class TestHealthEndpoint:
    def test_system_health(self, client):
        resp = client.get("/api/system-health/")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "ram_percent" in data


class TestAnalyticsEndpoint:
    def test_analytics(self, client):
        resp = client.get("/api/analytics/")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_events" in data
