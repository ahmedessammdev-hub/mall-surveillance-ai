"""
Shared test fixtures.
"""

import json
import os
import sys

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    from database.models import Base

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_frame():
    """Generate a sample BGR frame (640x480)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames():
    """Generate 16 sample BGR frames for embedding."""
    return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]


@pytest.fixture
def sample_embedding():
    """Generate a random normalized 768-dim embedding."""
    vec = np.random.randn(768).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


@pytest.fixture
def sample_detections():
    """Generate sample Detection objects."""
    from ai.detector import Detection

    return [
        Detection(bbox=(100.0, 100.0, 200.0, 300.0), confidence=0.95),
        Detection(bbox=(300.0, 150.0, 400.0, 350.0), confidence=0.88),
        Detection(bbox=(500.0, 200.0, 600.0, 400.0), confidence=0.75),
    ]


@pytest.fixture
def sample_tracked_persons():
    """Generate sample TrackedPerson objects."""
    import time
    from ai.tracker import TrackedPerson, TrajectoryPoint

    now = time.time()
    return [
        TrackedPerson(
            track_id=1,
            bbox=(100.0, 100.0, 200.0, 300.0),
            confidence=0.95,
            trajectory=[
                TrajectoryPoint(x=150, y=200, timestamp=now - 2),
                TrajectoryPoint(x=155, y=205, timestamp=now - 1),
                TrajectoryPoint(x=160, y=210, timestamp=now),
            ],
            first_seen=now - 10,
            last_seen=now,
            total_frames=30,
        ),
        TrackedPerson(
            track_id=2,
            bbox=(300.0, 150.0, 400.0, 350.0),
            confidence=0.88,
            trajectory=[
                TrajectoryPoint(x=350, y=250, timestamp=now - 2),
                TrajectoryPoint(x=345, y=248, timestamp=now - 1),
                TrajectoryPoint(x=340, y=245, timestamp=now),
            ],
            first_seen=now - 5,
            last_seen=now,
            total_frames=15,
        ),
    ]
