"""Tests for YOLO11 Person Detector."""

import numpy as np
import pytest

from ai.detector import Detection, PersonDetector


class TestDetection:
    def test_detection_creation(self):
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.95)
        assert det.bbox == (10.0, 20.0, 100.0, 200.0)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.center == (55.0, 110.0)
        assert det.width == 90.0
        assert det.height == 180.0

    def test_detection_to_dict(self):
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.85)
        d = det.to_dict()
        assert d["bbox"] == [10.0, 20.0, 100.0, 200.0]
        assert d["confidence"] == 0.85
        assert d["center"] == [55.0, 110.0]

    def test_detection_to_tlwh(self):
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.9)
        tlwh = det.to_tlwh()
        assert tlwh == (10.0, 20.0, 90.0, 180.0)

    def test_detection_to_xyxy(self):
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.9)
        xyxy = det.to_xyxy()
        assert xyxy == [10.0, 20.0, 100.0, 200.0]


class TestPersonDetector:
    """Detector model tests require GPU; skip if unavailable."""

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="CUDA not available",
    )
    def test_detect_returns_list(self, sample_frame):
        from config import settings

        detector = PersonDetector(settings.detector)
        result = detector.detect(sample_frame)
        assert isinstance(result, list)
        # On a random frame, we may or may not find persons
        for det in result:
            assert isinstance(det, Detection)
            assert 0 <= det.confidence <= 1
