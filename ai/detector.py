"""
YOLO11 Person Detector.

Detects persons in frames using Ultralytics YOLO11.
Returns bounding boxes, confidence scores, and class IDs.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single person detection result."""
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixel coordinates
    confidence: float
    class_id: int = 0  # COCO person class
    center: tuple[float, float] = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.width = x2 - x1
        self.height = y2 - y1

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "center": list(self.center),
        }

    def to_tlwh(self) -> tuple[float, float, float, float]:
        """Convert to (top-left-x, top-left-y, width, height)."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)

    def to_xyxy(self) -> list[float]:
        return list(self.bbox)


class PersonDetector:
    """YOLO11 person detection service.

    Uses the Ultralytics YOLO library to detect persons in video frames.
    Filters detections to only include person class (COCO class 0).
    """

    def __init__(self, config) -> None:
        self.config = config
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model. Lazy-imports ultralytics to avoid import-time GPU init."""
        from ultralytics import YOLO

        logger.info(f"Loading YOLO model: {self.config.model_name}")
        self._model = YOLO(self.config.model_name)

        # Warm up with a dummy image
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            device=self.config.device,
            verbose=False,
            conf=self.config.confidence_threshold,
        )
        logger.info("YOLO model loaded and warmed up")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect persons in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, C).

        Returns:
            List of Detection objects for persons found.
        """
        results = self._model.predict(
            frame,
            device=self.config.device,
            verbose=False,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            classes=[self.config.person_class_id],
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                if cls_id == self.config.person_class_id:
                    detections.append(
                        Detection(
                            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                            confidence=conf,
                            class_id=cls_id,
                        )
                    )

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detect persons in multiple frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of detection lists, one per frame.
        """
        all_detections: list[list[Detection]] = []
        results_list = self._model.predict(
            frames,
            device=self.config.device,
            verbose=False,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            classes=[self.config.person_class_id],
        )

        for results in results_list:
            frame_dets: list[Detection] = []
            if results.boxes is not None:
                boxes = results.boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    if cls_id == self.config.person_class_id:
                        frame_dets.append(
                            Detection(
                                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                                confidence=conf,
                                class_id=cls_id,
                            )
                        )
            all_detections.append(frame_dets)

        return all_detections

    @property
    def model(self):
        """Access underlying YOLO model (for tracking integration)."""
        return self._model
