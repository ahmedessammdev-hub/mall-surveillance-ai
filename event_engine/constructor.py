"""
Event Constructor.

Combines tracking, feature, and embedding data into high-level SecurityEvents.
Handles event deduplication across frames.
"""

import logging
import time
from typing import Optional

import numpy as np

from ai.feature_extractor import PersonFeatures, SceneFeatures
from ai.tracker import TrackedPerson
from event_engine.rules import EventRules
from event_engine.schemas import SecurityEvent

logger = logging.getLogger(__name__)


class EventConstructor:
    """Converts low-level observations into high-level security events.

    Applies detection rules, attaches embeddings, deduplicates events,
    and manages the event lifecycle.
    """

    def __init__(self, config, faiss_store=None) -> None:
        self.config = config
        self.rules = EventRules(config)
        self.faiss_store = faiss_store

        # Deduplication state: (camera_id, event_type) → last_event_time
        self._recent_events: dict[tuple[str, str], float] = {}

    def process_frame_data(
        self,
        camera_id: str,
        tracks: list[TrackedPerson],
        features: dict[str, object],
        embedding: Optional[np.ndarray] = None,
    ) -> list[SecurityEvent]:
        """Process tracking and feature data to construct events.

        Args:
            camera_id: Camera identifier.
            tracks: Current tracked persons.
            features: Dict with 'persons' (dict[int, PersonFeatures])
                      and 'scene' (SceneFeatures).
            embedding: Optional video embedding for this clip.

        Returns:
            List of detected SecurityEvents (deduplicated).
        """
        persons: dict[int, PersonFeatures] = features.get("persons", {})
        scene: SceneFeatures = features.get("scene", SceneFeatures())

        # Run all detection rules
        candidate_events = self.rules.evaluate_all(
            camera_id=camera_id,
            tracks=tracks,
            persons=persons,
            scene=scene,
        )

        # Deduplicate
        events: list[SecurityEvent] = []
        now = time.time()

        for event in candidate_events:
            key = (camera_id, event.event_type.value)
            last_time = self._recent_events.get(key, 0)

            if now - last_time < self.config.event_dedup_window:
                continue  # Skip duplicate

            self._recent_events[key] = now

            # Attach embedding
            if embedding is not None:
                event.embedding = embedding

            events.append(event)

        # Clean up old dedup entries
        cutoff = now - self.config.event_dedup_window * 2
        self._recent_events = {
            k: v for k, v in self._recent_events.items() if v > cutoff
        }

        if events:
            logger.info(
                f"Camera {camera_id}: {len(events)} event(s) detected — "
                + ", ".join(f"{e.event_type.value}({e.confidence:.2f})" for e in events)
            )

        return events
