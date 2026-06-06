"""
FAISS Vector Store for event embeddings.

Supports insert, search (top-k), persistence, and metadata storage.
Uses inner product (cosine similarity on L2-normalized vectors).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimilarEvent:
    """A retrieved similar event with its similarity score."""
    event_id: str
    score: float
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


class FAISSStore:
    """FAISS-based vector store for event embeddings.

    Stores event embeddings and metadata, supports similarity search.
    Uses IndexFlatIP (inner product) for cosine similarity on
    L2-normalized vectors.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.dimension = config.dimension
        self._index_path = Path(config.index_path)
        self._metadata_path = Path(config.metadata_path)

        # Parallel storage: FAISS index + metadata dict
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self._metadata: dict[int, dict] = {}  # FAISS internal ID → metadata
        self._event_id_map: dict[str, int] = {}  # event_id → FAISS internal ID
        self._next_id = 0

        # Load existing index if available
        self._load_index()

    def insert_event_embedding(
        self,
        event_id: str,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> int:
        """Insert an event embedding into the store.

        Args:
            event_id: Unique event identifier.
            embedding: L2-normalized embedding vector.
            metadata: Associated metadata (event_type, camera_id, etc.).

        Returns:
            Internal FAISS index ID.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Ensure float32
        embedding = embedding.astype(np.float32)

        # L2 normalize (should already be normalized, but ensure it)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embedding = embedding / norms

        # Add to FAISS index
        self._index.add(embedding)
        internal_id = self._next_id
        self._next_id += 1

        # Store metadata
        meta = metadata or {}
        meta["event_id"] = event_id
        self._metadata[internal_id] = meta
        self._event_id_map[event_id] = internal_id

        logger.debug(f"Inserted embedding for event {event_id} (index={internal_id})")
        return internal_id

    def search_similar_events(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[SimilarEvent]:
        """Search for events similar to the query embedding.

        Args:
            query_embedding: L2-normalized query vector.
            top_k: Number of results to return.

        Returns:
            List of SimilarEvent sorted by descending similarity.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k or self.config.top_k, self._index.ntotal)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        # L2 normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Search
        scores, indices = self._index.search(query_embedding, k)

        results: list[SimilarEvent] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata.get(int(idx), {})
            results.append(SimilarEvent(
                event_id=meta.get("event_id", ""),
                score=float(score),
                metadata=meta,
            ))

        return results

    def retrieve_top_k_events(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[SimilarEvent]:
        """Convenience alias for search_similar_events."""
        return self.search_similar_events(query_embedding, top_k=k)

    def get_embedding_count(self) -> int:
        """Return total number of embeddings in the store."""
        return self._index.ntotal

    def save_index(self) -> None:
        """Persist FAISS index and metadata to disk."""
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))

            meta_data = {
                "metadata": {str(k): v for k, v in self._metadata.items()},
                "event_id_map": self._event_id_map,
                "next_id": self._next_id,
            }
            with open(self._metadata_path, "w") as f:
                json.dump(meta_data, f, indent=2, default=str)

            logger.info(
                f"FAISS index saved: {self._index.ntotal} vectors → {self._index_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not self._index_path.exists():
            logger.info("No existing FAISS index found, starting fresh")
            return

        try:
            self._index = faiss.read_index(str(self._index_path))

            if self._metadata_path.exists():
                with open(self._metadata_path, "r") as f:
                    data = json.load(f)
                self._metadata = {int(k): v for k, v in data.get("metadata", {}).items()}
                self._event_id_map = data.get("event_id_map", {})
                self._next_id = data.get("next_id", self._index.ntotal)

            logger.info(
                f"FAISS index loaded: {self._index.ntotal} vectors from {self._index_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self._index = faiss.IndexFlatIP(self.dimension)
            self._metadata = {}
            self._event_id_map = {}
            self._next_id = 0

    def clear(self) -> None:
        """Clear the entire index."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata.clear()
        self._event_id_map.clear()
        self._next_id = 0
        logger.info("FAISS index cleared")
