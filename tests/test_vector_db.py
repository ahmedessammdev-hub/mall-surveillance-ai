"""Tests for FAISS Vector Store."""

import numpy as np
import pytest

from vector_db.faiss_store import FAISSStore, SimilarEvent


class TestFAISSStore:
    def setup_method(self):
        from config import FAISSSettings
        # Use temp settings to avoid touching real index
        settings = FAISSSettings(
            index_path="/tmp/test_faiss.index",
            metadata_path="/tmp/test_faiss_meta.json",
            dimension=768,
            top_k=5,
        )
        self.store = FAISSStore(settings)
        self.store.clear()

    def test_insert_embedding(self, sample_embedding):
        idx = self.store.insert_event_embedding(
            event_id="evt-1",
            embedding=sample_embedding,
            metadata={"event_type": "fight", "camera_id": "cam1"},
        )
        assert idx == 0
        assert self.store.get_embedding_count() == 1

    def test_search_returns_results(self, sample_embedding):
        self.store.insert_event_embedding("evt-1", sample_embedding, {"event_type": "fight"})
        self.store.insert_event_embedding("evt-2", sample_embedding * 0.99, {"event_type": "fight"})

        results = self.store.search_similar_events(sample_embedding, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SimilarEvent) for r in results)
        assert results[0].score >= results[1].score  # Sorted by score desc

    def test_search_empty_store(self, sample_embedding):
        results = self.store.search_similar_events(sample_embedding, top_k=5)
        assert results == []

    def test_retrieve_top_k(self, sample_embedding):
        for i in range(10):
            vec = np.random.randn(768).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            self.store.insert_event_embedding(f"evt-{i}", vec, {"event_type": "test"})

        results = self.store.retrieve_top_k_events(sample_embedding, k=3)
        assert len(results) == 3

    def test_clear(self, sample_embedding):
        self.store.insert_event_embedding("evt-1", sample_embedding, {})
        assert self.store.get_embedding_count() == 1
        self.store.clear()
        assert self.store.get_embedding_count() == 0

    def test_metadata_stored(self, sample_embedding):
        self.store.insert_event_embedding(
            "evt-1", sample_embedding,
            {"event_type": "fall", "camera_id": "cam2"},
        )
        results = self.store.search_similar_events(sample_embedding, top_k=1)
        assert results[0].metadata["event_type"] == "fall"
        assert results[0].metadata["camera_id"] == "cam2"
