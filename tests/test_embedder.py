"""Tests for VideoMAE Embedder."""

import numpy as np
import pytest

from ai.embedder import VideoEmbedder


class TestVideoEmbedder:
    """Embedder tests require GPU + model download; skip if unavailable."""

    @pytest.fixture
    def embedder(self):
        """Create embedder — skip if model can't be loaded."""
        try:
            from config import settings
            return VideoEmbedder(settings.embedder)
        except Exception:
            pytest.skip("VideoMAE model not available")

    @pytest.mark.slow
    def test_embedding_shape(self, embedder, sample_frames):
        embedding = embedder.generate_embedding(sample_frames)
        assert embedding.shape == (768,)

    @pytest.mark.slow
    def test_embedding_normalized(self, embedder, sample_frames):
        embedding = embedder.generate_embedding(sample_frames)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_frame_sampling_exact(self):
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 16
        result = VideoEmbedder._sample_frames(frames, 16)
        assert len(result) == 16

    def test_frame_sampling_more(self):
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 32
        result = VideoEmbedder._sample_frames(frames, 16)
        assert len(result) == 16

    def test_frame_sampling_fewer(self):
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 8
        result = VideoEmbedder._sample_frames(frames, 16)
        assert len(result) == 16

    def test_frame_sampling_empty(self):
        with pytest.raises(ValueError):
            VideoEmbedder._sample_frames([], 16)
