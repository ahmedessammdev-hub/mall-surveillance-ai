"""
VideoMAE Video Embedder.

Generates semantic video embeddings from temporal clips using
the VideoMAE model from HuggingFace. The [CLS] token from the
last hidden state is used as the 768-dimensional embedding.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VideoEmbedder:
    """Generate video embeddings using VideoMAE.

    Produces 768-dimensional embeddings from 16-frame video clips.
    These embeddings serve as the primary semantic representation
    for event retrieval and reasoning.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._load_model()

    def _load_model(self) -> None:
        """Load VideoMAE model and processor from HuggingFace."""
        from transformers import AutoImageProcessor, VideoMAEModel

        logger.info(f"Loading VideoMAE model: {self.config.model_name}")

        self._processor = AutoImageProcessor.from_pretrained(self.config.model_name)
        self._model = VideoMAEModel.from_pretrained(self.config.model_name)

        # Determine device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
            if self.config.device == "cuda":
                logger.warning("CUDA not available, falling back to CPU for VideoMAE")

        self._model = self._model.to(self._device)
        self._model.eval()

        if self.config.use_fp16 and self._device.type == "cuda":
            self._model = self._model.half()

        logger.info(
            f"VideoMAE loaded on {self._device} "
            f"(fp16={self.config.use_fp16 and self._device.type == 'cuda'})"
        )

    def generate_embedding(self, frames: list[np.ndarray]) -> np.ndarray:
        """Generate a single embedding from a list of video frames.

        Args:
            frames: List of BGR numpy arrays (H, W, C). Will be sampled
                    to self.config.num_frames if longer.

        Returns:
            Normalized embedding vector of shape (embedding_dim,).
        """
        # Sample frames to target count
        sampled = self._sample_frames(frames, self.config.num_frames)

        # Convert BGR to RGB and ensure contiguous memory layout
        rgb_frames = [np.ascontiguousarray(f[:, :, ::-1]) for f in sampled]  # BGR → RGB, copy

        # Process through VideoMAE processor
        inputs = self._processor(rgb_frames, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        if self.config.use_fp16 and self._device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Extract [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_dim)
        embedding = embedding.float().cpu().numpy().flatten()

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def generate_batch(self, clips: list[list[np.ndarray]]) -> np.ndarray:
        """Generate embeddings for multiple clips.

        Args:
            clips: List of clip frame lists.

        Returns:
            Array of shape (N, embedding_dim) with normalized embeddings.
        """
        embeddings = []
        for i in range(0, len(clips), self.config.batch_size):
            batch = clips[i : i + self.config.batch_size]
            for clip in batch:
                emb = self.generate_embedding(clip)
                embeddings.append(emb)

        return np.array(embeddings)

    @staticmethod
    def _sample_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
        """Uniformly sample frames to target count.

        If fewer frames are available, frames are repeated to reach target.
        """
        n = len(frames)
        if n == 0:
            raise ValueError("Cannot generate embedding from empty frame list")

        if n == target_count:
            return frames

        if n > target_count:
            # Uniform sampling
            indices = np.linspace(0, n - 1, target_count, dtype=int)
            return [frames[i] for i in indices]

        # Fewer frames than target — repeat to fill
        sampled = []
        for i in range(target_count):
            idx = int(i * n / target_count)
            sampled.append(frames[idx])
        return sampled

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def device(self) -> str:
        return str(self._device)
