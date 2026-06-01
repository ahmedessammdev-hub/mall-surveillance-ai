"""Embedding repository — CRUD operations for the Embedding table."""

from typing import Optional

from sqlalchemy.orm import Session

from database.models import Embedding


class EmbeddingRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        event_id: int,
        vector: bytes,
        model_name: str = "videomae-base",
        dimension: int = 768,
    ) -> Embedding:
        embedding = Embedding(
            event_id=event_id,
            vector_blob=vector,
            model_name=model_name,
            dimension=dimension,
        )
        self.session.add(embedding)
        self.session.flush()
        return embedding

    def get_by_id(self, embedding_id: int) -> Optional[Embedding]:
        return self.session.query(Embedding).filter(Embedding.id == embedding_id).first()

    def get_by_event_id(self, event_id: int) -> Optional[Embedding]:
        return self.session.query(Embedding).filter(Embedding.event_id == event_id).first()

    def get_all(self, limit: int = 100) -> list[Embedding]:
        return self.session.query(Embedding).order_by(Embedding.id.desc()).limit(limit).all()

    def delete(self, embedding_id: int) -> bool:
        emb = self.get_by_id(embedding_id)
        if not emb:
            return False
        self.session.delete(emb)
        self.session.flush()
        return True
