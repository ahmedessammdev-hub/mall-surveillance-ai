"""Semantic event search API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.schemas import SearchRequest, SearchResult
from database.connection import get_session_dependency
from database.repositories.event_repo import EventRepository
from database.repositories.embedding_repo import EmbeddingRepository

import numpy as np

router = APIRouter()


@router.post("/", response_model=list[SearchResult])
def search_events(
    data: SearchRequest,
    request: Request,
    session: Session = Depends(get_session_dependency),
):
    """Search for similar events using embedding-based retrieval."""
    app_state = request.app.state.app_state

    if not app_state or not app_state.faiss_store:
        raise HTTPException(status_code=503, detail="FAISS store not available")

    if app_state.faiss_store.get_embedding_count() == 0:
        return []

    # If searching by event ID, get its embedding
    if data.query_event_id:
        event_repo = EventRepository(session)
        emb_repo = EmbeddingRepository(session)

        event = event_repo.get_by_uid(data.query_event_id)
        if not event:
            # Try by int ID
            try:
                event = event_repo.get_by_id(int(data.query_event_id))
            except (ValueError, TypeError):
                pass

        if not event:
            raise HTTPException(status_code=404, detail="Source event not found")

        embedding_rec = emb_repo.get_by_event_id(event.id)
        if not embedding_rec:
            raise HTTPException(status_code=404, detail="No embedding for this event")

        query_vec = np.frombuffer(embedding_rec.vector_blob, dtype=np.float32)
    else:
        raise HTTPException(
            status_code=400,
            detail="query_event_id is required for search",
        )

    # Search FAISS
    results = app_state.faiss_store.search_similar_events(
        query_vec, top_k=data.top_k
    )

    return [
        SearchResult(
            event_id=r.event_id,
            score=r.score,
            event_type=r.metadata.get("event_type", ""),
            camera_id=r.metadata.get("camera_id", ""),
            timestamp=r.metadata.get("timestamp", ""),
        )
        for r in results
    ]
