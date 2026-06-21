"""
Mall Surveillance AI — Application Entrypoint

Initializes all subsystems and runs the processing pipeline.
Usage:
    uv run python main.py
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from config import settings
from database.connection import init_db, get_session
from database.repositories.camera_repo import CameraRepository
from stream.manager import StreamManager
from ai.detector import PersonDetector
from ai.tracker import MultiPersonTracker
from ai.feature_extractor import FeatureExtractor
from ai.embedder import VideoEmbedder
from event_engine.constructor import EventConstructor
from vector_db.faiss_store import FAISSStore
from reasoning.engine import ReasoningEngine
from alerts.engine import AlertEngine
from database.repositories.event_repo import EventRepository
from database.repositories.alert_repo import AlertRepository
from database.repositories.track_repo import TrackRepository
from database.repositories.embedding_repo import EmbeddingRepository
from database.repositories.audit_repo import AuditRepository

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mall_surveillance")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
class AppState:
    """Holds all runtime components. Passed around via dependency injection."""

    def __init__(self) -> None:
        self.running = True
        self.stream_manager: StreamManager | None = None
        self.detector: PersonDetector | None = None
        self.tracker_pool: dict[str, MultiPersonTracker] = {}
        self.feature_extractor: FeatureExtractor | None = None
        self.embedder: VideoEmbedder | None = None
        self.event_constructor: EventConstructor | None = None
        self.faiss_store: FAISSStore | None = None
        self.reasoning_engine: ReasoningEngine | None = None
        self.alert_engine: AlertEngine | None = None

        # Async event handling
        self.event_executor: ThreadPoolExecutor | None = None

        # LLM cooldown: (camera_id, event_type) -> last trigger time
        self._llm_cooldown: dict[tuple[str, str], float] = {}
        self._llm_cooldown_seconds: float = 30.0

        # Latest processed data (for dashboard consumption)
        self.latest_frames: dict[str, "object"] = {}
        self.latest_detections: dict[str, list] = {}
        self.latest_tracks: dict[str, list] = {}
        self.processing_fps: dict[str, float] = {}


app_state = AppState()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def init_components() -> None:
    """Initialize all AI/ML components and services."""
    logger.info("=" * 60)
    logger.info("  Mall Surveillance AI — Initializing")
    logger.info("=" * 60)

    # 1. Database
    logger.info("[1/8] Initializing database...")
    init_db()
    logger.info("  ✓ Database ready")

    # 2. Stream Manager
    logger.info("[2/8] Initializing stream manager...")
    app_state.stream_manager = StreamManager(settings.stream)
    logger.info("  ✓ Stream manager ready")

    # 3. Person Detector
    logger.info("[3/8] Loading YOLO11 person detector...")
    app_state.detector = PersonDetector(settings.detector)
    logger.info("  ✓ Detector ready")

    # 4. Feature Extractor
    logger.info("[4/8] Initializing feature extractor...")
    app_state.feature_extractor = FeatureExtractor(
        ema_alpha=settings.feature_extractor.ema_alpha,
        zones=settings.feature_extractor.zones,
        restricted_zones=settings.feature_extractor.restricted_zones,
    )
    logger.info("  ✓ Feature extractor ready")

    # 5. Video Embedder
    logger.info("[5/8] Loading VideoMAE embedder...")
    app_state.embedder = VideoEmbedder(settings.embedder)
    logger.info("  ✓ Embedder ready")

    # 6. FAISS Store
    logger.info("[6/8] Initializing FAISS vector store...")
    app_state.faiss_store = FAISSStore(settings.faiss)
    logger.info("  ✓ FAISS store ready")

    # 7. Event Engine
    logger.info("[7/8] Initializing event engine...")
    app_state.event_constructor = EventConstructor(
        settings.event,
        app_state.faiss_store,
    )
    logger.info("  ✓ Event engine ready")

    # 8. Reasoning & Alerts
    logger.info("[8/8] Initializing reasoning & alert engines...")
    app_state.reasoning_engine = ReasoningEngine(settings.llm)
    app_state.alert_engine = AlertEngine()
    logger.info("  ✓ Reasoning & alert engines ready")

    # 9. Event executor (async event handling)
    app_state.event_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="event-handler")

    # Load cameras from database
    _load_cameras_from_db()

    logger.info("=" * 60)
    logger.info("  All components initialized successfully!")
    logger.info("=" * 60)


def _load_cameras_from_db() -> None:
    """Load previously configured cameras from the database."""
    with get_session() as session:
        repo = CameraRepository(session)
        cameras = repo.get_all()
        for cam in cameras:
            if cam.status != "offline" and cam.rtsp_url:
                try:
                    app_state.stream_manager.add_camera(
                        camera_id=str(cam.id),
                        source=cam.rtsp_url,
                    )
                    logger.info(f"  Loaded camera: {cam.name} ({cam.rtsp_url})")
                except Exception as e:
                    logger.warning(f"  Failed to load camera {cam.name}: {e}")


# ---------------------------------------------------------------------------
# Processing Loop
# ---------------------------------------------------------------------------
def processing_loop() -> None:
    """Main processing loop — runs in a dedicated thread."""
    logger.info("Processing loop started")
    clip_buffers: dict[str, list] = {}  # camera_id -> list of frames for embedding
    max_clip_buffer = 64  # Prevent unbounded growth

    while app_state.running:
        try:
            frames = app_state.stream_manager.get_all_frames()
            if not frames:
                time.sleep(0.1)
                continue

            # Process cameras in parallel
            camera_ids = [cid for cid, f in frames.items() if f is not None]
            if len(camera_ids) > 1:
                with ThreadPoolExecutor(max_workers=len(camera_ids)) as cam_pool:
                    futures = {
                        cam_pool.submit(_process_single_camera, cid, frames[cid], clip_buffers): cid
                        for cid in camera_ids
                    }
                    for future in futures:
                        try:
                            future.result(timeout=5.0)
                        except Exception as e:
                            logger.error(f"Camera {futures[future]} processing error: {e}")
            else:
                for cid in camera_ids:
                    _process_single_camera(cid, frames[cid], clip_buffers)

        except Exception as e:
            logger.error(f"Processing loop error: {e}", exc_info=True)
            time.sleep(0.5)


def _process_single_camera(
    camera_id: str,
    frame: object,
    clip_buffers: dict[str, list],
) -> None:
    """Process a single camera frame through the full pipeline."""
    max_clip_buffer = 64

    t_start = time.perf_counter()

    # Store latest frame for dashboard
    app_state.latest_frames[camera_id] = frame

    # --- Detection ---
    detections = app_state.detector.detect(frame)
    app_state.latest_detections[camera_id] = detections

    # --- Tracking ---
    if camera_id not in app_state.tracker_pool:
        app_state.tracker_pool[camera_id] = MultiPersonTracker(
            settings.tracker
        )
    tracker = app_state.tracker_pool[camera_id]
    tracks = tracker.update(detections)
    app_state.latest_tracks[camera_id] = tracks

    # --- Feature Extraction ---
    features = app_state.feature_extractor.extract(
        tracks, frame.shape[:2]
    )

    # --- Clip buffering for embeddings ---
    if camera_id not in clip_buffers:
        clip_buffers[camera_id] = []
    clip_buffers[camera_id].append(frame)

    # Safety: trim buffer if too large
    if len(clip_buffers[camera_id]) > max_clip_buffer:
        clip_buffers[camera_id] = clip_buffers[camera_id][-max_clip_buffer:]
        logger.warning(f"Clip buffer for camera {camera_id} exceeded {max_clip_buffer}, trimmed")

    # When we have enough frames, generate embedding
    embedding = None
    if len(clip_buffers[camera_id]) >= settings.embedder.num_frames:
        clip = clip_buffers[camera_id][: settings.embedder.num_frames]
        clip_buffers[camera_id] = clip_buffers[camera_id][
            settings.embedder.num_frames // 2 :
        ]  # sliding window overlap
        try:
            embedding = app_state.embedder.generate_embedding(clip)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")

    # --- Event Construction ---
    events = app_state.event_constructor.process_frame_data(
        camera_id=camera_id,
        tracks=tracks,
        features=features,
        embedding=embedding,
    )

    # --- Submit events to async handler ---
    for event in events:
        if app_state.event_executor:
            app_state.event_executor.submit(_handle_event, event, camera_id)
        else:
            _handle_event(event, camera_id)

    # FPS tracking
    elapsed = time.perf_counter() - t_start
    app_state.processing_fps[camera_id] = (
        1.0 / elapsed if elapsed > 0 else 0.0
    )


def _handle_event(event, camera_id: str) -> None:
    """Handle a detected event: store, retrieve similar, reason, alert."""
    try:
        with get_session() as session:
            event_repo = EventRepository(session)
            alert_repo = AlertRepository(session)
            embedding_repo = EmbeddingRepository(session)
            audit_repo = AuditRepository(session)

            # Store event in DB
            db_event = event_repo.create_from_schema(event)

            # Store embedding if available
            if event.embedding is not None:
                embedding_repo.create(
                    event_id=db_event.id,
                    vector=event.embedding.tobytes(),
                    model_name=settings.embedder.model_name,
                    dimension=settings.embedder.embedding_dim,
                )

                # Insert into FAISS
                app_state.faiss_store.insert_event_embedding(
                    event_id=str(db_event.id),
                    embedding=event.embedding,
                    metadata={
                        "event_type": event.event_type.value,
                        "camera_id": camera_id,
                        "confidence": event.confidence,
                        "timestamp": event.timestamp.isoformat(),
                    },
                )

            # LLM Reasoning (only for high-confidence events with cooldown)
            reasoning_result = None
            if event.confidence >= settings.event.llm_trigger_threshold:
                # Check LLM cooldown
                cooldown_key = (camera_id, event.event_type.value)
                last_trigger = app_state._llm_cooldown.get(cooldown_key, 0)
                now = time.time()

                if now - last_trigger >= app_state._llm_cooldown_seconds:
                    app_state._llm_cooldown[cooldown_key] = now

                    similar = []
                    if event.embedding is not None:
                        similar = app_state.faiss_store.search_similar_events(
                            event.embedding, top_k=settings.faiss.top_k
                        )

                    try:
                        reasoning_result = app_state.reasoning_engine.analyze_event(
                            event=event,
                            similar_events=similar,
                            camera_meta={"camera_id": camera_id},
                        )
                    except Exception as e:
                        logger.warning(f"LLM reasoning failed: {e}")
                else:
                    logger.debug(
                        f"LLM cooldown active for {cooldown_key}, skipping reasoning"
                    )

            # Generate alert
            alert = app_state.alert_engine.process_event(event, reasoning_result)
            if alert:
                # Set the correct DB event ID (integer) instead of UUID
                alert.event_id = db_event.id
                alert_repo.create_from_schema(alert)
                logger.info(
                    f"🚨 ALERT [{alert.priority.value}] {alert.event_type.value} "
                    f"on camera {camera_id} — {alert.reasoning[:80]}"
                )

            # Audit log
            audit_repo.log_action(
                action="event_detected",
                entity_type="event",
                entity_id=str(db_event.id),
                details={
                    "event_type": event.event_type.value,
                    "confidence": event.confidence,
                    "camera_id": camera_id,
                },
            )

    except Exception as e:
        logger.error(f"Event handling error: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# FastAPI Server
# ---------------------------------------------------------------------------
def run_api_server() -> None:
    """Run FastAPI backend in a thread."""
    from backend.app import create_app

    api_app = create_app(app_state)
    config = uvicorn.Config(
        api_app,
        host=settings.backend.host,
        port=settings.backend.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


# ---------------------------------------------------------------------------
# Streamlit Launcher
# ---------------------------------------------------------------------------
def run_streamlit() -> None:
    """Launch Streamlit dashboard in a subprocess."""
    import subprocess

    frontend_path = Path(__file__).parent / "frontend" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(frontend_path),
        "--server.port", str(settings.frontend.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    logger.info(f"Starting Streamlit dashboard on port {settings.frontend.port}")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------
def shutdown(signum=None, frame=None) -> None:
    """Graceful shutdown."""
    logger.info("Shutting down...")
    app_state.running = False

    if app_state.event_executor:
        app_state.event_executor.shutdown(wait=False)

    if app_state.stream_manager:
        app_state.stream_manager.stop_all()

    if app_state.faiss_store:
        app_state.faiss_store.save_index()

    logger.info("Shutdown complete.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Application entrypoint."""
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Initialize all components
    init_components()

    # Start FastAPI in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True, name="api-server")
    api_thread.start()
    logger.info(
        f"FastAPI backend running on http://{settings.backend.host}:{settings.backend.port}"
    )

    # Start Streamlit dashboard
    run_streamlit()

    # Run main processing loop in the main thread
    try:
        processing_loop()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
