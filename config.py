"""
Centralized configuration for Mall Surveillance AI system.
Uses pydantic-settings for environment variable support and validation.
"""

from pathlib import Path
from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
CLIPS_DIR = STORAGE_DIR / "clips"
SNAPSHOTS_DIR = STORAGE_DIR / "snapshots"
MODELS_DIR = STORAGE_DIR / "models"
FAISS_DIR = STORAGE_DIR / "faiss"

# Create storage directories on import
for _d in (STORAGE_DIR, CLIPS_DIR, SNAPSHOTS_DIR, MODELS_DIR, FAISS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertPriority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class EventType(str, Enum):
    FIGHT = "fight"
    FALL = "fall"
    CROWD_PANIC = "crowd_panic"
    LOITERING = "loitering"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    VANDALISM = "vandalism"


class CameraStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class DatabaseSettings(BaseSettings):
    """Database configuration. Swap URL to postgresql+asyncpg://... for Postgres."""
    url: str = Field(
        default=f"sqlite:///{BASE_DIR / 'surveillance.db'}",
        description="SQLAlchemy database URL",
    )
    echo: bool = Field(default=False, description="Echo SQL statements")

    model_config = {"env_prefix": "DB_"}


class DetectorSettings(BaseSettings):
    """YOLO11 person detector configuration."""
    model_name: str = Field(default="yolo11n.pt", description="YOLO model variant")
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = Field(default="cuda", description="cuda or cpu")
    img_size: int = Field(default=640, description="Input image size")
    person_class_id: int = Field(default=0, description="COCO person class ID")

    model_config = {"env_prefix": "DETECTOR_"}


class TrackerSettings(BaseSettings):
    """ByteTrack tracker configuration."""
    tracker_type: str = Field(default="bytetrack.yaml")
    track_high_thresh: float = Field(default=0.5)
    track_low_thresh: float = Field(default=0.1)
    new_track_thresh: float = Field(default=0.6)
    track_buffer: int = Field(default=30, description="Frames to keep lost tracks")
    match_thresh: float = Field(default=0.8)
    max_trajectory_length: int = Field(default=300, description="Max trajectory points to keep")

    model_config = {"env_prefix": "TRACKER_"}


class EmbedderSettings(BaseSettings):
    """VideoMAE embedder configuration."""
    model_name: str = Field(
        default="MCG-NJU/videomae-base",
        description="HuggingFace model ID",
    )
    device: str = Field(default="cuda")
    num_frames: int = Field(default=16, description="Frames per clip for embedding")
    embedding_dim: int = Field(default=768, description="Embedding vector dimension")
    batch_size: int = Field(default=4)
    use_fp16: bool = Field(default=True, description="Use mixed precision")

    model_config = {"env_prefix": "EMBEDDER_"}


class StreamSettings(BaseSettings):
    """Camera stream configuration."""
    target_fps: int = Field(default=5, description="Target processing FPS per camera")
    reconnect_interval: float = Field(default=5.0, description="Seconds between reconnect attempts")
    max_reconnect_attempts: int = Field(default=10)
    frame_buffer_size: int = Field(default=30)
    default_resolution: tuple[int, int] = Field(default=(1280, 720))

    model_config = {"env_prefix": "STREAM_"}


class EventSettings(BaseSettings):
    """Event detection thresholds."""
    # Fight detection
    fight_speed_threshold: float = Field(default=150.0, description="px/sec")
    fight_proximity_threshold: float = Field(default=100.0, description="pixels")
    fight_interaction_duration: float = Field(default=2.0, description="seconds")

    # Fall detection
    fall_vertical_displacement: float = Field(default=80.0, description="pixels")
    fall_speed_drop_ratio: float = Field(default=0.3)

    # Crowd panic
    panic_avg_speed_threshold: float = Field(default=200.0, description="px/sec")
    panic_dispersion_threshold: float = Field(default=0.7)
    panic_min_crowd_size: int = Field(default=5)

    # Loitering
    loitering_dwell_time: float = Field(default=300.0, description="seconds (5 min)")
    loitering_max_speed: float = Field(default=20.0, description="px/sec")

    # Suspicious behavior
    suspicious_direction_changes: int = Field(default=5, description="In 30 sec window")
    suspicious_zone_violations: int = Field(default=2)

    # Vandalism
    vandalism_acceleration_threshold: float = Field(default=500.0, description="px/sec²")

    # General
    event_confidence_threshold: float = Field(default=0.6, description="Min confidence for event")
    llm_trigger_threshold: float = Field(default=0.7, description="Min confidence to trigger LLM")
    event_dedup_window: float = Field(default=10.0, description="Seconds to dedup same event type")

    model_config = {"env_prefix": "EVENT_"}


class FAISSSettings(BaseSettings):
    """FAISS vector store configuration."""
    index_path: str = Field(
        default=str(FAISS_DIR / "events.index"),
        description="Path to FAISS index file",
    )
    metadata_path: str = Field(
        default=str(FAISS_DIR / "events_meta.json"),
        description="Path to metadata JSON file",
    )
    dimension: int = Field(default=768)
    top_k: int = Field(default=5, description="Default number of similar events to retrieve")

    model_config = {"env_prefix": "FAISS_"}


class LLMSettings(BaseSettings):
    """LLM reasoning engine configuration."""
    provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    model: str = Field(default="llama3.1:8b")
    base_url: str = Field(default="http://localhost:11434")
    api_key: str = Field(default="", description="For OpenAI-compatible APIs")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024)
    timeout: float = Field(default=30.0, description="Seconds")

    model_config = {"env_prefix": "LLM_"}


class BackendSettings(BaseSettings):
    """FastAPI backend configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    cors_origins: list[str] = Field(default=["*"])

    model_config = {"env_prefix": "BACKEND_"}


class FrontendSettings(BaseSettings):
    """Streamlit frontend configuration."""
    port: int = Field(default=8501)
    api_url: str = Field(default="http://localhost:8000")

    model_config = {"env_prefix": "FRONTEND_"}


# ---------------------------------------------------------------------------
# Top-level settings aggregate
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Top-level application settings."""
    app_name: str = "Mall Surveillance AI"
    debug: bool = Field(default=False)

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    event: EventSettings = Field(default_factory=EventSettings)
    faiss: FAISSSettings = Field(default_factory=FAISSSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    frontend: FrontendSettings = Field(default_factory=FrontendSettings)


# Singleton instance
settings = Settings()
