<div align="center">
  <h1>Mall Surveillance AI</h1>
  <p>AI-powered intelligent security surveillance system with real-time event detection, LLM reasoning, and smart alerting.</p>
</div>

---

## Overview

A comprehensive **Event Understanding Platform** that combines Computer Vision, behavioral feature extraction, vector embeddings, and Large Language Models to detect, analyze, and explain security events in real-time.

**Detected Event Types:**
- Fight (P1 - Critical)
- Fall / Medical Emergency (P1 - Critical)
- Crowd Panic (P2 - High)
- Vandalism (P2 - High)
- Loitering (P3 - Medium)
- Suspicious Behavior (P3 - Medium)

---

## Architecture

```
Video Stream -> YOLO11 Detection -> ByteTrack Tracking -> Feature Extraction
                                                             |
                                                             v
                    LLM Reasoning <- FAISS Similar Events <- Event Engine
                         |
                         v
                    Alert Engine -> Dashboard (Streamlit) + REST API (FastAPI)
```

**Design:** Modular Monolith, Local-first, Privacy-preserving, Single-GPU/CPU capable.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Package Manager | `uv` |
| Backend | Python 3.12, FastAPI, SQLAlchemy, SQLite |
| Frontend | Streamlit (dark theme multi-page dashboard) |
| Detection | YOLO11 (Ultralytics) |
| Tracking | ByteTrack |
| Feature Extraction | Custom (KDTree, EMA smoothing, zone detection, posture estimation) |
| Video Embeddings | VideoMAE (HuggingFace Transformers) |
| Vector Store | FAISS |
| Reasoning | Ollama (qwen2.5:7b) or OpenAI API |
| System Monitor | psutil, GPUtil |

---

## Feature Extraction (v2)

The feature extractor computes **30+ features** per tracked person:

### Per-Person Features
| Category | Features |
|----------|----------|
| Motion | speed, acceleration, direction, speed_variance, speed_trend, direction_trend |
| Path | displacement, path_length, straightness, direction_changes |
| Dwell | dwell_time (continuous tracking with anchor point) |
| Fall Detection | vertical_displacement, bbox_aspect_ratio, posture_state (standing/sitting/fallen) |
| Proximity | nearest_person_distance, people_within_radius, interaction_duration, proximity_score |
| Zone | current_zone, is_in_restricted_zone |

### Scene-Level Features
person_count, crowd_density, avg_speed, max_speed, motion_energy, motion_dispersion, interaction_count, fallen_count, restricted_zone_violations

### Performance
- **O(n log n)** proximity computation via `scipy.spatial.cKDTree`
- **EMA temporal smoothing** (configurable alpha) for noise reduction
- **Resolution normalization** (thresholds scaled to 720p base)
- **Incremental path length** accumulation (O(1) per frame)

---

## Pipeline

1. **Stream Manager** — Thread-per-camera RTSP/file/webcam ingestion
2. **YOLO11 Detection** — Person detection (nano variant, CUDA/CPU)
3. **ByteTrack Tracking** — Persistent track IDs with trajectory history
4. **Feature Extraction** — 30+ motion/behavior/interaction features
5. **VideoMAE Embedding** — 768-dim semantic video vectors (every N frames)
6. **Event Engine** — 6 rule-based detectors with confidence scoring
7. **FAISS Store** — Similar historical event retrieval
8. **LLM Reasoning** — Event confirmation, risk assessment, security reports
9. **Alert Engine** — P1/P2/P3 prioritized alerts

### Pipeline Optimizations
- **Async event handling** — ThreadPoolExecutor(4), non-blocking frame loop
- **Parallel camera processing** — Multi-camera concurrent processing
- **LLM cooldown** — 30s per (camera, event_type) to prevent spam
- **Clip buffer safety** — Max 64 frames with automatic trimming

---

## Project Structure

```
mall-surveillance-ai/
├── ai/                         # AI pipeline
│   ├── detector.py             # YOLO11 person detection
│   ├── tracker.py              # ByteTrack multi-person tracking
│   ├── feature_extractor.py    # Motion/behavior/interaction features (KDTree, EMA, zones)
│   └── embedder.py             # VideoMAE video embeddings
├── alerts/
│   └── engine.py               # P1/P2/P3 alert prioritization
├── backend/
│   ├── app.py                  # FastAPI application factory
│   ├── schemas.py              # API request/response models
│   └── routers/                # API endpoints (cameras, events, alerts, search, analytics, health)
├── database/
│   ├── connection.py           # SQLAlchemy engine + session
│   ├── models.py               # ORM models (Camera, Track, Event, Alert, Embedding, AuditLog)
│   └── repositories/           # Repository pattern CRUD
├── event_engine/
│   ├── constructor.py          # Event construction + deduplication
│   ├── rules.py                # 6 rule-based event detectors
│   └── schemas.py              # Pydantic event schemas
├── frontend/
│   ├── app.py                  # Streamlit dashboard entrypoint
│   └── pages/                  # 7 dashboard pages
├── reasoning/
│   ├── engine.py               # LLM orchestration
│   ├── llm_client.py           # Ollama/OpenAI client
│   └── prompts.py              # System/analysis prompts
├── stream/
│   └── manager.py              # Multi-camera stream ingestion
├── vector_db/
│   └── faiss_store.py          # FAISS index management
├── tests/                      # Pytest test suite (52 tests)
├── config.py                   # Pydantic Settings (env var overrides)
├── main.py                     # Application entrypoint
└── pyproject.toml              # Dependencies
```

---

## How to Run

### Prerequisites
1. **Python 3.12**
2. **Ollama** — for local LLM reasoning

### 1. Install Ollama and pull the model
```bash
# Install Ollama
winget install Ollama.Ollama        # Windows
# or: curl -fsSL https://ollama.com/install.sh | sh  # Linux/Mac

# Pull the recommended model
ollama pull qwen2.5:7b
```

### 2. Install dependencies
```bash
# Install uv (if not installed)
pip install uv

# Install project dependencies
uv sync
```

### 3. Run the system
```bash
uv run python main.py
```

This single command starts:
- **AI Pipeline** — Detection, tracking, feature extraction, event detection
- **FastAPI Backend** — `http://localhost:8000`
- **Streamlit Dashboard** — `http://localhost:8501`

*First run will download YOLO11 and VideoMAE models automatically.*

### 4. Access the system
| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Health | http://localhost:8000/api/system-health |

---

## Configuration

All settings are in `config.py` and can be overridden via environment variables:

```bash
# LLM
export LLM_PROVIDER="ollama"          # ollama or openai
export LLM_MODEL="qwen2.5:7b"
export LLM_BASE_URL="http://localhost:11434"

# Detection
export DETECTOR_CONFIDENCE_THRESHOLD="0.45"
export DETECTOR_DEVICE="cuda"          # cuda or cpu

# Feature Extraction
export FEATURE_EMA_ALPHA="0.3"
export FEATURE_RESTRICTED_ZONES='["entrance","storage"]'

# Events
export EVENT_LLM_TRIGGER_THRESHOLD="0.7"
export EVENT_DEDUP_WINDOW="10.0"
```

---

## Dashboard Pages

1. **Live Cameras** — Real-time camera grid with detection overlays and FPS tracking
2. **Active Alerts** — Live alert feed with P1/P2/P3 priority filtering and acknowledgement
3. **Event Investigation** — Deep-dive into events with LLM reasoning, similar events, and motion analysis
4. **Event Search** — Filter-based and semantic search across historical events
5. **Analytics** — Charts for event distribution, alert trends, and operational insights
6. **Camera Management** — Add/configure/remove camera streams
7. **System Health** — CPU, RAM, GPU, FPS monitoring with gauge charts

---

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_feature_extractor.py -v

# Run with coverage
uv run pytest --cov=. --cov-report=html
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cameras` | List all cameras |
| POST | `/api/cameras` | Add a camera |
| GET | `/api/events` | List events (filterable) |
| GET | `/api/events/{id}` | Event detail |
| GET | `/api/alerts` | List alerts |
| POST | `/api/alerts/{id}/acknowledge` | Acknowledge alert |
| GET | `/api/tracks` | Active tracks |
| POST | `/api/search` | Semantic event search |
| GET | `/api/analytics` | Aggregated statistics |
| GET | `/api/system-health` | CPU/RAM/GPU/FPS |

---

<div align="center">
  <p>Built with AI by Ahmed Essam</p>
</div>
