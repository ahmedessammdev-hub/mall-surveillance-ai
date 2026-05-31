<div align="center">
  <h1>🛡️ Mall Surveillance AI</h1>
  <p>An intelligent, AI-powered security surveillance system that understands, analyzes, and explains security events in real-time.</p>
</div>

---

## 🌟 Overview
This project is not just a traditional violence detection system; it is a **comprehensive Event Understanding platform**. The system leverages Computer Vision, motion analysis, feature extraction, vector embeddings, and Large Language Models (LLMs) to detect complex security events (such as fights, falls, crowd panic, loitering, vandalism, and suspicious behavior), assess their risk, and provide actionable intelligence to security teams.

The architecture is designed as a **Modular Monolith**, optimized for local execution (Local-first) to ensure privacy, and tailored to run efficiently on single-GPU hardware (e.g., RTX 4060 / 4070 / 4080).

---

## 🏗️ Tech Stack
* **Environment & Package Management:** `uv` (fast and reliable)
* **Backend:** Python 3.12, FastAPI, SQLAlchemy, SQLite (easily migratable to PostgreSQL)
* **Frontend:** Streamlit (Professional multi-page dark theme dashboard)
* **Computer Vision:** PyTorch, Ultralytics YOLO11, OpenCV
* **Tracking:** ByteTrack
* **Video Embeddings:** VideoMAE (via HuggingFace Transformers)
* **Semantic Search (Vector Store):** FAISS
* **Reasoning Engine (LLM):** Ollama (for local offline LLMs) or OpenAI API

---

## 🔄 The Pipeline

1. **Stream Manager:** Ingests video from RTSP streams, local files, or webcams. It uses thread-per-camera architecture to ensure non-blocking stream reading.
2. **Detection & Tracking:**
   - **YOLO11** is used for high-accuracy person detection.
   - **ByteTrack** algorithms track individuals across frames, managing track lifecycles and trajectories.
3. **Feature Extraction:** Analyzes person speed, direction changes, interpersonal distance, and dwell time to build a comprehensive understanding of the scene and behaviors.
4. **Video Embeddings:** A **VideoMAE** model converts short video clips into semantic vectors to capture the context of the scene.
5. **Event Engine:** Applies smart rule-based heuristics to the extracted features to detect 6 types of events:
   - ⚔️ Fight
   - 🤕 Fall / Medical Emergency
   - 🏃 Crowd Panic
   - 🚶 Loitering
   - 🕵️ Suspicious Behavior
   - 💥 Vandalism
6. **FAISS Vector Store:** Saves event embeddings to allow semantic retrieval of "similar historical events" to aid the system's reasoning.
7. **LLM Reasoning:** Sends the detected event, motion features, and similar historical events to an LLM (via Ollama). The LLM confirms the event, assesses the risk level, and writes a security report with recommended actions.
8. **Alert Engine:** Converts confirmed events into prioritized alerts (P1, P2, P3) for the security dashboard.

---

## 📂 Project Structure

```text
yousef/
├── ai/                     # AI and Computer Vision models
│   ├── detector.py         # Person detection (YOLO11)
│   ├── tracker.py          # Person tracking (ByteTrack)
│   ├── feature_extractor.py# Motion and behavior feature extraction
│   └── embedder.py         # Video embeddings (VideoMAE)
├── alerts/                 # Alert management and prioritization
│   └── engine.py           # Maps events to P1/P2/P3 alerts
├── backend/                # FastAPI Backend
│   ├── app.py              # Main application factory
│   ├── schemas.py          # Pydantic models for API validation
│   └── routers/            # API endpoints (Cameras, Events, Alerts, etc.)
├── database/               # Database Layer (SQLAlchemy)
│   ├── connection.py       # SQLite connection setup
│   ├── models.py           # ORM Models (Camera, Track, Event, Alert, etc.)
│   └── repositories/       # Repository Pattern for data access
├── event_engine/           # Event detection heuristics
│   ├── constructor.py      # Event construction and deduplication
│   ├── rules.py            # Event detection rules
│   └── schemas.py          # Internal Pydantic event schemas
├── frontend/               # Streamlit Dashboard
│   ├── app.py              # Dashboard entrypoint and routing
│   └── pages/              # Dashboard pages (Live, Alerts, Investigation, etc.)
├── reasoning/              # LLM Reasoning Engine
│   ├── engine.py           # Context assembly and LLM orchestration
│   ├── llm_client.py       # Ollama / OpenAI API client
│   └── prompts.py          # System and analysis prompts
├── stream/                 # Video Stream Management
│   └── manager.py          # RTSP/File stream ingestion and health tracking
├── vector_db/              # Vector Database
│   └── faiss_store.py      # FAISS index management for semantic search
├── tests/                  # Pytest test suite
├── storage/                # Local storage for DB and FAISS indexes
├── config.py               # Centralized configuration (Pydantic Settings)
├── main.py                 # Application entrypoint to run the entire system
├── pyproject.toml          # uv project dependencies
└── README.md               # Project documentation (this file)
```

---

## 🚀 How to Run

### 1. Prerequisites
1. **Python 3.12**
2. **`uv`** package manager. To install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. For local AI reasoning, install **Ollama** and pull a small, fast model (e.g., `ollama run llama3.1` or `ollama run qwen2.5`).

### 2. Install Dependencies
Open your terminal in the project directory and run:
```bash
uv sync
```
*`uv` will automatically download and install all required packages (PyTorch, FastAPI, Streamlit, etc.) in a virtual environment.*

### 3. Configuration
The system uses `config.py` for default settings. You can override any setting using Environment Variables:
```bash
export LLM_PROVIDER="ollama"
export LLM_MODEL="llama3.1"
# If you prefer to use OpenAI:
# export LLM_PROVIDER="openai"
# export OPENAI_API_KEY="sk-..."
```

### 4. Run the System
To start the entire system (FastAPI backend, AI pipelines, and the Streamlit dashboard) with a single command:
```bash
uv run python main.py
```
*Note: YOLO and VideoMAE models will be downloaded automatically upon the first run.*

### 5. Access the System
Once the system is running, you can access:
* **Dashboard:** `http://localhost:8501` (Monitor cameras, alerts, and investigate events).
* **API Documentation (Swagger UI):** `http://localhost:8000/docs`.

---

## 🖥️ The Dashboard
The user interface features several professional pages:
1. 🎥 **Live Cameras:** Real-time video feeds with detection overlays and FPS tracking.
2. 🚨 **Active Alerts:** A live feed of security alerts segmented by priority, allowing operators to acknowledge incidents.
3. 🔍 **Event Investigation:** Deep-dive into specific events, viewing LLM reasoning, involved tracks, and similar historical events.
4. 🔎 **Event Search:** Search past events using dates, cameras, risk levels, and semantic similarity.
5. 📊 **Analytics:** Visual charts displaying operational insights, event distribution, and alert trends.
6. 📹 **Camera Management:** Add, configure, and remove camera streams.
7. 💻 **System Health:** Monitor server performance metrics (CPU, RAM, GPU VRAM, and processing latencies).

---

## 🧪 Testing
To run the automated test suite and ensure code integrity:
```bash
uv run pytest
```

---

<div align="center">
  <p>Built with ❤️ and AI by Ahmed Essam</p>
</div>
