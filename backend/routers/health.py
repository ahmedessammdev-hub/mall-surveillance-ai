"""System health monitoring API routes."""

import time

import psutil
from fastapi import APIRouter, Request

from backend.schemas import SystemHealthResponse

router = APIRouter()

_start_time = time.time()


@router.get("/", response_model=SystemHealthResponse)
def system_health(request: Request):
    """Get system health metrics: CPU, RAM, GPU, streams, etc."""
    app_state = request.app.state.app_state

    # CPU & RAM
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    ram_used_gb = round(ram.used / (1024 ** 3), 2)
    ram_total_gb = round(ram.total / (1024 ** 3), 2)

    # GPU
    gpu_name = ""
    gpu_mem_used = 0.0
    gpu_mem_total = 0.0
    gpu_util = 0.0
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_name = gpu.name
            gpu_mem_used = round(gpu.memoryUsed, 1)
            gpu_mem_total = round(gpu.memoryTotal, 1)
            gpu_util = round(gpu.load * 100, 1)
    except Exception:
        pass

    # Stream info
    active_streams = 0
    processing_fps = {}
    if app_state:
        if app_state.stream_manager:
            active_streams = app_state.stream_manager.camera_count
        processing_fps = {k: round(v, 1) for k, v in app_state.processing_fps.items()}

    # FAISS info
    faiss_vectors = 0
    if app_state and app_state.faiss_store:
        faiss_vectors = app_state.faiss_store.get_embedding_count()

    return SystemHealthResponse(
        cpu_percent=cpu_percent,
        ram_used_gb=ram_used_gb,
        ram_total_gb=ram_total_gb,
        ram_percent=round(ram.percent, 1),
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_mem_used,
        gpu_memory_total_mb=gpu_mem_total,
        gpu_utilization=gpu_util,
        active_streams=active_streams,
        processing_fps=processing_fps,
        faiss_vectors=faiss_vectors,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
