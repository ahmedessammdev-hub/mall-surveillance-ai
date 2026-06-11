"""System Health page — CPU, GPU, RAM, streams, latency monitoring."""

import streamlit as st
import plotly.graph_objects as go


def render(api_get):
    st.markdown("""
    <div class="header-card">
        <h1>💻 System Health</h1>
        <p>Real-time system performance monitoring — CPU, GPU, RAM, and processing metrics</p>
    </div>
    """, unsafe_allow_html=True)

    health = api_get("/system-health/")
    if not health:
        st.error("Unable to reach system health API. Is the backend running?")
        return

    # Top gauges: CPU, RAM, GPU
    c1, c2, c3 = st.columns(3)

    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health.get("cpu_percent", 0),
            title={"text": "CPU Usage", "font": {"color": "#e0e0ff", "size": 16}},
            number={"suffix": "%", "font": {"color": "#e0e0ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": "#8b5cf6"},
                "bgcolor": "#1e1e4a",
                "steps": [
                    {"range": [0, 60], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [60, 85], "color": "rgba(234, 179, 8, 0.2)"},
                    {"range": [85, 100], "color": "rgba(220, 38, 38, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 2},
                    "thickness": 0.8,
                    "value": 90,
                },
            },
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health.get("ram_percent", 0),
            title={"text": f"RAM ({health.get('ram_used_gb', 0)}/{health.get('ram_total_gb', 0)} GB)", "font": {"color": "#e0e0ff", "size": 16}},
            number={"suffix": "%", "font": {"color": "#e0e0ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": "#06b6d4"},
                "bgcolor": "#1e1e4a",
                "steps": [
                    {"range": [0, 60], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [60, 85], "color": "rgba(234, 179, 8, 0.2)"},
                    {"range": [85, 100], "color": "rgba(220, 38, 38, 0.2)"},
                ],
            },
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        gpu_util = health.get("gpu_utilization", 0)
        gpu_name = health.get("gpu_name", "No GPU")
        gpu_mem_used = health.get("gpu_memory_used_mb", 0)
        gpu_mem_total = health.get("gpu_memory_total_mb", 0)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gpu_util,
            title={"text": f"GPU ({gpu_name})", "font": {"color": "#e0e0ff", "size": 14}},
            number={"suffix": "%", "font": {"color": "#e0e0ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": "#22c55e"},
                "bgcolor": "#1e1e4a",
                "steps": [
                    {"range": [0, 60], "color": "rgba(34, 197, 94, 0.2)"},
                    {"range": [60, 85], "color": "rgba(234, 179, 8, 0.2)"},
                    {"range": [85, 100], "color": "rgba(220, 38, 38, 0.2)"},
                ],
            },
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Detailed metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="font-size: 24px;">{health.get('active_streams', 0)}</span>
            <span class="metric-label">Active Streams</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="font-size: 24px;">{health.get('faiss_vectors', 0)}</span>
            <span class="metric-label">FAISS Vectors</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        gpu_mem_pct = (gpu_mem_used / gpu_mem_total * 100) if gpu_mem_total > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="font-size: 24px;">{gpu_mem_used:.0f}/{gpu_mem_total:.0f}</span>
            <span class="metric-label">GPU VRAM (MB)</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        uptime = health.get("uptime_seconds", 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="font-size: 24px;">{hours}h {minutes}m</span>
            <span class="metric-label">Uptime</span>
        </div>
        """, unsafe_allow_html=True)

    # Processing FPS per camera
    st.divider()
    st.markdown("#### Processing FPS per Camera")
    fps_data = health.get("processing_fps", {})
    if fps_data:
        fig = go.Figure(data=[go.Bar(
            x=[f"Camera {k}" for k in fps_data.keys()],
            y=list(fps_data.values()),
            marker_color="#8b5cf6",
            text=[f"{v:.1f}" for v in fps_data.values()],
            textposition="auto",
        )])
        fig.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Camera",
            yaxis_title="FPS",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No active processing streams")

    # Refresh
    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False, key="health_refresh")
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()
