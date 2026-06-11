"""Live Cameras page — real-time video feeds with detection overlays."""

import streamlit as st


def render(api_get):
    st.markdown("""
    <div class="header-card">
        <h1>🎥 Live Cameras</h1>
        <p>Real-time video streams with detection overlays and tracking visualization</p>
    </div>
    """, unsafe_allow_html=True)

    # Fetch cameras
    cameras = api_get("/cameras/") or []

    if not cameras:
        st.info("No cameras configured. Go to **Camera Management** to add cameras.")
        return

    # Camera grid
    cols_per_row = st.selectbox("Grid Layout", [1, 2, 3, 4], index=1, key="cam_grid")
    cols = st.columns(cols_per_row)

    for idx, cam in enumerate(cameras):
        col = cols[idx % cols_per_row]
        with col:
            status_color = "#22c55e" if cam.get("status") == "online" else "#ef4444"
            status_icon = "●" if cam.get("status") == "online" else "○"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e1e4a, #252560);
                        border-radius: 12px; padding: 16px; margin-bottom: 16px;
                        border: 1px solid rgba(139,92,246,0.2);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #e0e0ff; font-weight: 600;">{cam.get('name', 'Camera')}</span>
                    <span style="color: {status_color}; font-size: 14px;">
                        {status_icon} {cam.get('status', 'offline').upper()}
                    </span>
                </div>
                <div style="background: #0f0f23; border-radius: 8px; height: 200px;
                            display: flex; align-items: center; justify-content: center;
                            color: #64748b; font-size: 14px;">
                    📹 {cam.get('resolution_w', 1280)}x{cam.get('resolution_h', 720)} @ {cam.get('fps', 5)} FPS
                </div>
                <div style="margin-top: 8px; display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8; font-size: 12px;">📍 {cam.get('location', 'N/A')}</span>
                    <span style="color: #94a3b8; font-size: 12px;">🏷️ {cam.get('zone', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Stream health
    st.subheader("Stream Health")
    health = api_get("/system-health/")
    if health:
        fps_data = health.get("processing_fps", {})
        if fps_data:
            cols = st.columns(len(fps_data))
            for i, (cam_id, fps) in enumerate(fps_data.items()):
                with cols[i]:
                    st.metric(f"Camera {cam_id}", f"{fps:.1f} FPS")
        else:
            st.caption("No active processing streams")
