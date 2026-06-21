"""Live Cameras page — real-time video feeds with detection overlays and LLM reasoning."""

import streamlit as st


def render(api_get):
    st.markdown("""
    <div class="header-card">
        <h1>🎥 Live Cameras</h1>
        <p>Real-time video streams with detection overlays and AI reasoning</p>
    </div>
    """, unsafe_allow_html=True)

    # Fetch data
    cameras = api_get("/cameras/") or []
    health = api_get("/system-health/") or {}
    fps_data = health.get("processing_fps", {})
    events = api_get("/events/recent", params={"limit": 20}) or []

    if not cameras:
        st.info("No cameras configured. Go to **Camera Management** or **Upload Video** to add cameras.")
        return

    # Summary bar
    online_count = len([c for c in cameras if c.get("status") == "online"])
    total_fps = sum(fps_data.values()) if fps_data else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="color: #22c55e;">{online_count}</span>
            <span class="metric-label">Online Cameras</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{total_fps:.1f}</span>
            <span class="metric-label">Total FPS</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{len(events)}</span>
            <span class="metric-label">Recent Events</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Refresh button (manual only, no auto-refresh)
    if st.button("🔄 Refresh Feeds", use_container_width=True):
        st.rerun()

    # Camera feed
    for cam in cameras:
        cam_id = cam.get("id")
        cam_fps = fps_data.get(str(cam_id), 0)
        status_color = "#22c55e" if cam.get("status") == "online" else "#ef4444"

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="color: #e0e0ff; font-weight: 600; font-size: 16px;">
                📹 {cam.get('name', 'Camera')}
            </span>
            <span style="color: {status_color}; font-size: 13px;">
                {cam.get('status', 'offline').upper()} | {cam_fps:.1f} FPS | 📍 {cam.get('location', 'N/A')}
            </span>
        </div>
        """, unsafe_allow_html=True)

        col_video, col_events = st.columns([2, 1])

        with col_video:
            # Show actual video frame
            try:
                import httpx
                resp = httpx.get(f"http://localhost:8000/api/cameras/{cam_id}/frame", timeout=3.0)
                if resp.status_code == 200:
                    st.image(resp.content, channels="BGR", use_container_width=True)
                else:
                    st.markdown(f"""
                    <div style="background: #0f0f23; border-radius: 8px; height: 300px;
                                display: flex; align-items: center; justify-content: center;
                                color: #64748b; font-size: 14px; flex-direction: column; gap: 8px;">
                        <span style="font-size: 32px;">📹</span>
                        <span>Waiting for frames...</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                st.markdown(f"""
                <div style="background: #0f0f23; border-radius: 8px; height: 300px;
                            display: flex; align-items: center; justify-content: center;
                            color: #64748b; font-size: 14px; flex-direction: column; gap: 8px;">
                    <span style="font-size: 32px;">📹</span>
                    <span>Stream not available</span>
                </div>
                """, unsafe_allow_html=True)

        with col_events:
            st.markdown("##### 🧠 AI Events & Reasoning")
            cam_events = [e for e in events if str(e.get("camera_id")) == str(cam_id)]

            if cam_events:
                for evt in cam_events[:3]:
                    event_type = evt.get("event_type", "unknown").replace("_", " ").title()
                    confidence = evt.get("confidence", 0)
                    risk = evt.get("risk_level", "low")
                    icon = {
                        "fight": "⚔️", "fall": "🤕", "crowd_panic": "🏃",
                        "loitering": "🚶", "suspicious_behavior": "🕵️", "vandalism": "💥",
                    }.get(evt.get("event_type", ""), "⚠️")

                    risk_badge = {
                        "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢",
                    }.get(risk, "⚪")

                    # Get reasoning from the event
                    import json
                    reasoning_text = ""
                    try:
                        reasoning_data = json.loads(evt.get("reasoning_json", "{}"))
                        reasoning_text = reasoning_data.get("reasoning", "")
                    except (json.JSONDecodeError, TypeError):
                        pass

                    with st.expander(f"{icon} {event_type} {risk_badge} ({confidence:.0%})", expanded=False):
                        st.markdown(f"**Risk:** {risk.upper()} | **Confidence:** {confidence:.1%}")
                        if reasoning_text:
                            st.info(f"🧠 **LLM Analysis:** {reasoning_text}")
                        else:
                            st.caption("No LLM reasoning (confidence below threshold)")
            else:
                st.caption("No events detected yet")

        st.divider()
