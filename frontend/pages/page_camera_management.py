"""Camera Management page — add, edit, remove cameras."""

import streamlit as st
import httpx


def render(api_get, api_post):
    st.markdown("""
    <div class="header-card">
        <h1>📹 Camera Management</h1>
        <p>Add, configure, and monitor surveillance cameras</p>
    </div>
    """, unsafe_allow_html=True)

    # Add camera form
    with st.expander("➕ Add New Camera", expanded=False):
        with st.form("add_camera_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Camera Name", placeholder="Main Entrance Camera")
                rtsp_url = st.text_input(
                    "Source URL",
                    placeholder="rtsp://... or /path/to/video.mp4 or 0 (webcam)",
                )
                location = st.text_input("Location", placeholder="Floor 1, Main Entrance")
            with col2:
                zone = st.text_input("Zone", placeholder="entrance")
                fps = st.number_input("Target FPS", min_value=1, max_value=30, value=5)
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    res_w = st.number_input("Width", value=1280)
                with res_col2:
                    res_h = st.number_input("Height", value=720)

            submitted = st.form_submit_button("Add Camera", type="primary")
            if submitted and name:
                try:
                    resp = httpx.post(
                        "http://localhost:8000/api/cameras/",
                        json={
                            "name": name,
                            "rtsp_url": rtsp_url,
                            "location": location,
                            "zone": zone,
                            "fps": fps,
                            "resolution_w": res_w,
                            "resolution_h": res_h,
                        },
                        timeout=10.0,
                    )
                    if resp.status_code == 201:
                        st.success(f"Camera '{name}' added successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to add camera: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Camera list
    cameras = api_get("/cameras/") or []

    if not cameras:
        st.info("No cameras configured. Use the form above to add your first camera.")
        return

    st.markdown(f"#### Configured Cameras ({len(cameras)})")

    for cam in cameras:
        status = cam.get("status", "offline")
        status_color = "#22c55e" if status == "online" else "#ef4444"
        status_icon = "●" if status == "online" else "○"

        with st.expander(
            f"{'🟢' if status == 'online' else '🔴'} "
            f"{cam.get('name', 'Unknown')} — {status.upper()}",
            expanded=False,
        ):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"""
                **ID:** {cam.get('id')}  
                **Source:** `{cam.get('rtsp_url', 'N/A')}`  
                **Location:** {cam.get('location', 'N/A')}  
                **Zone:** {cam.get('zone', 'N/A')}
                """)

            with col2:
                st.markdown(f"""
                **Resolution:** {cam.get('resolution_w', 0)}x{cam.get('resolution_h', 0)}  
                **Target FPS:** {cam.get('fps', 0)}  
                **Status:** <span style="color: {status_color};">{status_icon} {status.upper()}</span>  
                **Created:** {cam.get('created_at', 'N/A')[:19]}
                """, unsafe_allow_html=True)

            with col3:
                if st.button("🗑️ Delete", key=f"del_{cam.get('id')}"):
                    try:
                        resp = httpx.delete(
                            f"http://localhost:8000/api/cameras/{cam.get('id')}",
                            timeout=5.0,
                        )
                        if resp.status_code == 204:
                            st.success("Camera deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete camera")
                    except Exception as e:
                        st.error(f"Error: {e}")
