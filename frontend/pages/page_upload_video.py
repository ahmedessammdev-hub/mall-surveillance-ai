"""Upload Video page — upload video files to use as camera sources for testing."""

import os
import streamlit as st
from pathlib import Path


# Storage directory for uploaded videos
UPLOAD_DIR = Path(__file__).parent.parent.parent / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def render(api_get, api_post):
    st.markdown("""
    <div class="header-card">
        <h1>📤 Upload Video</h1>
        <p>Upload a video file to use as a camera source for testing the surveillance system</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    > Upload a video file (MP4, AVI, MOV) and register it as a camera source.
    > The system will process the video as if it were a live camera feed.
    """)

    # Upload form
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "wmv"],
            help="Upload a video file to use as a camera source",
        )

    with col2:
        camera_name = st.text_input("Camera Name", placeholder="Test Camera")
        location = st.text_input("Location", placeholder="Test Location")
        zone = st.text_input("Zone", placeholder="entrance")

    if uploaded_file is not None:
        st.divider()

        # Show file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f"""
        **File:** {uploaded_file.name}  
        **Size:** {file_size_mb:.1f} MB  
        **Type:** {uploaded_file.type}
        """)

        # Save button
        if st.button("💾 Save & Register as Camera", type="primary", use_container_width=True):
            # Save the file
            file_path = UPLOAD_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Video saved to: `{file_path}`")

            # Register as camera via API
            if not camera_name:
                cam_name = f"Video: {uploaded_file.name}"
            else:
                cam_name = camera_name

            result = api_post("/cameras/", json_data={
                "name": cam_name,
                "rtsp_url": str(file_path),
                "location": location or "Uploaded Video",
                "zone": zone or "test",
                "fps": 5,
                "resolution_w": 1280,
                "resolution_h": 720,
            })

            if result:
                st.success(f"Camera '{cam_name}' registered successfully!")
                st.info("Restart the system (`uv run python main.py`) to start processing this video.")
            else:
                st.warning("Video saved but failed to register camera via API. You can add it manually from Camera Management.")

    st.divider()

    # Show existing uploads
    st.markdown("#### Previously Uploaded Videos")
    if UPLOAD_DIR.exists():
        videos = list(UPLOAD_DIR.glob("*.*"))
        if videos:
            for v in videos:
                size_mb = v.stat().st_size / (1024 * 1024)
                st.markdown(f"- `{v.name}` ({size_mb:.1f} MB) — `{v}`")
        else:
            st.caption("No videos uploaded yet")
    else:
        st.caption("Upload directory not found")

    # Quick test with sample
    st.divider()
    st.markdown("#### Quick Test")
    st.markdown("""
    **Don't have a video?** You can use your webcam as a source:
    - Go to **📹 Camera Management**
    - Add a camera with source URL: `0`
    - This will use your default webcam
    """)
