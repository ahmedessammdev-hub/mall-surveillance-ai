"""Event Search page — search events by filters and semantic similarity."""

import streamlit as st
import pandas as pd


def render(api_get):
    st.markdown("""
    <div class="header-card">
        <h1>🔎 Event Search</h1>
        <p>Search security events by camera, date, type, and risk level</p>
    </div>
    """, unsafe_allow_html=True)

    # Search filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cameras = api_get("/cameras/") or []
        cam_options = ["All"] + [f"{c.get('id')} - {c.get('name')}" for c in cameras]
        cam_sel = st.selectbox("Camera", cam_options, key="search_cam")

    with col2:
        event_types = ["All", "fight", "fall", "crowd_panic", "loitering", "suspicious_behavior", "vandalism"]
        type_sel = st.selectbox("Event Type", event_types, key="search_type")

    with col3:
        risk_levels = ["All", "critical", "high", "medium", "low"]
        risk_sel = st.selectbox("Risk Level", risk_levels, key="search_risk")

    with col4:
        limit = st.number_input("Max Results", min_value=10, max_value=200, value=50, key="search_limit")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=None, key="search_start")
    with col_d2:
        end_date = st.date_input("End Date", value=None, key="search_end")

    # Build query
    params = {"limit": limit, "offset": 0}
    if cam_sel != "All":
        try:
            params["camera_id"] = int(cam_sel.split(" - ")[0])
        except (ValueError, IndexError):
            pass
    if type_sel != "All":
        params["event_type"] = type_sel
    if risk_sel != "All":
        params["risk_level"] = risk_sel
    if start_date:
        params["start_date"] = f"{start_date}T00:00:00"
    if end_date:
        params["end_date"] = f"{end_date}T23:59:59"

    st.divider()

    # Search button
    if st.button("🔍 Search", type="primary", key="search_btn"):
        events = api_get("/events/", params=params) or []

        if not events:
            st.warning("No events found matching the criteria.")
            return

        st.success(f"Found {len(events)} events")

        # Results table
        df = pd.DataFrame([
            {
                "ID": e.get("id"),
                "Type": e.get("event_type", "").replace("_", " ").title(),
                "Camera": e.get("camera_id"),
                "Confidence": f"{e.get('confidence', 0):.1%}",
                "Risk": e.get("risk_level", "").upper(),
                "Persons": e.get("person_count", 0),
                "Timestamp": e.get("timestamp", "")[:19],
            }
            for e in events
        ])

        # Color-code risk column
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Type": st.column_config.TextColumn("Event Type", width="medium"),
                "Camera": st.column_config.NumberColumn("Cam", width="small"),
                "Confidence": st.column_config.TextColumn("Conf", width="small"),
                "Risk": st.column_config.TextColumn("Risk", width="small"),
                "Persons": st.column_config.NumberColumn("👥", width="small"),
                "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            },
        )
    else:
        st.caption("Configure filters and click Search to find events.")
