"""Active Alerts page — real-time alert feed with filtering and acknowledgement."""

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def render(api_get, api_post):
    st.markdown("""
    <div class="header-card">
        <h1>🚨 Active Alerts</h1>
        <p>Real-time security alert feed with priority filtering and acknowledgement</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        priority_filter = st.selectbox("Priority", ["All", "P1", "P2", "P3"], key="alert_prio")
    with col2:
        ack_filter = st.selectbox("Status", ["Unacknowledged", "Acknowledged", "All"], key="alert_ack")
    with col3:
        type_filter = st.selectbox(
            "Event Type",
            ["All", "fight", "fall", "crowd_panic", "loitering", "suspicious_behavior", "vandalism"],
            key="alert_type",
        )

    # Build params
    params = {"limit": 50}
    if priority_filter != "All":
        params["priority"] = priority_filter
    if ack_filter == "Unacknowledged":
        params["acknowledged"] = False
    elif ack_filter == "Acknowledged":
        params["acknowledged"] = True
    if type_filter != "All":
        params["event_type"] = type_filter

    # Fetch alerts
    alerts = api_get("/alerts/", params=params) or []

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    p1_count = len([a for a in alerts if a.get("priority") == "P1"])
    p2_count = len([a for a in alerts if a.get("priority") == "P2"])
    p3_count = len([a for a in alerts if a.get("priority") == "P3"])

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{len(alerts)}</span>
            <span class="metric-label">Total Alerts</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="color: #dc2626;">{p1_count}</span>
            <span class="metric-label">P1 Critical</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="color: #f59e0b;">{p2_count}</span>
            <span class="metric-label">P2 High</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="color: #3b82f6;">{p3_count}</span>
            <span class="metric-label">P3 Medium</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if not alerts:
        st.success("No active alerts. All clear!")
        return

    # Alert feed
    for alert in alerts:
        priority = alert.get("priority", "P3")
        badge_class = f"alert-{priority.lower()}"
        risk = alert.get("risk_level", "low")
        risk_class = f"risk-{risk}"
        ack_status = "Acknowledged" if alert.get("acknowledged") else "Pending"
        event_type = alert.get("event_type", "Unknown").replace("_", " ").title()

        event_icons = {
            "Fight": "⚔️", "Fall": "🤕", "Crowd Panic": "🏃",
            "Loitering": "🚶", "Suspicious Behavior": "🕵️", "Vandalism": "💥",
        }
        icon = event_icons.get(event_type, "⚠️")

        with st.expander(
            f"{'🔴' if priority == 'P1' else '🟠' if priority == 'P2' else '🔵'} "
            f"[{priority}] {icon} {event_type} — "
            f"{alert.get('timestamp', '')[:19]}",
            expanded=(priority == "P1" and not alert.get("acknowledged")),
        ):
            col_a, col_b = st.columns([3, 1])

            with col_a:
                st.markdown(f"""
                **Reasoning:** {alert.get('reasoning', 'N/A')}

                **Recommended Action:** {alert.get('recommended_action', 'N/A')}
                """, unsafe_allow_html=True)

                # Show key metrics in a row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Confidence", f"{alert.get('confidence', 0):.1%}")
                with m2:
                    risk_colors = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                    st.metric("Risk", f"{risk_colors.get(risk, '⚪')} {risk.upper()}")
                with m3:
                    st.metric("Status", ack_status)

            with col_b:
                if not alert.get("acknowledged"):
                    if st.button("Acknowledge", key=f"ack_{alert.get('id')}"):
                        result = api_post(
                            f"/alerts/{alert.get('id')}/acknowledge",
                            json_data={"user": "operator"},
                        )
                        if result:
                            st.success("Alert acknowledged!")
                            st.rerun()
                else:
                    st.caption(f"By: {alert.get('acknowledged_by', 'N/A')}")
                    st.caption(f"At: {alert.get('acknowledged_at', 'N/A')[:19] if alert.get('acknowledged_at') else 'N/A'}")

    # Auto-refresh
    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False, key="alert_refresh")
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()
