"""Event Investigation page — deep-dive into specific events."""

import json
import streamlit as st
import plotly.graph_objects as go


def render(api_get):
    st.markdown("""
    <div class="header-card">
        <h1>🔍 Event Investigation</h1>
        <p>Deep-dive into security events with timeline, similar events, and LLM reasoning</p>
    </div>
    """, unsafe_allow_html=True)

    # Fetch recent events
    events = api_get("/events/recent", params={"limit": 50}) or []

    if not events:
        st.info("No events recorded yet. Events will appear as the system detects activity.")
        return

    # Event selector
    event_options = {
        f"[{e.get('event_type', '?')}] {e.get('timestamp', '')[:19]} (Cam {e.get('camera_id')}, Conf: {e.get('confidence', 0):.1%})": e
        for e in events
    }

    selected_label = st.selectbox("Select Event to Investigate", list(event_options.keys()))
    event = event_options.get(selected_label)

    if not event:
        return

    st.divider()

    # Event summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="font-size: 20px;">
                {event.get('event_type', 'N/A').replace('_', ' ').title()}
            </span>
            <span class="metric-label">Event Type</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        risk = event.get("risk_level", "low")
        risk_colors = {"critical": "#dc2626", "high": "#f97316", "medium": "#eab308", "low": "#22c55e"}
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="font-size: 20px; color: {risk_colors.get(risk, '#8b5cf6')};">
                {risk.upper()}
            </span>
            <span class="metric-label">Risk Level</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="font-size: 20px;">{event.get('confidence', 0):.1%}</span>
            <span class="metric-label">Confidence</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="font-size: 20px;">{event.get('person_count', 0)}</span>
            <span class="metric-label">Persons</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Two-column layout: Details + Reasoning
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Details", "🧠 Reasoning", "🔗 Similar Events", "📊 Motion Features"])

    with tab1:
        st.markdown("#### Event Details")
        st.json({
            "event_id": event.get("uid"),
            "camera_id": event.get("camera_id"),
            "timestamp": event.get("timestamp"),
            "event_type": event.get("event_type"),
            "confidence": event.get("confidence"),
            "risk_level": event.get("risk_level"),
            "person_count": event.get("person_count"),
            "crowd_density": event.get("crowd_density"),
            "location": event.get("location"),
        })

        st.markdown("#### Involved Tracks")
        try:
            tracks = json.loads(event.get("involved_tracks_json", "[]"))
            if tracks:
                for t in tracks:
                    st.markdown(
                        f"- **Track {t.get('track_id', '?')}**: "
                        f"speed={t.get('speed', 0):.1f} px/s, "
                        f"role={t.get('role', 'N/A')}"
                    )
            else:
                st.caption("No specific tracks recorded")
        except json.JSONDecodeError:
            st.caption("Track data unavailable")

    with tab2:
        st.markdown("#### LLM Reasoning Report")
        try:
            reasoning = json.loads(event.get("reasoning_json", "{}"))
            if reasoning:
                st.info(reasoning.get("reasoning", "No reasoning available"))
                st.markdown(f"**Recommended Action:** {reasoning.get('recommended_action', 'N/A')}")
                st.markdown(f"**Requires Intervention:** {'Yes ⚠️' if reasoning.get('requires_intervention') else 'No'}")
            else:
                st.caption("No LLM reasoning available for this event. "
                          "Reasoning is generated only for events above the confidence threshold.")
        except json.JSONDecodeError:
            st.caption("Reasoning data unavailable")

    with tab3:
        st.markdown("#### Similar Historical Events")
        try:
            similar = json.loads(event.get("similar_events_json", "[]"))
            if similar:
                for s in similar:
                    st.markdown(
                        f"- **Event {s.get('event_id', '?')}**: "
                        f"similarity={s.get('score', 0):.3f}, "
                        f"type={s.get('event_type', 'N/A')}"
                    )
            else:
                st.caption("No similar events found. The system builds historical context over time.")
        except json.JSONDecodeError:
            st.caption("Similar events data unavailable")

    with tab4:
        st.markdown("#### Motion & Behavior Features")
        try:
            motion = json.loads(event.get("motion_features_json", "{}"))
            behavior = json.loads(event.get("behavior_scores_json", "{}"))

            if motion:
                st.markdown("**Motion Features:**")
                st.json(motion)
            if behavior:
                st.markdown("**Behavior Scores:**")
                # Visualize as bar chart
                if behavior:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(behavior.keys()),
                            y=list(behavior.values()),
                            marker_color="#8b5cf6",
                        )
                    ])
                    fig.update_layout(
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except json.JSONDecodeError:
            st.caption("Feature data unavailable")

    # Event timeline
    st.divider()
    st.markdown("#### Event Timeline (Recent)")
    timeline_events = events[:20]
    if timeline_events:
        fig = go.Figure()
        for evt in timeline_events:
            color_map = {
                "fight": "#dc2626", "fall": "#f97316", "crowd_panic": "#eab308",
                "loitering": "#3b82f6", "suspicious_behavior": "#8b5cf6", "vandalism": "#ec4899",
            }
            fig.add_trace(go.Scatter(
                x=[evt.get("timestamp")],
                y=[evt.get("confidence", 0)],
                mode="markers",
                marker=dict(
                    size=max(10, evt.get("confidence", 0) * 30),
                    color=color_map.get(evt.get("event_type"), "#8b5cf6"),
                    opacity=0.8,
                ),
                name=evt.get("event_type", "unknown"),
                hovertext=f"Type: {evt.get('event_type')}<br>Conf: {evt.get('confidence', 0):.2%}<br>Risk: {evt.get('risk_level')}",
            ))

        fig.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Time",
            yaxis_title="Confidence",
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
