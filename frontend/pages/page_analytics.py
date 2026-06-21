"""Analytics page — event counts, alert statistics, trends."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def render(api_get):
    st.markdown("""
    <div class="header-card">
        <h1>📊 Analytics</h1>
        <p>Event statistics, alert trends, and operational insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Fetch analytics
    data = api_get("/analytics/") or {}

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{data.get('total_events', 0)}</span>
            <span class="metric-label">Total Events</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{data.get('total_alerts', 0)}</span>
            <span class="metric-label">Total Alerts</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="color: #dc2626;">{data.get('active_alerts', 0)}</span>
            <span class="metric-label">Active Alerts</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value" style="color: #22c55e;">{data.get('active_cameras', 0)}/{data.get('total_cameras', 0)}</span>
            <span class="metric-label">Cameras Online</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Charts row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Events by Type")
        events_by_type = data.get("events_by_type", {})
        if events_by_type:
            color_map = {
                "fight": "#dc2626", "fall": "#f97316", "crowd_panic": "#eab308",
                "loitering": "#3b82f6", "suspicious_behavior": "#8b5cf6", "vandalism": "#ec4899",
            }
            colors = [color_map.get(t, "#8b5cf6") for t in events_by_type.keys()]
            labels = [t.replace("_", " ").title() for t in events_by_type.keys()]

            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=list(events_by_type.values()),
                marker_color=colors,
                text=list(events_by_type.values()),
                textposition="auto",
            )])
            fig.update_layout(
                template="plotly_dark",
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=60),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No event data available yet")

    with col_right:
        st.markdown("#### Alerts by Priority")
        alerts_by_priority = data.get("alerts_by_priority", {})
        if alerts_by_priority:
            priority_colors = {"P1": "#dc2626", "P2": "#f59e0b", "P3": "#3b82f6"}
            fig = go.Figure(data=[go.Pie(
                labels=list(alerts_by_priority.keys()),
                values=list(alerts_by_priority.values()),
                marker=dict(colors=[priority_colors.get(p, "#8b5cf6") for p in alerts_by_priority.keys()]),
                hole=0.5,
                textinfo="label+value",
                textfont_color="white",
            )])
            fig.update_layout(
                template="plotly_dark",
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                legend=dict(font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No alert data available yet")

    # Charts row 2
    st.divider()
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("#### Events by Risk Level")
        events_by_risk = data.get("events_by_risk", {})
        if events_by_risk:
            risk_colors = {"critical": "#dc2626", "high": "#f97316", "medium": "#eab308", "low": "#22c55e"}
            labels = [r.title() for r in events_by_risk.keys()]
            colors = [risk_colors.get(r, "#8b5cf6") for r in events_by_risk.keys()]

            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=list(events_by_risk.values()),
                marker_color=colors,
                text=list(events_by_risk.values()),
                textposition="auto",
            )])
            fig.update_layout(
                template="plotly_dark",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No risk data available yet")

    with col_r2:
        st.markdown("#### Recent Event Feed")
        events = api_get("/events/recent", params={"limit": 10}) or []
        if events:
            for e in events:
                icon = {
                    "fight": "⚔️", "fall": "🤕", "crowd_panic": "🏃",
                    "loitering": "🚶", "suspicious_behavior": "🕵️", "vandalism": "💥",
                }.get(e.get("event_type", ""), "⚠️")
                risk_badge = {
                    "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢",
                }.get(e.get("risk_level", ""), "⚪")
                st.markdown(
                    f"{icon} **{e.get('event_type', '?').replace('_', ' ').title()}** "
                    f"{risk_badge} — "
                    f"Cam {e.get('camera_id')}, {e.get('confidence', 0):.0%} conf, "
                    f"{e.get('timestamp', '')[:16]}"
                )
        else:
            st.caption("No recent events")

    # Charts row 3: Confidence distribution
    st.divider()
    col_l3, col_r3 = st.columns(2)

    with col_l3:
        st.markdown("#### Confidence Distribution")
        events = api_get("/events/recent", params={"limit": 100}) or []
        if events:
            confidences = [e.get("confidence", 0) for e in events if e.get("confidence")]
            if confidences:
                fig = go.Figure(data=[go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    marker_color="#8b5cf6",
                    opacity=0.8,
                )])
                fig.update_layout(
                    template="plotly_dark",
                    height=300,
                    xaxis_title="Confidence Score",
                    yaxis_title="Count",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No confidence data available")
        else:
            st.caption("No events to analyze")

    with col_r3:
        st.markdown("#### Events Over Time")
        events = api_get("/events/recent", params={"limit": 50}) or []
        if events:
            # Group by hour
            from collections import Counter
            hours = [e.get("timestamp", "")[:13] for e in events if e.get("timestamp")]
            hour_counts = Counter(hours)
            if hour_counts:
                sorted_hours = sorted(hour_counts.keys())
                fig = go.Figure(data=[go.Scatter(
                    x=sorted_hours,
                    y=[hour_counts[h] for h in sorted_hours],
                    mode="lines+markers",
                    line=dict(color="#8b5cf6", width=2),
                    marker=dict(size=6),
                )])
                fig.update_layout(
                    template="plotly_dark",
                    height=300,
                    xaxis_title="Time (Hour)",
                    yaxis_title="Events",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No events to analyze")
