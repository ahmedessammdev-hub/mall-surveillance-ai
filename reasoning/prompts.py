"""
Prompt templates for the LLM reasoning engine.

Reusable templates that format event data, similar events,
and security rules into structured prompts.
"""

SYSTEM_PROMPT = """You are an AI security analyst for a shopping mall surveillance system.
Your role is to analyze security events detected by computer vision models and provide:
1. Accurate event classification
2. Risk assessment
3. Whether immediate intervention is needed
4. Clear reasoning for your assessment
5. Specific recommended actions

You must respond ONLY in valid JSON format. No additional text outside the JSON.
Be concise but thorough in your reasoning."""


EVENT_ANALYSIS_PROMPT = """Analyze the following security event detected by the surveillance system.

## Current Event
- **Event Type (detected):** {event_type}
- **Camera ID:** {camera_id}
- **Timestamp:** {timestamp}
- **Confidence Score:** {confidence}
- **Person Count:** {person_count}
- **Crowd Density:** {crowd_density}

### Behavior Scores
{behavior_scores}

### Motion Features
{motion_features}

### Involved Persons
{involved_tracks}

## Similar Historical Events
{similar_events}

## Security Rules
- P1 (Critical): Fights, Medical Emergencies → Immediate response required
- P2 (High): Crowd Panic, Vandalism → Rapid response required
- P3 (Medium): Loitering, Suspicious Activity → Monitor and assess

## Camera Context
{camera_context}

---

Respond with a JSON object containing exactly these fields:
{{
    "event_type": "the confirmed or corrected event type",
    "risk_level": "low|medium|high|critical",
    "requires_intervention": true or false,
    "reasoning": "2-3 sentence explanation of your assessment",
    "recommended_action": "specific action for security personnel"
}}"""


RISK_ASSESSMENT_PROMPT = """Assess the risk level of this security event.

Event: {event_type}
Confidence: {confidence}
Person Count: {person_count}
Location: {camera_id}
Behavior Indicators: {behavior_scores}

Historical similar events had these outcomes:
{historical_outcomes}

Respond with a JSON object:
{{
    "risk_level": "low|medium|high|critical",
    "risk_factors": ["list", "of", "risk", "factors"],
    "mitigating_factors": ["list", "of", "mitigating", "factors"],
    "escalation_probability": 0.0 to 1.0
}}"""


ACTION_RECOMMENDATION_PROMPT = """Given this security event, recommend specific actions.

Event Type: {event_type}
Risk Level: {risk_level}
Location: Camera {camera_id}
Person Count: {person_count}
Reasoning: {reasoning}

Respond with a JSON object:
{{
    "immediate_actions": ["action 1", "action 2"],
    "follow_up_actions": ["action 1", "action 2"],
    "personnel_needed": "number and type of personnel",
    "estimated_response_time": "in minutes"
}}"""


def format_event_analysis_prompt(
    event,
    similar_events: list,
    camera_context: dict,
) -> str:
    """Format the event analysis prompt with actual data."""

    # Format behavior scores
    behavior_str = "\n".join(
        f"- {k}: {v:.3f}" for k, v in event.behavior_scores.items()
    ) or "- No behavior scores available"

    # Format motion features
    motion_str = "\n".join(
        f"- {k}: {v}" for k, v in event.motion_features.items()
    ) or "- No motion features available"

    # Format involved tracks
    tracks_str = "\n".join(
        f"- Track {t.track_id}: speed={t.speed:.1f}px/s, role={t.role}"
        for t in event.involved_tracks
    ) or "- No specific persons tracked"

    # Format similar events
    if similar_events:
        similar_str = "\n".join(
            f"- Event {se.event_id}: type={se.metadata.get('event_type', 'unknown')}, "
            f"similarity={se.score:.3f}, camera={se.metadata.get('camera_id', 'unknown')}"
            for se in similar_events[:5]
        )
    else:
        similar_str = "- No similar historical events found"

    # Format camera context
    ctx_str = "\n".join(
        f"- {k}: {v}" for k, v in camera_context.items()
    ) or "- No additional camera context"

    return EVENT_ANALYSIS_PROMPT.format(
        event_type=event.event_type.value,
        camera_id=event.camera_id,
        timestamp=event.timestamp.isoformat(),
        confidence=f"{event.confidence:.3f}",
        person_count=event.person_count,
        crowd_density=f"{event.crowd_density:.6f}",
        behavior_scores=behavior_str,
        motion_features=motion_str,
        involved_tracks=tracks_str,
        similar_events=similar_str,
        camera_context=ctx_str,
    )
