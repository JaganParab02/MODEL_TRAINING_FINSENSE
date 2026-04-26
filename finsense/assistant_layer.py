"""
Assistant Layer — Lightweight Decision Guidance
================================================
Provides structured recommendations BEFORE the agent acts,
based on current environment state, active events, context,
and optional memory signals.

This is a system-integrated layer, NOT a chatbot.
It does not replace the agent — it guides decisions.
"""


def get_assistant_tip(obs: dict, memory=None) -> dict:
    """
    Returns structured decision guidance based on:
    - Budget state
    - Active events
    - Context
    - Optional memory signals
    """

    exp = obs.get("current_expense", {}) or {}
    events = obs.get("active_events", []) or []

    amount = float(exp.get("amount", 0))
    category = exp.get("category", "")
    necessity = exp.get("necessity_tag", "")
    context = exp.get("context", "")

    daily_allowance = float(obs.get("daily_allowance", 0))
    goal_remaining = float(obs.get("goal_remaining", 0))

    recommendation = "allow"
    reasons = []

    # Budget constraint
    if amount > daily_allowance:
        recommendation = "reduce"
        reasons.append("exceeds daily allowance")

    # Event awareness
    for event in events:
        if event == "fuel_crisis" and category == "transport":
            recommendation = "reduce"
            reasons.append("transport inflated due to fuel_crisis")
        elif event == "inflation" and category in ["food", "utility"]:
            recommendation = "reduce"
            reasons.append(f"prices inflated due to {event}")
        else:
            reasons.append(f"active {event}")

    # Context awareness
    if context == "emergency" and necessity != "essential":
        recommendation = "reduce"
        reasons.append("non-essential spending during emergency context")
    elif context == "weekend" and necessity == "discretionary":
        recommendation = "reduce"
        reasons.append("discretionary spending during weekend context")

    # Memory signal (safe optional)
    if memory is not None:
        try:
            bias_action, confidence = memory.get_memory_bias_with_confidence(
                expense_type=category,
                context=context,
                event_type=events[0] if events else "none"
            )
            if bias_action and confidence >= 0.65:
                recommendation = bias_action
                reasons.append("based on past successful episodes from memory")
        except Exception:
            pass

    return {
        "recommendation": recommendation,
        "reason": ", ".join(reasons) if reasons else "within safe limits"
    }
