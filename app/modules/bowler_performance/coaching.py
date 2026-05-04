from __future__ import annotations

import json
from statistics import mean
from typing import Any

from loguru import logger

from app.ai.google import get_google_genai_client, get_google_genai_status
from app.modules.bowler_performance.models import (
    BowlerCoachingFeedback,
    BowlerPerformanceResult,
)


def generate_single_delivery_coaching(
    result: BowlerPerformanceResult,
) -> BowlerCoachingFeedback:
    fallback = _build_single_delivery_fallback(result)
    return _generate_with_ai(
        prompt=_build_single_delivery_prompt(result),
        fallback=fallback,
    )


def generate_multi_delivery_coaching(
    deliveries: list[BowlerPerformanceResult],
) -> BowlerCoachingFeedback:
    fallback = _build_multi_delivery_fallback(deliveries)
    return _generate_with_ai(
        prompt=_build_multi_delivery_prompt(deliveries),
        fallback=fallback,
    )


def _generate_with_ai(
    *,
    prompt: str,
    fallback: BowlerCoachingFeedback,
) -> BowlerCoachingFeedback:
    status = get_google_genai_status()
    if not status.enabled:
        return fallback

    client = get_google_genai_client()
    if client is None:
        return fallback

    try:
        response = client.models.generate_content(
            model=status.model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={
                "max_output_tokens": 900,
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "analysisScope": {"type": "string"},
                        "sampleSize": {"type": "integer"},
                        "summary": {"type": "string"},
                        "strengths": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "improvements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "nextSteps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "analysisScope",
                        "sampleSize",
                        "summary",
                        "strengths",
                        "improvements",
                        "nextSteps",
                    ],
                },
            },
        )
        payload = json.loads(response.text)
        ai_feedback = BowlerCoachingFeedback.model_validate(payload)
    except Exception as exc:
        logger.warning("bowler_performance coaching generation failed: {}", exc)
        return fallback

    return _sanitize_feedback(ai_feedback, fallback)


def _sanitize_feedback(
    ai_feedback: BowlerCoachingFeedback,
    fallback: BowlerCoachingFeedback,
) -> BowlerCoachingFeedback:
    return BowlerCoachingFeedback(
        analysis_scope=(
            ai_feedback.analysis_scope
            if ai_feedback.analysis_scope in {"single_delivery", "multi_delivery"}
            else fallback.analysis_scope
        ),
        sample_size=max(1, int(ai_feedback.sample_size or fallback.sample_size)),
        summary=ai_feedback.summary.strip() if ai_feedback.summary.strip() else fallback.summary,
        strengths=_sanitize_list(ai_feedback.strengths, fallback.strengths),
        improvements=_sanitize_list(ai_feedback.improvements, fallback.improvements),
        next_steps=_sanitize_list(ai_feedback.next_steps, fallback.next_steps),
    )


def _sanitize_list(values: list[str], fallback: list[str]) -> list[str]:
    cleaned = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    return cleaned[:4] if cleaned else fallback


def _build_single_delivery_fallback(
    result: BowlerPerformanceResult,
) -> BowlerCoachingFeedback:
    strengths: list[str] = []
    improvements: list[str] = []
    next_steps: list[str] = []

    if result.speed_kmh is not None:
        strengths.append(f"Recorded pace is {result.speed_kmh:.1f} km/h, giving you a clear baseline.")
    if result.length_class is not None:
        strengths.append(f"Length profile reads as {result.length_class.value.replace('_', ' ')}.")
    if result.trajectory_reliable:
        strengths.append("Trajectory reconstruction was reliable enough to trust the ball-flight read.")
    elif result.trajectory_warning:
        improvements.append(result.trajectory_warning)

    delivery_features = result.delivery_features
    wicket_risk = result.wicket_risk
    if delivery_features is not None:
        if delivery_features.line_bucket:
            improvements.append(
                f"Current line projects to {delivery_features.line_bucket.replace('_', ' ')}; tighten the release to hit your intended channel more often."
            )
        if delivery_features.pace_bucket:
            next_steps.append(
                f"Track whether your next spell stays in the {delivery_features.pace_bucket.replace('_', ' ')} pace bucket."
            )
    if wicket_risk is not None:
        next_steps.append(
            f"Wicket chance is {wicket_risk.percentage:.1f}%, so compare this delivery against your best wicket-taking balls."
        )
    if result.swing_metres is not None:
        next_steps.append(
            f"Measured swing was {result.swing_metres:.2f} m; repeat the same seam and wrist cues if that movement is intentional."
        )

    if not strengths:
        strengths.append("The delivery produced enough tracking data to build a bowling profile.")
    if not improvements:
        improvements.append("Focus on repeating the same release position and target line for the next few balls.")
    if not next_steps:
        next_steps.append("Use the next over to repeat this delivery shape and compare pace, line, and bounce.")

    return BowlerCoachingFeedback(
        analysis_scope="single_delivery",
        sample_size=1,
        summary=_build_single_delivery_summary(result),
        strengths=strengths[:3],
        improvements=improvements[:3],
        next_steps=next_steps[:3],
    )


def _build_multi_delivery_fallback(
    deliveries: list[BowlerPerformanceResult],
) -> BowlerCoachingFeedback:
    speeds = [delivery.speed_kmh for delivery in deliveries if delivery.speed_kmh is not None]
    swings = [abs(delivery.swing_metres) for delivery in deliveries if delivery.swing_metres is not None]
    wicket_risks = [
        delivery.wicket_risk.percentage
        for delivery in deliveries
        if delivery.wicket_risk is not None
    ]
    length_counts: dict[str, int] = {}
    line_counts: dict[str, int] = {}

    for delivery in deliveries:
        if delivery.length_class is not None:
            key = delivery.length_class.value
            length_counts[key] = length_counts.get(key, 0) + 1
        line_bucket = delivery.delivery_features.line_bucket if delivery.delivery_features else None
        if line_bucket:
            line_counts[line_bucket] = line_counts.get(line_bucket, 0) + 1

    modal_length = max(length_counts, key=length_counts.get) if length_counts else None
    modal_line = max(line_counts, key=line_counts.get) if line_counts else None
    strengths = [
        f"Average pace across the spell is {mean(speeds):.1f} km/h." if speeds else "You now have a multi-ball pace baseline for this spell.",
        (
            f"Most deliveries landed in the {modal_length.replace('_', ' ')} zone."
            if modal_length
            else "The sample gives you a usable first read on length distribution."
        ),
    ]
    improvements = [
        (
            f"Line pattern is clustering around {modal_line.replace('_', ' ')}; refine alignment if that is not your intended attacking line."
            if modal_line
            else "Line variation is worth reviewing ball by ball to sharpen your target lane."
        ),
        (
            f"Average wicket threat is {mean(wicket_risks):.1f}%, so review which deliveries created the biggest threat."
            if wicket_risks
            else "Pair the ball-flight data with match intent so each delivery type has a clearer purpose."
        ),
    ]
    next_steps = [
        "Use this spell summary as your benchmark and compare the next set of deliveries against it.",
        (
            f"Average swing magnitude is {mean(swings):.2f} m; monitor whether you can reproduce that movement on demand."
            if swings
            else "Track whether your next spell produces a more repeatable movement pattern."
        ),
    ]
    return BowlerCoachingFeedback(
        analysis_scope="multi_delivery",
        sample_size=len(deliveries),
        summary=_build_multi_delivery_summary(
            sample_size=len(deliveries),
            avg_speed=(mean(speeds) if speeds else None),
            modal_length=modal_length,
            modal_line=modal_line,
        ),
        strengths=[item for item in strengths if item][:3],
        improvements=[item for item in improvements if item][:3],
        next_steps=[item for item in next_steps if item][:3],
    )


def _build_single_delivery_summary(result: BowlerPerformanceResult) -> str:
    parts: list[str] = []
    if result.speed_kmh is not None:
        parts.append(f"{result.speed_kmh:.1f} km/h pace")
    if result.length_class is not None:
        parts.append(f"{result.length_class.value.replace('_', ' ')} length")
    if result.delivery_features and result.delivery_features.line_bucket:
        parts.append(
            f"{result.delivery_features.line_bucket.replace('_', ' ')} line"
        )
    if not parts:
        return "Single-delivery bowling review is ready."
    return "Single-delivery bowling review: " + ", ".join(parts) + "."


def _build_multi_delivery_summary(
    *,
    sample_size: int,
    avg_speed: float | None,
    modal_length: str | None,
    modal_line: str | None,
) -> str:
    parts = [f"{sample_size} deliveries reviewed"]
    if avg_speed is not None:
        parts.append(f"{avg_speed:.1f} km/h average pace")
    if modal_length is not None:
        parts.append(f"{modal_length.replace('_', ' ')} as the main length")
    if modal_line is not None:
        parts.append(f"{modal_line.replace('_', ' ')} as the common line")
    return "Spell review: " + ", ".join(parts) + "."


def _build_single_delivery_prompt(result: BowlerPerformanceResult) -> str:
    payload = _single_delivery_payload(result)
    return f"""You are an expert cricket fast-bowling coach writing UI-safe feedback.

Analyze this single delivery:
{json.dumps(payload, indent=2)}

Return valid JSON only.
Rules:
- analysisScope must be "single_delivery".
- sampleSize must be 1.
- summary must be 1-2 concise sentences.
- strengths, improvements, and nextSteps must each contain 1 to 3 short bullet-style strings.
- Keep the coaching practical, specific, and understandable for a mobile app user.
- Do not invent measurements that are not present in the payload.
"""


def _build_multi_delivery_prompt(deliveries: list[BowlerPerformanceResult]) -> str:
    payload = {
        "sampleSize": len(deliveries),
        "deliveries": [_single_delivery_payload(delivery) for delivery in deliveries[:12]],
        "aggregate": _aggregate_payload(deliveries),
    }
    return f"""You are an expert cricket bowling coach writing UI-safe spell feedback.

Analyze this multi-delivery bowling sample:
{json.dumps(payload, indent=2)}

Return valid JSON only.
Rules:
- analysisScope must be "multi_delivery".
- sampleSize must match the payload sampleSize.
- summary must describe the spell at a high level in 1-2 concise sentences.
- strengths, improvements, and nextSteps must each contain 1 to 3 short bullet-style strings.
- Focus on repeatability, control, and wicket-taking potential across the sample.
- Do not invent measurements or claim certainty beyond the payload.
"""


def _single_delivery_payload(result: BowlerPerformanceResult) -> dict[str, Any]:
    delivery_features = result.delivery_features
    wicket_risk = result.wicket_risk
    return {
        "speedKmh": round(result.speed_kmh, 2) if result.speed_kmh is not None else None,
        "swingMetres": (
            round(result.swing_metres, 3) if result.swing_metres is not None else None
        ),
        "lengthClass": result.length_class.value if result.length_class is not None else None,
        "trajectoryReliable": result.trajectory_reliable,
        "trajectoryWarning": result.trajectory_warning,
        "lineBucket": delivery_features.line_bucket if delivery_features is not None else None,
        "physicalLineBucket": (
            delivery_features.physical_line_bucket if delivery_features is not None else None
        ),
        "paceBucket": delivery_features.pace_bucket if delivery_features is not None else None,
        "releaseToBounceMs": (
            round(delivery_features.release_to_bounce_ms, 1)
            if delivery_features is not None and delivery_features.release_to_bounce_ms is not None
            else None
        ),
        "bounceToContactMs": (
            round(delivery_features.bounce_to_contact_ms, 1)
            if delivery_features is not None and delivery_features.bounce_to_contact_ms is not None
            else None
        ),
        "wicketRiskPercentage": (
            round(wicket_risk.percentage, 2) if wicket_risk is not None else None
        ),
        "wicketRiskBand": wicket_risk.risk_band.value if wicket_risk is not None else None,
    }


def _aggregate_payload(deliveries: list[BowlerPerformanceResult]) -> dict[str, Any]:
    speeds = [delivery.speed_kmh for delivery in deliveries if delivery.speed_kmh is not None]
    swings = [abs(delivery.swing_metres) for delivery in deliveries if delivery.swing_metres is not None]
    wicket_risks = [
        delivery.wicket_risk.percentage
        for delivery in deliveries
        if delivery.wicket_risk is not None
    ]
    length_counts: dict[str, int] = {}
    line_counts: dict[str, int] = {}
    for delivery in deliveries:
        if delivery.length_class is not None:
            key = delivery.length_class.value
            length_counts[key] = length_counts.get(key, 0) + 1
        line_bucket = delivery.delivery_features.line_bucket if delivery.delivery_features else None
        if line_bucket:
            line_counts[line_bucket] = line_counts.get(line_bucket, 0) + 1
    return {
        "avgSpeedKmh": round(mean(speeds), 2) if speeds else None,
        "maxSpeedKmh": round(max(speeds), 2) if speeds else None,
        "avgSwingMagnitude": round(mean(swings), 3) if swings else None,
        "avgWicketRiskPercentage": round(mean(wicket_risks), 2) if wicket_risks else None,
        "lengthBreakdown": length_counts,
        "lineBreakdown": line_counts,
    }
