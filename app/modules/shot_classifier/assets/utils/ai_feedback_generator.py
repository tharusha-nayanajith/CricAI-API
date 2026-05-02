"""AI feedback helpers for shot classifier and future AI-backed modules."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

import app.config as config_module
from app.ai.google import get_google_genai_client


class AIFeedbackGenerator:
    """Generate coaching feedback using the app-level AI configuration."""

    def __init__(self, model_name: str | None = None, client: Any | None = None):
        settings = config_module.get_settings()
        self.model_name = model_name or settings.ai_model
        self.client = client if client is not None else get_google_genai_client()

    def generate_feedback(
        self,
        predicted_shot: str,
        confidence: float,
        mistakes: list[dict[str, Any]],
    ) -> str:
        if self.client is None:
            return self._rule_based_shot_feedback(predicted_shot, confidence, mistakes)

        prompt = self._build_shot_feedback_prompt(predicted_shot, confidence, mistakes)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"max_output_tokens": 220},
            )
        except Exception as exc:
            logger.warning("Gemini shot feedback generation failed: {}", exc)
            return self._rule_based_shot_feedback(predicted_shot, confidence, mistakes)

        feedback = getattr(response, "text", "")
        if isinstance(feedback, str) and feedback.strip():
            return feedback.strip()
        return self._rule_based_shot_feedback(predicted_shot, confidence, mistakes)

    def _build_shot_feedback_prompt(
        self,
        predicted_shot: str,
        confidence: float,
        mistakes: list[dict[str, Any]],
    ) -> str:
        mistakes_json = json.dumps(mistakes[:5], indent=2)
        return f"""You are an expert cricket batting coach.

A model classified the player's shot as {predicted_shot} with confidence {confidence:.2f}.
Detected movement deviations:
{mistakes_json}

Write 2-3 concise coaching sentences.
- Mention whether the shot identification is high or low confidence.
- Mention the most important correction if deviations are present.
- Keep the tone practical and specific."""

    def _rule_based_shot_feedback(
        self,
        predicted_shot: str,
        confidence: float,
        mistakes: list[dict[str, Any]],
    ) -> str:
        if confidence < 0.7:
            return f"Low confidence prediction for {predicted_shot}. Review form and positioning."
        if mistakes:
            return f"Significant deviation from correct {predicted_shot} form. Focus on improving the identified body positions."
        return f"Good {predicted_shot} execution. Continue practicing and refining technique."
