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

    def generate_stance_feedback(self, consistency_data: dict[str, Any]) -> dict[str, str]:
        if self.client is None:
            return self._generate_rule_based_feedback(consistency_data)

        prompt = f"""You are an expert cricket batting coach analyzing a player's stance consistency.

Analysis Results:
- Overall Consistency Score: {consistency_data['overall_consistency']:.1f}%
- Number of Videos Analyzed: {consistency_data['total_videos']}
- Consistency Standard Deviation: {consistency_data['consistency_std']:.1f}
- Most Consistent Stance: Video {consistency_data['most_consistent']['video_index']} ({consistency_data['most_consistent']['consistency_score']:.1f}%)
- Least Consistent Stance: Video {consistency_data['least_consistent']['video_index']} ({consistency_data['least_consistent']['consistency_score']:.1f}%)

Individual Scores:
{json.dumps(consistency_data['individual_scores'], indent=2)}

Respond in JSON with keys overall_assessment, strengths, improvements, motivation.
Keep each section concise and coaching-focused."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={
                    "max_output_tokens": 1000,
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "overall_assessment": {"type": "string"},
                            "strengths": {"type": "string"},
                            "improvements": {"type": "string"},
                            "motivation": {"type": "string"},
                        },
                        "required": [
                            "overall_assessment",
                            "strengths",
                            "improvements",
                            "motivation",
                        ],
                    },
                },
            )
            return json.loads(response.text)
        except Exception as exc:
            logger.warning("Gemini stance feedback generation failed: {}", exc)
            return self._generate_rule_based_feedback(consistency_data)

    def generate_comparison_insights(self, pairwise_similarities: list[dict[str, Any]]) -> list[str]:
        insights = []

        most_similar = max(pairwise_similarities, key=lambda x: x['similarity'])
        insights.append(
            f"Videos {most_similar['video_1']} and {most_similar['video_2']} show the highest similarity "
            f"({most_similar['similarity']:.1f}%), indicating consistent technique in these attempts."
        )

        least_similar = min(pairwise_similarities, key=lambda x: x['similarity'])
        insights.append(
            f"Videos {least_similar['video_1']} and {least_similar['video_2']} differ the most "
            f"({least_similar['similarity']:.1f}%). Review these to identify inconsistency patterns."
        )

        similarities = [p['similarity'] for p in pairwise_similarities]
        avg_similarity = sum(similarities) / len(similarities)

        if avg_similarity >= 85:
            insights.append("Your stances are remarkably similar to each other, showing excellent repeatability.")
        elif avg_similarity >= 75:
            insights.append("Most of your stances are quite similar, with some minor variations to address.")
        else:
            insights.append("There's notable variation between different stance attempts. Focus on developing a consistent setup routine.")

        return insights

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

    def _generate_rule_based_feedback(self, consistency_data: dict[str, Any]) -> dict[str, str]:
        overall_score = consistency_data['overall_consistency']
        std_dev = consistency_data['consistency_std']
        total_videos = consistency_data['total_videos']

        if overall_score >= 90:
            assessment_level = "Excellent"
            assessment = f"Outstanding stance consistency! Your {total_videos} stances show remarkable uniformity with a {overall_score:.1f}% consistency score. This level of technical consistency is characteristic of professional players."
        elif overall_score >= 80:
            assessment_level = "Very Good"
            assessment = f"Strong stance consistency achieved! You've maintained {overall_score:.1f}% consistency across {total_videos} deliveries. This shows solid technical foundation with minor variations that can be refined."
        elif overall_score >= 70:
            assessment_level = "Good"
            assessment = f"Good stance consistency at {overall_score:.1f}%. Your technique is developing well, though there's room for improvement in maintaining identical positioning across all {total_videos} deliveries."
        elif overall_score >= 60:
            assessment_level = "Moderate"
            assessment = f"Moderate consistency at {overall_score:.1f}%. Your stance varies noticeably between deliveries. Focus on developing muscle memory through repetition and deliberate practice."
        else:
            assessment_level = "Needs Work"
            assessment = f"Your stance consistency needs attention. At {overall_score:.1f}%, there's significant variation between deliveries. This inconsistency can affect shot execution and timing."

        if std_dev < 5:
            strengths = f"Your stance stability is impressive - all {total_videos} stances are within a tight range (σ={std_dev:.1f}). "
        elif std_dev < 10:
            strengths = f"You maintain reasonable stability across most deliveries (σ={std_dev:.1f}). "
        else:
            strengths = "Some individual stances show good form. "

        best_video = consistency_data['most_consistent']['video_index']
        strengths += f"Video {best_video} demonstrates your best stance positioning - use this as your reference point."

        improvements_list = []
        if std_dev > 10:
            improvements_list.append("High variation between stances suggests inconsistent setup routine. Develop a pre-delivery ritual to ensure identical positioning each time.")
        if overall_score < 80:
            improvements_list.append("Focus on maintaining consistent feet positioning - shoulder-width apart, weight evenly distributed.")
            improvements_list.append("Keep your head still and eyes level throughout the stance phase.")

        worst_video = consistency_data['least_consistent']['video_index']
        improvements_list.append(
            f"Review Video {worst_video} carefully - it shows the most deviation from your average stance. Identify specific differences (feet placement, hand position, body alignment)."
        )

        if len(consistency_data['individual_scores']) >= 5:
            low_scores = [s for s in consistency_data['individual_scores'] if s['consistency_score'] < 70]
            if low_scores:
                improvements_list.append(
                    f"{len(low_scores)} of your stances fall below 70% consistency. Practice with a mirror or video recording to ensure repeatability."
                )

        improvements = " ".join(improvements_list)

        if overall_score >= 85:
            motivation = "Exceptional work! Your consistency is at an elite level. Keep refining the details to maintain this standard under pressure situations."
        elif overall_score >= 75:
            motivation = "You're on the right track! With focused practice on the highlighted areas, you can achieve professional-level consistency."
        elif overall_score >= 60:
            motivation = "Remember, consistency is built through repetition. Every practice session is an opportunity to groove your technique. Stay patient and persistent!"
        else:
            motivation = "Don't be discouraged! Even professional players work continuously on stance consistency. Focus on one aspect at a time, and you'll see steady improvement."

        return {
            "overall_assessment": assessment,
            "strengths": strengths,
            "improvements": improvements,
            "motivation": motivation,
            "assessment_level": assessment_level,
        }
