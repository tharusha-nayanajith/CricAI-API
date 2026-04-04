"""
AI Feedback Generator
Generates realistic coaching feedback using Gemini AI via Google GenAI SDK
"""

import os
import json
from typing import Dict, List
from google import genai
from google.genai.errors import APIError  # Import specific error for better handling


class AIFeedbackGenerator:
    """Generate realistic coaching feedback using Gemini AI"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize AI feedback generator
        
        Args:
            api_key: Gemini API key (defaults to environment variable GEMINI_API_KEY)
        """
        # --- CHANGE 1: Use GEMINI_API_KEY and Google GenAI Client ---
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not set. Falling back to rule-based feedback.")
            self.client = None
        else:
            # Initialize the Gemini Client. The SDK automatically detects the environment variable
            # but we pass the key explicitly for consistency with the original code structure.
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                # Handle cases where client initialization might fail unexpectedly
                print(f"Gemini Client initialization failed: {e}. Falling back to rule-based feedback.")
                self.client = None

    # --- NO CHANGE: generate_stance_feedback logic remains the same for the fallback system ---
    def generate_stance_feedback(self, consistency_data: Dict) -> Dict[str, str]:
        """
        Generate comprehensive coaching feedback based on consistency analysis
        
        Args:
            consistency_data: Stance consistency analysis results
            
        Returns:
            Dictionary with different types of feedback
        """
        if self.client is None:
            return self._generate_rule_based_feedback(consistency_data)
        
        try:
            return self._generate_ai_feedback(consistency_data)
        # Catch the specific API Error from the Gemini SDK
        except APIError as e:
            print(f"Gemini API feedback generation failed: {e}. Using rule-based fallback.")
            return self._generate_rule_based_feedback(consistency_data)
        except Exception as e:
            # Catch other non-API errors (e.g., JSON parsing failure)
            print(f"AI feedback generation failed: {e}. Using rule-based fallback.")
            return self._generate_rule_based_feedback(consistency_data)
    
    def _generate_ai_feedback(self, consistency_data: Dict) -> Dict[str, str]:
        """Generate feedback using Gemini AI"""
        
        prompt = f"""You are an expert cricket batting coach analyzing a player's stance consistency.

Analysis Results:
- Overall Consistency Score: {consistency_data['overall_consistency']:.1f}%
- Number of Videos Analyzed: {consistency_data['total_videos']}
- Consistency Standard Deviation: {consistency_data['consistency_std']:.1f}
- Most Consistent Stance: Video {consistency_data['most_consistent']['video_index']} ({consistency_data['most_consistent']['consistency_score']:.1f}%)
- Least Consistent Stance: Video {consistency_data['least_consistent']['video_index']} ({consistency_data['least_consistent']['consistency_score']:.1f}%)

Individual Scores:
{json.dumps(consistency_data['individual_scores'], indent=2)}

Please provide:
1. A brief overall assessment (2-3 sentences) about their stance consistency
2. Specific strengths identified in their stance
3. Areas for improvement with actionable advice
4. A motivational closing remark

Format your response as JSON with keys: "overall_assessment", "strengths", "improvements", "motivation"
Keep each section concise and coaching-focused."""

        # --- CHANGE 2: Replace Anthropic API call with Gemini API call ---
        # The prompt is exactly the same, which is perfect for a direct replacement.
        
        # Use a model suitable for structured, high-quality output.
        # gemini-2.5-pro or gemini-2.5-flash are good choices.
        model_name = "gemini-2.5-flash" 

        response = self.client.models.generate_content(
            model=model_name,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={
                "max_output_tokens": 1000,
                # Force the model to output a JSON object using response schema
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "overall_assessment": {"type": "string"},
                        "strengths": {"type": "string"},
                        "improvements": {"type": "string"},
                        "motivation": {"type": "string"},
                    },
                    "required": ["overall_assessment", "strengths", "improvements", "motivation"]
                }
            }
        )
        
        # Parse AI response
        # The content should be a valid JSON string due to the config
        response_text = response.text
        
        # Try to extract JSON from response
        try:
            # The model is configured for JSON output, so we expect a clean parse.
            feedback = json.loads(response_text)
        except json.JSONDecodeError:
            # Manual parsing fallback if JSON config fails to produce clean output (unlikely with this setup)
            print(f"Warning: JSON Decode Error on response: {response_text[:100]}...")
            feedback = {
                "overall_assessment": response_text[:200],
                "strengths": "AI-generated feedback parsing in progress (Check JSON format)",
                "improvements": "Please review your individual scores",
                "motivation": "Keep practicing for better consistency!"
            }
        
        return feedback
    
    # --- NO CHANGE: Rule-based fallback remains as is ---
    def _generate_rule_based_feedback(self, consistency_data: Dict) -> Dict[str, str]:
        """Generate feedback using rule-based logic (fallback)"""
        
        overall_score = consistency_data['overall_consistency']
        std_dev = consistency_data['consistency_std']
        total_videos = consistency_data['total_videos']
        
        # Overall Assessment
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
        
        # Strengths
        if std_dev < 5:
            strengths = f"Your stance stability is impressive - all {total_videos} stances are within a tight range (σ={std_dev:.1f}). "
        elif std_dev < 10:
            strengths = f"You maintain reasonable stability across most deliveries (σ={std_dev:.1f}). "
        else:
            strengths = f"Some individual stances show good form. "
        
        # Best performance
        best_video = consistency_data['most_consistent']['video_index']
        strengths += f"Video {best_video} demonstrates your best stance positioning - use this as your reference point."
        
        # Improvements
        improvements_list = []
        
        if std_dev > 10:
            improvements_list.append("High variation between stances suggests inconsistent setup routine. Develop a pre-delivery ritual to ensure identical positioning each time.")
        
        if overall_score < 80:
            improvements_list.append("Focus on maintaining consistent feet positioning - shoulder-width apart, weight evenly distributed.")
            improvements_list.append("Keep your head still and eyes level throughout the stance phase.")
        
        worst_video = consistency_data['least_consistent']['video_index']
        improvements_list.append(f"Review Video {worst_video} carefully - it shows the most deviation from your average stance. Identify specific differences (feet placement, hand position, body alignment).")
        
        if len(consistency_data['individual_scores']) >= 5:
            low_scores = [s for s in consistency_data['individual_scores'] if s['consistency_score'] < 70]
            if low_scores:
                improvements_list.append(f"{len(low_scores)} of your stances fall below 70% consistency. Practice with a mirror or video recording to ensure repeatability.")
        
        improvements = " ".join(improvements_list)
        
        # Motivation
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
            "assessment_level": assessment_level
        }
    
    # --- NO CHANGE: generate_comparison_insights remains as is ---
    def generate_comparison_insights(self, pairwise_similarities: List[Dict]) -> List[str]:
        """Generate insights from pairwise comparisons"""
        
        insights = []
        
        # Find most similar pair
        most_similar = max(pairwise_similarities, key=lambda x: x['similarity'])
        insights.append(
            f"Videos {most_similar['video_1']} and {most_similar['video_2']} show the highest similarity "
            f"({most_similar['similarity']:.1f}%), indicating consistent technique in these attempts."
        )
        
        # Find least similar pair
        least_similar = min(pairwise_similarities, key=lambda x: x['similarity'])
        insights.append(
            f"Videos {least_similar['video_1']} and {least_similar['video_2']} differ the most "
            f"({least_similar['similarity']:.1f}%). Review these to identify inconsistency patterns."
        )
        
        # Overall spread
        similarities = [p['similarity'] for p in pairwise_similarities]
        avg_similarity = sum(similarities) / len(similarities)
        
        if avg_similarity >= 85:
            insights.append("Your stances are remarkably similar to each other, showing excellent repeatability.")
        elif avg_similarity >= 75:
            insights.append("Most of your stances are quite similar, with some minor variations to address.")
        else:
            insights.append("There's notable variation between different stance attempts. Focus on developing a consistent setup routine.")
        
        return insights