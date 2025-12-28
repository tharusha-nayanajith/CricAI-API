"""
Stance Consistency Service
Main service for analyzing batting stance consistency
"""

import numpy as np
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.frame_extractor import FrameExtractor
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.pose_estimator import PoseEstimator
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.stance_detector import StanceDetector
from features.SHOT_CLASSIFICATION_SYSTEM.data_preprocessing.stance_consistency_calculator import StanceConsistencyCalculator
from features.SHOT_CLASSIFICATION_SYSTEM.utils.ai_feedback_generator import AIFeedbackGenerator


class StanceConsistencyService:
    """Service for stance consistency analysis"""
    
    def __init__(self):
        """Initialize all required components"""
        self.frame_extractor = FrameExtractor(fps=10)
        self.pose_estimator = PoseEstimator()
        self.stance_detector = StanceDetector()
        self.consistency_calculator = StanceConsistencyCalculator()
        self.feedback_generator = AIFeedbackGenerator()
        
        print("Stance Consistency Service initialized successfully")
    
    def process_single_video(self, video_path: str) -> Dict:
        """
        Process a single video and extract stance keypoints
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing stance data
        """
        print(f"\nProcessing video: {video_path}")
        
        # Step 1: Extract frames
        print("Step 1: Extracting frames...")
        frames, fps = self.frame_extractor.extract_frames(video_path)
        print(f"Extracted {len(frames)} frames at {fps} FPS")
        
        # Step 2: Detect stance and extract keypoints
        print("Step 2: Detecting stance and extracting keypoints...")
        stance_keypoints, stance_frame_idx = self.stance_detector.extract_stance_keypoints(
            frames, self.pose_estimator
        )
        
        result = {
        'video_path': video_path,
        'stance_keypoints': stance_keypoints,
        'stance_frame_index': int(stance_frame_idx),
        'total_frames': int(len(frames)),
        'stance_frame_percentage': float((stance_frame_idx / len(frames)) * 100)
}
        
        print(f"✓ Stance detected at frame {stance_frame_idx} ({result['stance_frame_percentage']:.1f}% into video)")
        
        return result
    
    def analyze_consistency(self, video_paths: List[str]) -> Dict:
        """
        Analyze stance consistency across multiple videos
        
        Args:
            video_paths: List of paths to video files (6 videos)
            
        Returns:
            Complete consistency analysis with feedback
        """
        if len(video_paths) < 2:
            raise ValueError("Need at least 2 videos for consistency analysis")
        
        if len(video_paths) > 10:
            raise ValueError("Maximum 10 videos allowed for analysis")
        
        print(f"\n{'='*70}")
        print(f"Starting Stance Consistency Analysis")
        print(f"Number of videos: {len(video_paths)}")
        print(f"{'='*70}")
        
        # Step 1: Process all videos and extract stance keypoints
        stance_data_list = []
        stance_keypoints_list = []
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[Video {i}/{len(video_paths)}]")
            try:
                stance_data = self.process_single_video(video_path)
                stance_data_list.append(stance_data)
                stance_keypoints_list.append(stance_data['stance_keypoints'])
            except Exception as e:
                print(f"Error processing video {i}: {str(e)}")
                raise
        
        print(f"\n{'='*70}")
        print("All videos processed successfully!")
        print(f"{'='*70}\n")
        
        # Step 2: Calculate consistency using cosine similarity
        print("Calculating stance consistency...")
        consistency_results = self.consistency_calculator.analyze_consistency(stance_keypoints_list)
        
        # Step 3: Generate AI feedback
        print("Generating coaching feedback...")
        feedback = self.feedback_generator.generate_stance_feedback(consistency_results)
        
        # Step 4: Generate comparison insights
        comparison_insights = self.feedback_generator.generate_comparison_insights(
            consistency_results['pairwise_similarities']
        )
        
        # Compile final results
        final_results = {
            'summary': {
                'total_videos_analyzed': len(video_paths),
                'overall_consistency_score': consistency_results['overall_consistency'],
                'consistency_rating': self._get_consistency_rating(
                    consistency_results['overall_consistency']
                ),
                'consistency_std': consistency_results['consistency_std']
            },
            'individual_video_scores': consistency_results['individual_scores'],
            'stance_detection_details': [
    {
        'video_index': int(i + 1),
        'stance_frame': int(data['stance_frame_index']),
        'total_frames': int(data['total_frames']),
        'stance_timing': f"{float(data['stance_frame_percentage']):.1f}%"
    }
    for i, data in enumerate(stance_data_list)
],
            'consistency_analysis': {
                'most_consistent_video': consistency_results['most_consistent'],
                'least_consistent_video': consistency_results['least_consistent'],
                'pairwise_similarities': consistency_results['pairwise_similarities']
            },
            'feedback': feedback,
            'insights': comparison_insights
        }
        
        print("\n" + "="*70)
        print("STANCE CONSISTENCY ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nOverall Consistency Score: {final_results['summary']['overall_consistency_score']:.1f}%")
        print(f"Rating: {final_results['summary']['consistency_rating']}")
        print("="*70 + "\n")
        
        return final_results
    
    def _get_consistency_rating(self, score: float) -> str:
        """Get rating label for consistency score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Moderate"
        else:
            return "Needs Improvement"


# Global service instance
_stance_service = None

def get_stance_service() -> StanceConsistencyService:
    """Get or create stance service instance"""
    global _stance_service
    if _stance_service is None:
        _stance_service = StanceConsistencyService()
    return _stance_service