"""
3D Skeleton Animator
Generates professional biomechanical skeleton visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle
import io
import tempfile
import os
import matplotlib
import base64
from typing import Dict, List, Tuple

matplotlib.use("Agg")  # REQUIRED for FastAPI / servers


class SkeletonAnimator:
    """Generate 3D skeleton animations from pose data"""
    
    def __init__(self):
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Skeleton connections
        self.skeleton_lines = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Torso
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Right arm
            (6, 8), (8, 10),
            # Left arm
            (5, 7), (7, 9),
            # Right leg
            (12, 14), (14, 16),
            # Left leg
            (11, 13), (13, 15)
        ]
        
        # Joint colors by error severity
        self.severity_colors = {
            'critical': '#e74c3c',    # Red
            'major': '#f39c12',       # Orange
            'minor': '#f1c40f',       # Yellow
            'normal': '#2ecc71'       # Green
        }
        
        # Joint to keypoint mapping
        self.joint_keypoints = {
            'front_elbow': [6, 8, 10],   # right shoulder, elbow, wrist
            'back_elbow': [5, 7, 9],     # left shoulder, elbow, wrist
            'front_knee': [12, 14, 16],  # right hip, knee, ankle
            'back_knee': [11, 13, 15],   # left hip, knee, ankle
            'torso_bend': [5, 11, 13],   # left shoulder, hip, knee
            'shoulder_rotation': [5, 6, 12]  # left shoulder, right shoulder, right hip
        }
    
    def generate_3d_skeleton(self, keypoints: np.ndarray, mistakes: List[Dict],
                            view_angle: Tuple[int, int] = (30, 45)) -> str:
        """
        Generate 3D skeleton visualization with error highlighting
        
        Args:
            keypoints: (17, 2) array of 2D keypoints
            mistakes: List of detected mistakes
            view_angle: (elevation, azimuth) for 3D view
            
        Returns:
            Base64-encoded image
        """
        # Convert 2D to pseudo-3D (estimate depth from pose)
        keypoints_3d = self._estimate_3d_from_2d(keypoints)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Mark error joints
        error_joints = self._get_error_joint_indices(mistakes)
        
        # Draw skeleton
        self._draw_3d_skeleton(ax, keypoints_3d, error_joints)
        
        # Set view
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Labels and title
        ax.set_xlabel('X (lateral)')
        ax.set_ylabel('Y (depth)')
        ax.set_zlabel('Z (vertical)')
        ax.set_title('Biomechanical Analysis - 3D Skeleton', fontsize=14, fontweight='bold')
        
        # Equal aspect ratio
        self._set_equal_aspect_3d(ax, keypoints_3d)
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_base64
    
    def generate_comparison_view(self, actual_keypoints: np.ndarray,
                                intended_shot_prototype: np.ndarray,
                                mistakes: List[Dict]) -> str:
        """
        Generate side-by-side comparison: Actual vs Intended
        
        Args:
            actual_keypoints: User's actual pose
            intended_shot_prototype: Correct pose for intended shot
            mistakes: Detected mistakes
            
        Returns:
            Base64-encoded comparison image
        """
        fig = plt.figure(figsize=(18, 8))
        
        # Left: Actual execution (with errors marked)
        ax1 = fig.add_subplot(121, projection='3d')
        actual_3d = self._estimate_3d_from_2d(actual_keypoints)
        error_joints = self._get_error_joint_indices(mistakes)
        self._draw_3d_skeleton(ax1, actual_3d, error_joints)
        ax1.set_title('Your Execution\n(Errors Highlighted)', fontsize=14, fontweight='bold', color='#e74c3c')
        ax1.view_init(elev=20, azim=45)
        self._set_equal_aspect_3d(ax1, actual_3d)
        
        # Right: Intended shot (correct form)
        ax2 = fig.add_subplot(122, projection='3d')
        intended_3d = self._estimate_3d_from_2d(intended_shot_prototype)
        self._draw_3d_skeleton(ax2, intended_3d, {})  # No errors
        ax2.set_title('Correct Form\n(Target Position)', fontsize=14, fontweight='bold', color='#2ecc71')
        ax2.view_init(elev=20, azim=45)
        self._set_equal_aspect_3d(ax2, intended_3d)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_base64
    
    def generate_multi_angle_animation(self, keypoints: np.ndarray,
                                  mistakes: List[Dict]) -> str:
        """
        Generate rotating 3D animation (360° view)
        Returns: Base64-encoded GIF
        """
        keypoints_3d = self._estimate_3d_from_2d(keypoints)
        error_joints = self._get_error_joint_indices(mistakes)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            self._draw_3d_skeleton(ax, keypoints_3d, error_joints)
            ax.view_init(elev=20, azim=frame)
            self._set_equal_aspect_3d(ax, keypoints_3d)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=np.arange(0, 360, 3),
            interval=50
        )

        # ✅ SAVE TO TEMP FILE (REQUIRED)
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            gif_path = tmp.name

        anim.save(
            gif_path,
            writer=animation.PillowWriter(fps=20)
        )

        plt.close(fig)

        # ✅ READ → BASE64
        with open(gif_path, "rb") as f:
            gif_base64 = base64.b64encode(f.read()).decode("utf-8")

        # ✅ CLEANUP
        os.remove(gif_path)

        return gif_base64
    
    def _estimate_3d_from_2d(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Estimate 3D coordinates from 2D pose
        Uses biomechanical constraints
        """
        keypoints_3d = np.zeros((len(keypoints_2d), 3))
        
        # X, Z from 2D (frontal plane)
        keypoints_3d[:, 0] = keypoints_2d[:, 0]  # X (lateral)
        keypoints_3d[:, 2] = -keypoints_2d[:, 1]  # Z (vertical, flip Y)
        
        # Estimate Y (depth) based on body structure
        # Center at hips
        hip_center = (keypoints_2d[11] + keypoints_2d[12]) / 2
        
        for i in range(len(keypoints_2d)):
            # Head and shoulders: forward
            if i in [0, 1, 2, 3, 4, 5, 6]:
                keypoints_3d[i, 1] = 50
            # Elbows: slight forward
            elif i in [7, 8]:
                keypoints_3d[i, 1] = 30
            # Hands: further forward (bat contact plane)
            elif i in [9, 10]:
                keypoints_3d[i, 1] = 80
            # Hips: at center
            elif i in [11, 12]:
                keypoints_3d[i, 1] = 0
            # Legs: slightly back
            else:
                keypoints_3d[i, 1] = -20
        
        # Normalize
        keypoints_3d = keypoints_3d - np.mean(keypoints_3d, axis=0)
        
        return keypoints_3d
    
    def _draw_3d_skeleton(self, ax, keypoints_3d: np.ndarray, error_joints: Dict):
        """Draw 3D skeleton with color-coded errors"""
        
        # Draw connections (bones)
        for start_idx, end_idx in self.skeleton_lines:
            # Check if this connection involves an error joint
            is_error = (start_idx in error_joints or end_idx in error_joints)
            
            if is_error:
                severity = max(
                    error_joints.get(start_idx, ('normal', 0))[1],
                    error_joints.get(end_idx, ('normal', 0))[1]
                )
                color = self._get_severity_color(severity)
                linewidth = 4
                alpha = 1.0
            else:
                color = '#34495e'
                linewidth = 2
                alpha = 0.7
            
            ax.plot3D(
                [keypoints_3d[start_idx, 0], keypoints_3d[end_idx, 0]],
                [keypoints_3d[start_idx, 1], keypoints_3d[end_idx, 1]],
                [keypoints_3d[start_idx, 2], keypoints_3d[end_idx, 2]],
                color=color, linewidth=linewidth, alpha=alpha
            )
        
        # Draw joints (keypoints)
        for i, kpt in enumerate(keypoints_3d):
            if i in error_joints:
                severity = error_joints[i][1]
                color = self._get_severity_color(severity)
                size = 150
                marker = 'X'  # Error marker
            else:
                color = '#2ecc71'
                size = 80
                marker = 'o'
            
            ax.scatter(kpt[0], kpt[1], kpt[2], 
                      c=color, s=size, marker=marker,
                      edgecolors='white', linewidths=2, alpha=0.9)
        
        # Grid and background
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#ecf0f1')
    
    def _get_error_joint_indices(self, mistakes: List[Dict]) -> Dict:
        """
        Map mistakes to keypoint indices
        Returns: {keypoint_idx: (severity_name, severity_level)}
        """
        error_map = {}
        severity_levels = {'critical': 3, 'major': 2, 'minor': 1, 'negligible': 0}
        
        for mistake in mistakes:
            joint_id = mistake['joint_id']
            severity = mistake['severity']
            severity_level = severity_levels.get(severity, 0)
            
            if joint_id in self.joint_keypoints:
                for kpt_idx in self.joint_keypoints[joint_id]:
                    # Keep highest severity if multiple errors affect same joint
                    if kpt_idx not in error_map or error_map[kpt_idx][1] < severity_level:
                        error_map[kpt_idx] = (severity, severity_level)
        
        return error_map
    
    def _get_severity_color(self, severity_level: int) -> str:
        """Get color based on severity level"""
        if severity_level >= 3:
            return self.severity_colors['critical']
        elif severity_level >= 2:
            return self.severity_colors['major']
        elif severity_level >= 1:
            return self.severity_colors['minor']
        else:
            return self.severity_colors['normal']
    
    def _set_equal_aspect_3d(self, ax, keypoints_3d: np.ndarray):
        """Set equal aspect ratio for 3D plot"""
        max_range = np.array([
            keypoints_3d[:, 0].max() - keypoints_3d[:, 0].min(),
            keypoints_3d[:, 1].max() - keypoints_3d[:, 1].min(),
            keypoints_3d[:, 2].max() - keypoints_3d[:, 2].min()
        ]).max() / 2.0
        
        mid = np.mean(keypoints_3d, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)