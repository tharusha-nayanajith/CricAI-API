"""
Calibration Service for Camera Homography Calculation
Handles perspective transformation from image pixels to real-world coordinates
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel


class Point2D(BaseModel):
    """2D point in pixel or real-world coordinates"""
    x: float
    y: float


class StumpDetection(BaseModel):
    """Stump detection with bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    center_x: int
    center_y: int
    
    @property
    def bottom_center(self) -> Point2D:
        """Get bottom center point of the stump"""
        return Point2D(x=self.center_x, y=self.y2)
    
    @property
    def bottom_left(self) -> Point2D:
        """Get bottom left point of the stump"""
        return Point2D(x=self.x1, y=self.y2)
    
    @property
    def bottom_right(self) -> Point2D:
        """Get bottom right point of the stump"""
        return Point2D(x=self.x2, y=self.y2)


def yards_to_meters(yards: float) -> float:
    """
    Convert yards to meters
    
    Args:
        yards: Distance in yards
        
    Returns:
        Distance in meters (1 yard = 0.9144 meters)
    """
    return yards * 0.9144


def separate_near_far_stumps(stumps: List[StumpDetection]) -> Tuple[List[StumpDetection], List[StumpDetection]]:
    """
    Separate stumps into near (closer to camera) and far sets
    
    Near stumps have higher Y coordinates (bottom of image)
    Far stumps have lower Y coordinates (top of image)
    
    Args:
        stumps: List of all detected stumps
        
    Returns:
        Tuple of (near_stumps, far_stumps)
        
    Raises:
        ValueError: If not exactly 6 stumps detected
    """
    if len(stumps) != 6:
        raise ValueError(f"Expected 6 stumps for calibration, found {len(stumps)}")
    
    # Sort by Y coordinate (descending - higher Y = nearer to camera)
    sorted_stumps = sorted(stumps, key=lambda s: s.center_y, reverse=True)
    
    # Split into two groups of 3
    near_stumps = sorted_stumps[:3]
    far_stumps = sorted_stumps[3:]
    
    return near_stumps, far_stumps


def extract_stump_corners(stumps: List[StumpDetection]) -> Tuple[Point2D, Point2D]:
    """
    Extract left and right corner points from a set of stumps
    
    For a set of 3 stumps, find the leftmost and rightmost stumps
    and return their bottom corner positions
    
    Args:
        stumps: List of 3 stumps (near or far set)
        
    Returns:
        Tuple of (left_point, right_point) at bottom of stumps
        
    Raises:
        ValueError: If not exactly 3 stumps provided
    """
    if len(stumps) != 3:
        raise ValueError(f"Expected 3 stumps in set, found {len(stumps)}")
    
    # Sort by X coordinate to find leftmost and rightmost
    sorted_by_x = sorted(stumps, key=lambda s: s.center_x)
    
    leftmost = sorted_by_x[0]
    rightmost = sorted_by_x[-1]
    
    # Use bottom-left of leftmost and bottom-right of rightmost
    left_point = leftmost.bottom_left
    right_point = rightmost.bottom_right
    
    return left_point, right_point


def calculate_homography(
    near_stumps: List[StumpDetection],
    far_stumps: List[StumpDetection],
    pitch_length_yards: float
) -> np.ndarray:
    """
    Calculate homography matrix for perspective transformation
    
    Maps image pixel coordinates to real-world coordinates (meters)
    using detected stump positions and known pitch dimensions
    
    Args:
        near_stumps: List of 3 stumps closer to camera
        far_stumps: List of 3 stumps farther from camera
        pitch_length_yards: Length of cricket pitch in yards
        
    Returns:
        3x3 homography matrix as numpy array
        
    Raises:
        ValueError: If stump counts are incorrect or homography cannot be computed
    """
    # Convert pitch length to meters
    pitch_length_m = yards_to_meters(pitch_length_yards)
    
    # Cricket pitch standard width between outer stumps: ~3.04m (10 feet)
    # Using 3.04m as the distance between leftmost and rightmost stump centers
    pitch_width_m = 3.04
    
    # Extract corner points from stump sets
    near_left, near_right = extract_stump_corners(near_stumps)
    far_left, far_right = extract_stump_corners(far_stumps)
    
    # Define source points (pixel coordinates in image)
    src_points = np.array([
        [near_left.x, near_left.y],      # Near left
        [near_right.x, near_right.y],    # Near right
        [far_right.x, far_right.y],      # Far right
        [far_left.x, far_left.y]         # Far left
    ], dtype=np.float32)
    
    # Define destination points (real-world coordinates in meters)
    # Origin at center of near stumps, Y-axis pointing away from camera
    half_width = pitch_width_m / 2
    
    dst_points = np.array([
        [-half_width, 0],                    # Near left
        [half_width, 0],                     # Near right
        [half_width, pitch_length_m],        # Far right
        [-half_width, pitch_length_m]        # Far left
    ], dtype=np.float32)
    
    # Calculate homography matrix
    try:
        homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return homography_matrix
    except cv2.error as e:
        raise ValueError(f"Failed to calculate homography: {str(e)}")


def transform_point_to_real_world(
    pixel_point: Point2D,
    homography_matrix: np.ndarray
) -> Point2D:
    """
    Transform a pixel coordinate to real-world coordinate using homography
    
    Args:
        pixel_point: Point in image pixel coordinates
        homography_matrix: 3x3 homography matrix
        
    Returns:
        Point in real-world coordinates (meters)
    """
    # Convert point to homogeneous coordinates
    pixel_coords = np.array([[pixel_point.x, pixel_point.y]], dtype=np.float32)
    
    # Apply perspective transformation
    real_world_coords = cv2.perspectiveTransform(
        pixel_coords.reshape(-1, 1, 2),
        homography_matrix
    )
    
    # Extract transformed coordinates
    x, y = real_world_coords[0][0]
    
    return Point2D(x=float(x), y=float(y))


def serialize_matrix(matrix: np.ndarray) -> str:
    """
    Serialize numpy matrix to JSON string for database storage
    
    Args:
        matrix: Numpy array (typically 3x3 homography matrix)
        
    Returns:
        JSON string representation
    """
    import json
    return json.dumps(matrix.tolist())


def deserialize_matrix(matrix_json: str) -> np.ndarray:
    """
    Deserialize JSON string back to numpy matrix
    
    Args:
        matrix_json: JSON string representation of matrix
        
    Returns:
        Numpy array
    """
    import json
    matrix_list = json.loads(matrix_json)
    return np.array(matrix_list, dtype=np.float32)
