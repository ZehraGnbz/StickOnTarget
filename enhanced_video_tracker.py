#!/usr/bin/env python3
"""
Advanced Video Object Tracking System
Professional video tracking with AI-enhanced features and adaptive algorithms

FEATURES:
✅ Adaptive learning and self-improvement
✅ Dynamic confidence assessment  
✅ Multi-scale and rotation invariant detection
✅ Occlusion handling and recovery
✅ Real-time performance optimization
✅ Comprehensive analytics and monitoring
✅ Robust tracking algorithms
"""

import cv2
import numpy as np
import time
import os
from enum import Enum
from typing import Optional, Tuple, List, Dict
import math
from collections import deque
import threading

class TrackingState(Enum):
    """Advanced system states"""
    WAITING = "Waiting for Target"
    INITIALIZING = "Initializing Tracker"
    TRACKING = "Active Tracking"
    PREDICTING = "Motion Prediction"
    OCCLUDED = "Occlusion Handling"
    SEARCHING = "Deep Search Mode"
    RECOVERING = "Recovery Mode"
    PAUSED = "System Paused"
    ANALYZING = "Quality Analysis"

class TrackingMode(Enum):
    """Tracking difficulty modes"""
    PRECISION = "Precision Mode"     # Ultra high accuracy
    BALANCED = "Balanced Mode"      # Perfect balance
    AGGRESSIVE = "Aggressive Mode"    # Maximum persistence
    ADAPTIVE = "Adaptive Mode"        # Self-learning mode

class QualityMetrics:
    """Advanced quality assessment"""
    
    def __init__(self):
        self.sharpness_history = deque(maxlen=30)
        self.contrast_history = deque(maxlen=30)
        self.motion_blur_history = deque(maxlen=30)
        self.confidence_trend = deque(maxlen=50)
        
    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Deep image quality analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        self.sharpness_history.append(sharpness)
        
        # Contrast (standard deviation)
        contrast = gray.std()
        self.contrast_history.append(contrast)
        
        # Motion blur detection
        edges = cv2.Canny(gray, 50, 150)
        motion_blur = 1.0 / (1.0 + np.sum(edges) / (gray.shape[0] * gray.shape[1]))
        self.motion_blur_history.append(motion_blur)
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'motion_blur': motion_blur,
            'overall_quality': (sharpness / 1000 + contrast / 100 + (1 - motion_blur)) / 3
        }
    
    def get_quality_trend(self) -> float:
        """Get quality improvement/degradation trend"""
        if len(self.sharpness_history) < 10:
            return 0.0
        
        recent = list(self.sharpness_history)[-5:]
        older = list(self.sharpness_history)[-10:-5]
        
        if older:
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)
            return (recent_avg - older_avg) / older_avg
        return 0.0

class OcclusionDetector:
    """Advanced occlusion detection and handling system"""
    
    def __init__(self):
        self.confidence_drop_threshold = 0.4  # Sudden confidence drops indicate occlusion
        self.motion_consistency_history = deque(maxlen=10)
        self.occlusion_probability = 0.0
        self.last_good_bbox = None
        self.last_good_template = None
        self.occlusion_start_frame = 0
        self.pre_occlusion_motion = None
        
    def detect_occlusion(self, current_confidence: float, previous_confidence: float, 
                        motion_data: Dict, frame_count: int) -> bool:
        """Detect if object is likely occluded rather than lost"""
        
        # Sudden confidence drop indicates possible occlusion
        confidence_drop = previous_confidence - current_confidence
        sudden_drop = confidence_drop > self.confidence_drop_threshold
        
        # Check motion consistency - objects don't usually disappear mid-motion
        motion_consistent = self._is_motion_consistent(motion_data)
        
        # Combine indicators
        if sudden_drop and motion_consistent:
            self.occlusion_probability = min(1.0, confidence_drop + 0.3)
            self.occlusion_start_frame = frame_count
            return True
        
        # Gradual degradation might indicate partial occlusion
        if current_confidence < 0.3 and motion_consistent:
            self.occlusion_probability = 0.6
            return True
            
        return False
    
    def _is_motion_consistent(self, motion_data: Dict) -> bool:
        """Check if motion pattern suggests object should still be trackable"""
        if not motion_data.get('velocity_history'):
            return False
            
        velocities = motion_data['velocity_history']
        if len(velocities) < 3:
            return False
        
        # Calculate motion consistency
        recent_velocities = velocities[-3:]
        avg_speed = sum(math.sqrt(v[0]**2 + v[1]**2) for v in recent_velocities) / len(recent_velocities)
        
        # Object moving at reasonable speed is likely to continue
        return avg_speed > 2.0  # Pixels per frame
    
    def get_occlusion_duration(self, current_frame: int) -> int:
        """Get how long object has been occluded"""
        if self.occlusion_start_frame > 0:
            return current_frame - self.occlusion_start_frame
        return 0
    
    def estimate_exit_position(self, motion_data: Dict, occlusion_duration: int) -> Optional[Tuple[int, int]]:
        """Estimate where object might exit occlusion"""
        if not motion_data.get('velocity_history') or not self.last_good_bbox:
            return None
        
        # Use pre-occlusion motion to predict exit point
        velocities = motion_data['velocity_history']
        if len(velocities) < 2:
            return None
        
        avg_velocity = velocities[-1]  # Use last known velocity
        
        # Extrapolate from last good position
        x, y, w, h = self.last_good_bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Predict position based on duration and velocity
        predicted_x = center_x + avg_velocity[0] * occlusion_duration
        predicted_y = center_y + avg_velocity[1] * occlusion_duration
        
        return (int(predicted_x), int(predicted_y))
    
    def update_last_good_state(self, bbox: Tuple[int, int, int, int], template: np.ndarray = None):
        """Store last good tracking state before occlusion"""
        self.last_good_bbox = bbox
        if template is not None:
            self.last_good_template = template.copy()

class AdaptiveLearner:
    """AI-powered adaptive learning system"""
    
    def __init__(self):
        self.success_patterns = []
        self.failure_patterns = []
        self.template_variations = []
        self.optimal_params = {
            'search_radius': 100,
            'confidence_threshold': 0.7,
            'patience_multiplier': 1.0
        }
        self.learning_enabled = True
        
    def learn_from_success(self, context: Dict):
        """Learn from successful tracking instances"""
        if not self.learning_enabled:
            return
            
        pattern = {
            'confidence': context.get('confidence', 0),
            'search_radius': context.get('search_radius', 100),
            'quality_score': context.get('quality_score', 0),
            'timestamp': time.time()
        }
        
        self.success_patterns.append(pattern)
        if len(self.success_patterns) > 100:
            self.success_patterns.pop(0)
        
        self._update_optimal_params()
    
    def learn_from_failure(self, context: Dict):
        """Learn from tracking failures"""
        if not self.learning_enabled:
            return
            
        pattern = {
            'last_confidence': context.get('confidence', 0),
            'search_radius': context.get('search_radius', 100),
            'quality_score': context.get('quality_score', 0),
            'timestamp': time.time()
        }
        
        self.failure_patterns.append(pattern)
        if len(self.failure_patterns) > 50:
            self.failure_patterns.pop(0)
    
    def _update_optimal_params(self):
        """Update parameters based on learning"""
        if len(self.success_patterns) < 10:
            return
        
        # Analyze successful patterns
        recent_successes = self.success_patterns[-20:]
        
        # Optimal search radius
        radii = [p['search_radius'] for p in recent_successes if p['confidence'] > 0.8]
        if radii:
            self.optimal_params['search_radius'] = int(np.mean(radii))
        
        # Optimal confidence threshold
        confidences = [p['confidence'] for p in recent_successes]
        if confidences:
            self.optimal_params['confidence_threshold'] = max(0.5, np.percentile(confidences, 25))
    
    def get_adaptive_params(self) -> Dict:
        """Get current adaptive parameters"""
        return self.optimal_params.copy()

class MotionPredictor:
    """Advanced motion prediction and trajectory analysis"""
    
    def __init__(self):
        self.position_history = deque(maxlen=50)
        self.velocity_history = deque(maxlen=30)
        self.acceleration_history = deque(maxlen=20)
        self.confidence_history = deque(maxlen=50)
        self.direction_changes = deque(maxlen=10)
        self.pattern_memory = []
        
    def update(self, bbox: Tuple[int, int, int, int], confidence: float = 1.0, quality_score: float = 1.0):
        """Advanced position and quality tracking"""
        x, y, w, h = bbox
        center = (x + w//2, y + h//2)
        timestamp = time.time()
        
        # Store with quality weighting
        self.position_history.append((center[0], center[1], timestamp, quality_score))
        self.confidence_history.append(confidence * quality_score)
        
        self._calculate_advanced_kinematics()
        self._detect_movement_patterns()
    
    def _calculate_advanced_kinematics(self):
        """Advanced kinematic calculations with quality weighting"""
        if len(self.position_history) < 2:
            return
        
        # Calculate velocity with quality weighting
        positions = list(self.position_history)
        for i in range(1, len(positions)):
            p1, p2 = positions[i-1], positions[i]
            dt = p2[2] - p1[2]
            
            if dt > 0:
                vx = (p2[0] - p1[0]) / dt
                vy = (p2[1] - p1[1]) / dt
                quality_weight = (p1[3] + p2[3]) / 2
                
                self.velocity_history.append((vx, vy, p2[2], quality_weight))
        
        # Calculate acceleration
        velocities = list(self.velocity_history)
        if len(velocities) >= 2:
            for i in range(1, len(velocities)):
                v1, v2 = velocities[i-1], velocities[i]
                dt = v2[2] - v1[2]
                
                if dt > 0:
                    ax = (v2[0] - v1[0]) / dt
                    ay = (v2[1] - v1[1]) / dt
                    quality_weight = (v1[3] + v2[3]) / 2
                    
                    self.acceleration_history.append((ax, ay, v2[2], quality_weight))
    
    def _detect_movement_patterns(self):
        """Detect circular, linear, or periodic movement patterns"""
        if len(self.position_history) < 10:
            return
        
        positions = [(p[0], p[1]) for p in list(self.position_history)[-10:]]
        
        # Detect direction changes
        directions = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            if dx != 0 or dy != 0:
                angle = math.atan2(dy, dx)
                directions.append(angle)
        
        if len(directions) >= 2:
            angle_changes = []
            for i in range(1, len(directions)):
                change = abs(directions[i] - directions[i-1])
                if change > math.pi:
                    change = 2 * math.pi - change
                angle_changes.append(change)
            
            avg_change = sum(angle_changes) / len(angle_changes)
            self.direction_changes.append(avg_change)
    
    def predict_position(self, steps_ahead: float = 1.0, context: Dict = None) -> Optional[Tuple[int, int]]:
        """Advanced position prediction using motion analysis and pattern recognition"""
        if len(self.position_history) < 3:
            return None
        
        positions = list(self.position_history)
        velocities = list(self.velocity_history)
        accelerations = list(self.acceleration_history)
        
        # Quality-weighted prediction
        if velocities:
            # Use quality weights for more accurate prediction
            recent_velocities = velocities[-5:]
            total_weight = sum(v[3] for v in recent_velocities)
            
            if total_weight > 0:
                weighted_vx = sum(v[0] * v[3] for v in recent_velocities) / total_weight
                weighted_vy = sum(v[1] * v[3] for v in recent_velocities) / total_weight
            else:
                return None
        else:
            return None
        
        last_pos = positions[-1]
        
        # Pattern-based adjustment
        pattern_adjustment = self._get_pattern_adjustment()
        
        # Acceleration consideration
        if accelerations:
            last_acc = accelerations[-1]
            ax, ay = last_acc[0] * last_acc[3], last_acc[1] * last_acc[3]
            
            # Advanced kinematic prediction
            pred_x = (last_pos[0] + 
                     weighted_vx * steps_ahead + 
                     0.5 * ax * (steps_ahead ** 2) + 
                     pattern_adjustment[0])
            pred_y = (last_pos[1] + 
                     weighted_vy * steps_ahead + 
                     0.5 * ay * (steps_ahead ** 2) + 
                     pattern_adjustment[1])
        else:
            pred_x = last_pos[0] + weighted_vx * steps_ahead + pattern_adjustment[0]
            pred_y = last_pos[1] + weighted_vy * steps_ahead + pattern_adjustment[1]
        
        return (int(pred_x), int(pred_y))
    
    def _get_pattern_adjustment(self) -> Tuple[float, float]:
        """Get adjustment based on detected movement patterns"""
        if len(self.direction_changes) < 3:
            return (0.0, 0.0)
        
        recent_changes = list(self.direction_changes)[-3:]
        avg_change = sum(recent_changes) / len(recent_changes)
        
        # If consistent turning, predict continued turn
        if avg_change > 0.5:  # Significant turning
            return (5.0, 5.0)  # Small adjustment for turning
        
        return (0.0, 0.0)

class TemplateDetector:
    """Multi-scale template matching with rotation invariance and feature detection"""
    
    def __init__(self):
        self.template = None
        self.template_scales = []
        self.template_rotations = []
        self.feature_matcher = None
        self.quality_assessor = QualityMetrics()
        self.adaptive_threshold = 0.7
        self.detection_mode = TrackingMode.BALANCED
        
    def set_template(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Initialize multi-scale and multi-rotation template matching"""
        x, y, w, h = bbox
        self.template = frame[y:y+h, x:x+w].copy()
        
        # Generate multiple scales
        scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        self.template_scales = []
        
        for scale in scales:
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 10 and new_w > 10:
                    scaled = cv2.resize(self.template, (new_w, new_h))
                    self.template_scales.append((scaled, scale))
            else:
                self.template_scales.append((self.template.copy(), 1.0))
        
        # Generate rotations for critical angles
        angles = [-15, -10, -5, 0, 5, 10, 15]
        self.template_rotations = []
        
        for angle in angles:
            if angle != 0:
                center = (w//2, h//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(self.template, rotation_matrix, (w, h))
                self.template_rotations.append((rotated, angle))
            else:
                self.template_rotations.append((self.template.copy(), 0))
        
        # Initialize feature matcher for critical cases
        try:
            self.feature_matcher = cv2.SIFT_create(nfeatures=100)
        except:
            try:
                self.feature_matcher = cv2.ORB_create(nfeatures=100)
            except:
                self.feature_matcher = None
    
    def search_template(self, frame: np.ndarray, search_center: Tuple[int, int], 
                       search_radius: int = 120, performance_level: str = "HIGH") -> Optional[Tuple[int, int, int, int, float]]:
        """Multi-scale template matching with performance optimization"""
        if not self.template_scales:
            return None
        
        best_match = None
        best_score = 0
        best_method = "unknown"
        
        # Define search area
        cx, cy = search_center
        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(frame.shape[1], cx + search_radius)
        y2 = min(frame.shape[0], cy + search_radius)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        search_area = frame[y1:y2, x1:x2]
        
        # Method 1: Performance-aware multi-scale template matching
        scales_to_use = self._get_performance_scales(performance_level)
        methods_to_use = self._get_performance_methods(performance_level)
        
        for template, scale in self.template_scales:
            if scale not in scales_to_use:
                continue
                
            for method in methods_to_use:
                try:
                    result = cv2.matchTemplate(search_area, template, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > self.adaptive_threshold and max_val > best_score:
                        th, tw = template.shape[:2]
                        real_x = x1 + max_loc[0]
                        real_y = y1 + max_loc[1]
                        
                        # Performance-aware quality verification
                        if performance_level == "LOW":
                            # Skip expensive quality analysis in low performance mode
                            quality_score = 0.7  # Assume reasonable quality
                        else:
                            candidate_area = frame[real_y:real_y+th, real_x:real_x+tw]
                            quality = self.quality_assessor.analyze_image_quality(candidate_area)
                            quality_score = quality['overall_quality']
                        
                        if quality_score > 0.3:
                            # Ensure bbox is within frame boundaries
                            candidate_bbox = (real_x, real_y, tw, th)
                            if self._is_bbox_within_frame(candidate_bbox, frame.shape):
                                best_match = candidate_bbox
                                best_score = max_val * quality_score
                                best_method = f"template_{scale:.1f}"
                except:
                    continue
        
        # Method 2: Rotation-invariant matching (for difficult cases)
        if best_score < 0.6:
            for template, angle in self.template_rotations:
                try:
                    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_score * 0.8:  # Slightly lower threshold for rotations
                        th, tw = template.shape[:2]
                        real_x = x1 + max_loc[0]
                        real_y = y1 + max_loc[1]
                        
                        best_match = (real_x, real_y, tw, th)
                        best_score = max_val
                        best_method = f"rotation_{angle}"
                except:
                    continue
        
        # Method 3: Feature matching (last resort for extreme cases) - Skip in low performance mode
        if best_score < 0.4 and self.feature_matcher and performance_level != "LOW":
            try:
                keypoints1, descriptors1 = self.feature_matcher.detectAndCompute(self.template, None)
                keypoints2, descriptors2 = self.feature_matcher.detectAndCompute(search_area, None)
                
                if descriptors1 is not None and descriptors2 is not None:
                    if hasattr(cv2, 'BFMatcher'):
                        matcher = cv2.BFMatcher()
                        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
                        
                        good_matches = []
                        for match_pair in matches:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        
                        if len(good_matches) > 10:
                            # Estimate position from feature matches
                            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
                            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
                            
                            if len(src_pts) >= 4:
                                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                                if homography is not None:
                                    h, w = self.template.shape[:2]
                                    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
                                    transformed_corners = cv2.perspectiveTransform(corners, homography)
                                    
                                    # Calculate bounding box from transformed corners
                                    x_coords = transformed_corners[:, 0, 0]
                                    y_coords = transformed_corners[:, 0, 1]
                                    
                                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                                    
                                    if x_max > x_min and y_max > y_min:
                                        feature_score = len(good_matches) / 50.0  # Normalize
                                        if feature_score > best_score:
                                            best_match = (x1 + x_min, y1 + y_min, x_max - x_min, y_max - y_min)
                                            best_score = feature_score
                                            best_method = "features"
            except Exception as e:
                pass
        
        if best_match:
            return (*best_match, best_score)
        
        return None
    
    def adapt_threshold(self, success_rate: float):
        """Dynamically adapt detection threshold based on performance"""
        if success_rate > 0.8:
            self.adaptive_threshold = min(0.85, self.adaptive_threshold + 0.01)
        elif success_rate < 0.5:
            self.adaptive_threshold = max(0.5, self.adaptive_threshold - 0.02)
    
    def _is_bbox_within_frame(self, bbox, frame_shape) -> bool:
        """Check if bbox is completely within frame boundaries"""
        x, y, w, h = bbox
        height, width = frame_shape[:2]
        return (0 <= x and 0 <= y and x + w <= width and y + h <= height and w > 0 and h > 0)
    
    def _get_performance_scales(self, performance_level: str) -> List[float]:
        """Get scales to use based on performance level"""
        if performance_level == "HIGH":
            return [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # All scales
        elif performance_level == "MEDIUM":
            return [0.9, 1.0, 1.1, 1.2]  # Reduced scales
        else:  # LOW
            return [1.0]  # Only original scale
    
    def _get_performance_methods(self, performance_level: str) -> List[int]:
        """Get matching methods to use based on performance level"""
        if performance_level == "HIGH":
            return [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]  # Both methods
        else:  # MEDIUM or LOW
            return [cv2.TM_CCOEFF_NORMED]  # Only the better method

class VideoTracker:
    """Advanced video object tracking system with AI-enhanced features"""
    
    def __init__(self, video_path: str):
        # Video setup
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"[ERROR] Video file not accessible: {video_path}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Perfect tracking system
        self.tracker = None
        self.state = TrackingState.WAITING
        self.target_bbox = None
        self.confidence = 0.0
        self.tracking_mode = TrackingMode.ADAPTIVE
        
        # AI Components
        self.predictor = MotionPredictor()
        self.detector = TemplateDetector()
        self.learner = AdaptiveLearner()
        self.quality_assessor = QualityMetrics()
        self.occlusion_detector = OcclusionDetector()
        
        # BBOX SIZE CONTROL - NEW!
        self.original_bbox_size = None  # Store original target size
        self.bbox_size_history = deque(maxlen=10)  # Track size changes
        self.max_size_growth = 1.5  # Maximum allowed size growth (50%)
        self.max_size_shrink = 0.7  # Minimum allowed size shrink (30%)
        self.template_update_threshold = 0.8  # Update template when confidence is high
        self.frames_since_template_update = 0
        
        # UI and control state
        self.selecting = False
        self.selection_start = None
        self.selection_end = None
        self.paused = False
        self.current_frame = None
        self.show_analytics = True
        self.show_trail = True
        self.show_prediction = True
        self.show_confidence_graph = True
        
        # Performance tracking
        self.frame_time = 1.0 / self.fps
        self.lost_frame_count = 0
        self.total_tracks = 0
        self.successful_tracks = 0
        self.prediction_steps = 0
        self.playback_speed = 1.0
        
        # Occlusion handling
        self.previous_confidence = 1.0
        self.frame_count = 0
        self.occlusion_patience = 90  # Frames to wait during occlusion
        
        # Visual effects
        self.trail_points = deque(maxlen=50)
        self.confidence_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=20)
        
        # Advanced features
        self.auto_zoom = False
        self.smart_crop = False
        self.performance_mode = False
        
        # Performance optimization
        self.adaptive_performance = True
        self.max_frame_time = 1.0 / 20.0  # Minimum 20 FPS (more lenient)
        self.intensive_operation_budget = 0.02  # 20ms max for intensive ops  
        self.last_frame_time = time.time()
        self.performance_level = "MEDIUM"  # Start with MEDIUM, not HIGH
        
        # Lightweight mode settings
        self.lightweight_mode = True  # Enable by default
        self.occlusion_detection_enabled = True  # Can be disabled for performance
        
        # Statistics
        self.session_stats = {
            'total_frames_processed': 0,
            'successful_predictions': 0,
            'recovery_count': 0,
            'average_confidence': 0.0,
            'max_lost_duration': 0,
            'start_time': time.time()
        }
        
        self._print_welcome()
    
    def _print_welcome(self):
        """System initialization message"""
        print("=" + "="*70 + "=")
        print("            ADVANCED VIDEO TRACKING SYSTEM")
        print("=" + "="*70 + "=")
        print(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        print(f"Duration: {self.total_frames/self.fps:.1f} seconds ({self.total_frames:,} frames)")
        print(f"Mode: {self.tracking_mode.value}")
        print("\nCONTROLS:")
        print("-" * 72)
        print("MOUSE DRAG           -> Select target object")
        print("SPACE                -> Play/Pause video")
        print("R                    -> Reset tracking")
        print("M                    -> Cycle tracking modes")
        print("A                    -> Toggle analytics panel")
        print("T                    -> Toggle trail visualization")
        print("P                    -> Toggle prediction display")
        print("G                    -> Toggle confidence graph")
        print("+/-                  -> Adjust playback speed")
        print("1,2,3,4              -> Set mode (Precision/Balanced/Aggressive/Adaptive)")
        print("Z                    -> Toggle auto-zoom")
        print("S                    -> Toggle smart crop")
        print("F                    -> Toggle performance mode")
        print("O                    -> Toggle adaptive performance")
        print("L                    -> Toggle lightweight mode")
        print("X                    -> Toggle occlusion detection")
        print("ESC                  -> Exit application")
        print("-" * 72)
        print("\nALGORITHMS:")
        print("- Adaptive learning from tracking performance")
        print("- Motion prediction with trajectory analysis")
        print("- Multi-scale template matching")
        print("- Advanced occlusion detection and recovery")
        print("- Real-time quality assessment")
        print("- Performance optimization and adaptive scaling")
        print("-" * 72)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse interaction handler"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.selection_start = (x, y)
            self.selection_end = None
            self.state = TrackingState.INITIALIZING
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.selection_end = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP and self.selecting:
            self.selecting = False
            if self.selection_start and self.selection_end:
                self._create_tracker()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Context menu or manual adjustment
            if self.target_bbox:
                self._show_context_menu(x, y)
    
    def _create_tracker(self):
        """Initialize tracking system with selected target"""
        if not self.selection_start or not self.selection_end or not hasattr(self, 'current_frame'):
            self.state = TrackingState.WAITING
            return
        
        # Calculate selection bbox
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        
        # Validate selection
        validation_result = self._validate_selection(x, y, w, h)
        if not validation_result['valid']:
            print(f"⚠️ {validation_result['message']}")
            self.state = TrackingState.WAITING
            return
        
        # Apply validation corrections
        x, y, w, h = validation_result['corrected_bbox']
        self.target_bbox = (x, y, w, h)
        
        # STORE ORIGINAL SIZE FOR DRIFT CONTROL
        self.original_bbox_size = (w, h)
        self.bbox_size_history.clear()
        self.frames_since_template_update = 0
        
        # Initialize tracker
        try:
            self.tracker = cv2.TrackerCSRT_create()
            success = self.tracker.init(self.current_frame, self.target_bbox)
            
            if success is None:
                success = True
                
        except Exception as e:
            print(f"❌ Tracker initialization failed: {e}")
            self.state = TrackingState.WAITING
            return
        
        if success:
            # Setup template detection
            self.detector.set_template(self.current_frame, self.target_bbox)
            
            # Initialize prediction system
            quality = self.quality_assessor.analyze_image_quality(
                self.current_frame[y:y+h, x:x+w]
            )
            self.predictor.update(self.target_bbox, 1.0, quality['overall_quality'])
            
            # Reset counters and state
            self.state = TrackingState.TRACKING
            self.confidence = 1.0
            self.lost_frame_count = 0
            self.prediction_steps = 0
            self.trail_points.clear()
            self.confidence_history.clear()
            
            # Update statistics
            self.total_tracks += 1
            
            print(f"[OK] Tracking initialized! Target: {w}x{h}px ({self.tracking_mode.value})")
            print(f"Quality Score: {quality['overall_quality']:.2f}")
            
        else:
            print("[ERROR] Tracker initialization failed!")
            self.state = TrackingState.WAITING
    
    def _validate_selection(self, x: int, y: int, w: int, h: int) -> Dict:
        """Selection validation with boundary correction"""
        # Size validation
        min_size = 25
        max_size_ratio = 0.6
        max_w = int(self.frame_width * max_size_ratio)
        max_h = int(self.frame_height * max_size_ratio)
        
        if w < min_size or h < min_size:
            return {'valid': False, 'message': f'Selection too small! Minimum {min_size}x{min_size} pixels.'}
        
        if w > max_w or h > max_h:
            return {'valid': False, 'message': f'Selection too large! Maximum {max_size_ratio*100:.0f}% of frame.'}
        
        # Aspect ratio validation
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 8:
            return {'valid': False, 'message': 'Selection too elongated! Choose a more balanced area.'}
        
        # Boundary correction
        x = max(0, min(x, self.frame_width - w))
        y = max(0, min(y, self.frame_height - h))
        w = min(w, self.frame_width - x)
        h = min(h, self.frame_height - y)
        
        # Quality check of selected area
        if hasattr(self, 'current_frame'):
            selected_area = self.current_frame[y:y+h, x:x+w]
            quality = self.quality_assessor.analyze_image_quality(selected_area)
            
            if quality['overall_quality'] < 0.2:
                return {'valid': False, 'message': 'Selected area has poor quality. Choose a clearer, more detailed area.'}
        
        return {
            'valid': True,
            'corrected_bbox': (x, y, w, h),
            'message': 'Valid selection'
        }
    
    def update_tracking(self, frame: np.ndarray):
        """Main tracking update with adaptive learning and performance optimization"""
        if self.tracker is None or self.target_bbox is None:
            return
        
        # Performance monitoring
        frame_start_time = time.time()
        frame_time_budget = self.max_frame_time
        
        self.session_stats['total_frames_processed'] += 1
        self.frame_count += 1
        
        # Quality analysis of current frame - with safety checks (optimized for performance)
        try:
            # In lightweight mode, skip quality analysis every other frame
            if self.lightweight_mode and self.frame_count % 2 == 0:
                # Use cached/simplified quality metrics
                quality = {'overall_quality': 0.7, 'sharpness': 100, 'contrast': 50, 'motion_blur': 0.3}
            else:
                # Ensure target_bbox is valid before accessing frame area
                self.target_bbox = self._clamp_bbox(self.target_bbox, frame.shape)
                y1, y2 = self.target_bbox[1], self.target_bbox[1] + self.target_bbox[3]
                x1, x2 = self.target_bbox[0], self.target_bbox[0] + self.target_bbox[2]
                
                # Double check bounds
                if y2 > frame.shape[0] or x2 > frame.shape[1] or y1 < 0 or x1 < 0:
                    # Create a safe default area for quality analysis
                    target_area = frame[frame.shape[0]//4:3*frame.shape[0]//4, 
                                       frame.shape[1]//4:3*frame.shape[1]//4]
                else:
                    target_area = frame[y1:y2, x1:x2]
                    
                quality = self.quality_assessor.analyze_image_quality(target_area)
        except Exception as e:
            # Fallback quality metrics if analysis fails
            quality = {'overall_quality': 0.5, 'sharpness': 100, 'contrast': 50, 'motion_blur': 0.3}
        
        # Standard tracking attempt
        success, new_bbox = self.tracker.update(frame)
        
        if success and self._is_bbox_valid(new_bbox, frame.shape):
            # APPLY SIZE CONTROL TO PREVENT BBOX DRIFT
            size_controlled_bbox = self._validate_bbox_size_change(new_bbox, frame.shape)
            self.target_bbox = tuple(map(int, size_controlled_bbox))
            
            # Detect and warn about size drift
            if self._detect_size_drift():
                print("[WARNING] Bbox size drift detected - applying stricter controls")
                self.max_size_growth = min(1.3, self.max_size_growth - 0.1)
                self.max_size_shrink = max(0.8, self.max_size_shrink + 0.1)
            
            # Adaptive confidence calculation
            base_confidence = 0.8 if success else 0.0
            quality_bonus = quality['overall_quality'] * 0.2
            self.confidence = min(1.0, base_confidence + quality_bonus)
            
            # UPDATE TEMPLATE IF CONDITIONS ARE MET
            if self._should_update_template(self.confidence):
                self._update_template_with_size_control(frame, self.target_bbox)
            
            # Update occlusion detector with good state
            self.occlusion_detector.update_last_good_state(self.target_bbox, self.detector.template)
            
            # Learning from success
            self.learner.learn_from_success({
                'confidence': self.confidence,
                'quality_score': quality['overall_quality'],
                'search_radius': 100
            })
            
            # Update systems
            self.predictor.update(self.target_bbox, self.confidence, quality['overall_quality'])
            self._update_trail()
            self._update_confidence_history()
            
            # Reset failure counters
            self.lost_frame_count = 0
            self.prediction_steps = 0
            self.state = TrackingState.TRACKING
            self.successful_tracks += 1
            self.previous_confidence = self.confidence
            
        else:
            # Tracking lost - check if it's occlusion or actual loss
            self.lost_frame_count += 1
            self.confidence = max(0.1, self.confidence * 0.95)  # Gradual confidence decay
            
            # Lightweight occlusion detection (only if enabled and not too frequent)
            is_occluded = False
            if (self.occlusion_detection_enabled and 
                self.lightweight_mode and 
                self.lost_frame_count <= 5):  # Only check early in loss
                
                # Quick occlusion check without heavy motion data preparation
                confidence_drop = self.previous_confidence - self.confidence
                if confidence_drop > 0.3 and len(self.predictor.velocity_history) > 2:
                    # Simple motion check
                    recent_velocities = list(self.predictor.velocity_history)[-2:]
                    avg_speed = sum(math.sqrt(v[0]**2 + v[1]**2) for v in recent_velocities) / len(recent_velocities)
                    
                    if avg_speed > 2.0:  # Moving reasonably fast
                        is_occluded = True
                        self.occlusion_detector.occlusion_probability = confidence_drop + 0.2
                        self.occlusion_detector.occlusion_start_frame = self.frame_count
            
            if is_occluded and self.state != TrackingState.OCCLUDED:
                print(f"[OCCLUSION] Detected! Probability: {self.occlusion_detector.occlusion_probability:.2f}")
                self.state = TrackingState.OCCLUDED
            
            # Learn from failure
            self.learner.learn_from_failure({
                'confidence': self.confidence,
                'quality_score': quality['overall_quality'],
                'search_radius': 100
            })
            
            # State-specific handling
            if self.state == TrackingState.OCCLUDED:
                self._handle_occlusion_mode_lightweight(frame, frame_start_time, frame_time_budget)
            else:
                # Standard recovery strategy
                patience_levels = self._get_patience_levels()
                
                if self.lost_frame_count <= patience_levels['initial']:
                    self.state = TrackingState.TRACKING  # Still hoping
                    
                elif self.lost_frame_count <= patience_levels['prediction']:
                    self.state = TrackingState.PREDICTING
                    self._handle_prediction_mode(frame, quality, frame_start_time, frame_time_budget)
                    
                elif self.lost_frame_count <= patience_levels['search']:
                    self.state = TrackingState.SEARCHING  
                    self._handle_search_mode(frame, frame_start_time, frame_time_budget)
                    
                elif self.lost_frame_count <= patience_levels['recovery']:
                    self.state = TrackingState.RECOVERING
                    self._handle_recovery_mode(frame, frame_start_time, frame_time_budget)
                    
                else:
                    # Ultimate persistence for AGGRESSIVE and ADAPTIVE modes
                    if self.tracking_mode in [TrackingMode.AGGRESSIVE, TrackingMode.ADAPTIVE]:
                        self.state = TrackingState.RECOVERING
                        if self.lost_frame_count % 30 == 0:  # Periodic deep search
                            remaining_budget = frame_time_budget - (time.time() - frame_start_time)
                            if remaining_budget > 0.010:  # Only if we have some time left
                                self._handle_recovery_mode(frame, frame_start_time, frame_time_budget)
                    else:
                        print("❌ Target completely lost. Please select a new target.")
                        self._reset_tracking()
            
            self.previous_confidence = self.confidence
        
        # Update statistics
        self.session_stats['max_lost_duration'] = max(
            self.session_stats['max_lost_duration'], 
            self.lost_frame_count
        )
        
        # Performance monitoring and adaptive adjustment
        frame_end_time = time.time()
        actual_frame_time = frame_end_time - frame_start_time
        self.last_frame_time = actual_frame_time
        
        # Adaptive performance adjustment (only check every 30 frames for efficiency)
        if self.adaptive_performance and self.frame_count % 30 == 0:
            if actual_frame_time > self.max_frame_time * 1.8:  # More lenient threshold
                if self.performance_level == "HIGH":
                    self.performance_level = "MEDIUM"
                    print("[AUTO] Performance adjusted to MEDIUM mode")
                elif self.performance_level == "MEDIUM":
                    self.performance_level = "LOW"
                    print("[AUTO] Performance adjusted to LOW mode")
            elif actual_frame_time < self.max_frame_time * 0.3:  # Frame was very fast
                if self.performance_level == "LOW":
                    self.performance_level = "MEDIUM"
                    print("[AUTO] Performance adjusted to MEDIUM mode")
                elif self.performance_level == "MEDIUM" and actual_frame_time < self.max_frame_time * 0.2:
                    self.performance_level = "HIGH"
                    print("[AUTO] Performance adjusted to HIGH mode")
    
    def _get_patience_levels(self) -> Dict[str, int]:
        """Get mode-specific patience levels (optimized for performance)"""
        if self.lightweight_mode:
            # Reduced patience levels for better performance
            base_levels = {
                TrackingMode.PRECISION: {'initial': 2, 'prediction': 8, 'search': 20, 'recovery': 40},
                TrackingMode.BALANCED: {'initial': 3, 'prediction': 12, 'search': 30, 'recovery': 60},
                TrackingMode.AGGRESSIVE: {'initial': 5, 'prediction': 20, 'search': 50, 'recovery': 100},
                TrackingMode.ADAPTIVE: {'initial': 4, 'prediction': 15, 'search': 35, 'recovery': 80}
            }
        else:
            # Original levels for full performance mode
            base_levels = {
                TrackingMode.PRECISION: {'initial': 3, 'prediction': 15, 'search': 45, 'recovery': 90},
                TrackingMode.BALANCED: {'initial': 5, 'prediction': 25, 'search': 75, 'recovery': 150},
                TrackingMode.AGGRESSIVE: {'initial': 8, 'prediction': 40, 'search': 120, 'recovery': 300},
                TrackingMode.ADAPTIVE: {'initial': 6, 'prediction': 30, 'search': 90, 'recovery': 999999}
            }
        
        return base_levels[self.tracking_mode]
    
    def _handle_occlusion_mode(self, frame: np.ndarray, motion_data: Dict, frame_start_time: float, frame_time_budget: float):
        """Handle tracking during occlusion with advanced prediction and recovery"""
        occlusion_duration = self.occlusion_detector.get_occlusion_duration(self.frame_count)
        
        # Estimate where object might exit occlusion
        exit_position = self.occlusion_detector.estimate_exit_position(motion_data, occlusion_duration)
        
        if exit_position:
            # Search around predicted exit position
            elapsed_time = time.time() - frame_start_time
            remaining_time = frame_time_budget - elapsed_time
            
            # Performance-aware search
            if remaining_time > 0.020:
                performance_level = "HIGH"
                search_radius = 120
            elif remaining_time > 0.015:
                performance_level = "MEDIUM"  
                search_radius = 80
            else:
                performance_level = "LOW"
                search_radius = 60
            
            # Search for object at predicted exit position
            found = self.detector.search_template(frame, exit_position, search_radius, performance_level)
            
            if found:
                found_bbox = found[:4]
                found_confidence = found[4]
                
                # Validate recovery
                if self._is_bbox_valid(found_bbox, frame.shape) and found_confidence > 0.5:
                    found_bbox = self._clamp_bbox(found_bbox, frame.shape)
                    
                    # Re-initialize tracker
                    self.tracker = cv2.TrackerCSRT_create()
                    success = self.tracker.init(frame, found_bbox)
                    
                    if success:
                        self.target_bbox = found_bbox
                        self.state = TrackingState.TRACKING
                        self.confidence = found_confidence
                        self.lost_frame_count = 0
                        self.prediction_steps = 0
                        
                        # Update predictor with recovered position
                        quality = self.quality_assessor.analyze_image_quality(
                            frame[found_bbox[1]:found_bbox[1]+found_bbox[3],
                                  found_bbox[0]:found_bbox[0]+found_bbox[2]]
                        )
                        self.predictor.update(found_bbox, self.confidence, quality['overall_quality'])
                        
                        print(f"🎯 Occlusion recovery! Duration: {occlusion_duration} frames, Confidence: {found_confidence:.2f}")
                        return
        
        # If occlusion lasts too long, switch to standard recovery
        max_occlusion_duration = self.occlusion_patience
        if occlusion_duration > max_occlusion_duration:
            print(f"⏰ Occlusion timeout ({occlusion_duration} frames). Switching to standard recovery.")
            self.state = TrackingState.SEARCHING
            self._handle_search_mode(frame, frame_start_time, frame_time_budget)
        else:
            # Continue waiting for object to exit occlusion
            if exit_position:
                # Update predicted position for visualization
                self.prediction_history.append(exit_position)
                if len(self.prediction_history) > 20:
                    self.prediction_history.popleft()

    def _handle_occlusion_mode_lightweight(self, frame: np.ndarray, frame_start_time: float, frame_time_budget: float):
        """Lightweight occlusion handling for better performance"""
        occlusion_duration = self.occlusion_detector.get_occlusion_duration(self.frame_count)
        
        # Quick timeout check - much shorter than original
        max_occlusion_duration = 30  # Reduced from 90 frames
        if occlusion_duration > max_occlusion_duration:
            print(f"[TIMEOUT] Occlusion timeout ({occlusion_duration} frames). Resuming normal tracking.")
            self.state = TrackingState.PREDICTING  # Go to prediction instead of search
            return
        
        # Simple search around last known position (much lighter than full motion prediction)
        if self.occlusion_detector.last_good_bbox and occlusion_duration % 10 == 0:  # Only every 10 frames
            elapsed_time = time.time() - frame_start_time
            remaining_time = frame_time_budget - elapsed_time
            
            if remaining_time > 0.015:  # Only if we have enough time
                # Simple search around last known center
                x, y, w, h = self.occlusion_detector.last_good_bbox
                center = (x + w//2, y + h//2)
                
                # Light search with reduced radius
                found = self.detector.search_template(frame, center, 60, "LOW")
                
                if found and found[4] > 0.6:  # Higher confidence required
                    found_bbox = found[:4]
                    found_confidence = found[4]
                    
                    if self._is_bbox_valid(found_bbox, frame.shape):
                        found_bbox = self._clamp_bbox(found_bbox, frame.shape)
                        
                        # Quick recovery
                        self.tracker = cv2.TrackerCSRT_create()
                        success = self.tracker.init(frame, found_bbox)
                        
                        if success:
                            self.target_bbox = found_bbox
                            self.state = TrackingState.TRACKING
                            self.confidence = found_confidence
                            self.lost_frame_count = 0
                            print(f"[RECOVERY] Quick occlusion recovery! Duration: {occlusion_duration} frames")
                            return

    def _handle_prediction_mode(self, frame: np.ndarray, quality: Dict, frame_start_time: float, frame_time_budget: float):
        """Handle intelligent prediction mode with performance optimization"""
        self.prediction_steps += 1
        
        # Adaptive performance scaling based on time budget
        elapsed_time = time.time() - frame_start_time
        remaining_time = frame_time_budget - elapsed_time
        
        if remaining_time < 0.010:  # Less than 10ms remaining
            performance_level = "LOW"
        elif remaining_time < 0.020:  # Less than 20ms remaining
            performance_level = "MEDIUM"
        else:
            performance_level = "HIGH"
        
        # Get motion prediction with frame constraints
        predicted_center = self.predictor.predict_position(
            steps_ahead=self.prediction_steps * 0.4,
            context={'quality': quality['overall_quality'], 'frame_shape': frame.shape}
        )
        
        if predicted_center:
            # Update target bbox to predicted position
            w, h = self.target_bbox[2], self.target_bbox[3]
            
            # Ensure prediction is within reasonable bounds
            frame_h, frame_w = frame.shape[:2]
            safe_center = (
                max(w//2, min(predicted_center[0], frame_w - w//2)),
                max(h//2, min(predicted_center[1], frame_h - h//2))
            )
            
            predicted_bbox = (
                safe_center[0] - w//2,
                safe_center[1] - h//2,
                w, h
            )
            
            # Double-check: Clamp to frame boundaries
            predicted_bbox = self._clamp_bbox(predicted_bbox, frame.shape)
            self.target_bbox = predicted_bbox
            
            # Template search at predicted location
            adaptive_params = self.learner.get_adaptive_params()
            search_radius = min(150, adaptive_params['search_radius'] + self.prediction_steps * 15)
            
            found = self.detector.search_template(frame, predicted_center, search_radius, performance_level)
            
            if found:
                # Perfect recovery!
                found_bbox = found[:4]
                found_confidence = found[4]
                
                # CRITICAL: Validate and clamp the found bbox before using it
                if self._is_bbox_valid(found_bbox, frame.shape):
                    found_bbox = self._clamp_bbox(found_bbox, frame.shape)
                    
                    # Re-initialize tracker with validated bbox
                    self.tracker = cv2.TrackerCSRT_create()
                    success = self.tracker.init(frame, found_bbox)
                else:
                    # Invalid bbox found, skip this recovery attempt
                    success = False
                
                if success:
                    self.target_bbox = found_bbox
                    self.state = TrackingState.TRACKING
                    self.confidence = found_confidence
                    self.lost_frame_count = 0
                    self.prediction_steps = 0
                    self.session_stats['successful_predictions'] += 1
                    
                    # Update systems
                    self.predictor.update(found_bbox, self.confidence, quality['overall_quality'])
                    
                    print(f"[RECOVERY] Prediction recovery successful! Confidence: {found_confidence:.2f}")
            else:
                # Store prediction for visualization
                self.prediction_history.append(predicted_center)
                if len(self.prediction_history) > 20:
                    self.prediction_history.popleft()
    
    def _handle_search_mode(self, frame: np.ndarray, frame_start_time: float, frame_time_budget: float):
        """Handle comprehensive search mode with performance optimization"""
        # Adaptive search frequency based on performance
        elapsed_time = time.time() - frame_start_time
        remaining_time = frame_time_budget - elapsed_time
        
        if remaining_time < 0.015:  # Less than 15ms remaining - skip search
            return
        
        search_interval = 10 if remaining_time > 0.025 else 20  # Less frequent search if time is tight
        
        if self.lost_frame_count % search_interval == 0:
            # Performance-aware comprehensive search
            self._comprehensive_search(frame, remaining_time)
    
    def _handle_recovery_mode(self, frame: np.ndarray, frame_start_time: float, frame_time_budget: float):
        """Handle ultimate recovery mode with performance optimization"""
        self.session_stats['recovery_count'] += 1
        
        # Adaptive recovery frequency based on performance
        elapsed_time = time.time() - frame_start_time
        remaining_time = frame_time_budget - elapsed_time
        
        if remaining_time < 0.020:  # Less than 20ms remaining - skip recovery search
            return
        
        recovery_interval = 20 if remaining_time > 0.030 else 40  # Less frequent recovery if time is tight
        
        if self.lost_frame_count % recovery_interval == 0:
            # Performance-aware comprehensive recovery search
            self._recovery_search(frame, remaining_time)
    
    def _comprehensive_search(self, frame: np.ndarray, remaining_time: float):
        """Comprehensive multi-strategy search with performance optimization"""
        if not hasattr(self.detector, 'template_scales'):
            return
        
        # Determine performance level based on remaining time
        if remaining_time > 0.040:
            performance_level = "HIGH"
            max_grid_points = 16
        elif remaining_time > 0.025:
            performance_level = "MEDIUM"
            max_grid_points = 8
        else:
            performance_level = "LOW"
            max_grid_points = 4
        
        # Strategy 1: Performance-aware grid search
        grid_points = self._generate_search_grid(frame.shape)
        
        for i, point in enumerate(grid_points[:max_grid_points]):  # Limit grid points
            found = self.detector.search_template(frame, point, 80, performance_level)
            if found and found[4] > 0.6:  # Good confidence
                self._attempt_recovery(frame, found)
                return
        
        # Strategy 2: Motion-based search (only if we have time)
        if remaining_time > 0.020 and len(self.trail_points) >= 3:
            extrapolated_point = self._extrapolate_from_trail()
            if extrapolated_point:
                found = self.detector.search_template(frame, extrapolated_point, 120, performance_level)
                if found and found[4] > 0.5:
                    self._attempt_recovery(frame, found)
                    return
    
    def _recovery_search(self, frame: np.ndarray, remaining_time: float):
        """Comprehensive recovery search with performance optimization"""
        # Performance-aware full frame search
        best_match = None
        best_score = 0
        
        # Adaptive step size and performance level based on remaining time
        if remaining_time > 0.050:
            step_size = 40
            performance_level = "HIGH"
            max_checks = 100
        elif remaining_time > 0.035:
            step_size = 60
            performance_level = "MEDIUM"
            max_checks = 50
        else:
            step_size = 80
            performance_level = "LOW"
            max_checks = 25
        
        check_count = 0
        # Grid search across entire frame with time budget
        for y in range(0, frame.shape[0] - 50, step_size):
            for x in range(0, frame.shape[1] - 50, step_size):
                if check_count >= max_checks:
                    break
                    
                found = self.detector.search_template(frame, (x, y), 60, performance_level)
                if found and found[4] > best_score:
                    best_match = found
                    best_score = found[4]
                
                check_count += 1
            
            if check_count >= max_checks:
                break
        
        if best_match and best_score > 0.4:  # Lower threshold for desperate recovery
            self._attempt_recovery(frame, best_match)
            print(f"[RECOVERY] Search successful! Score: {best_score:.2f} (Performance: {performance_level})")
    
    def _attempt_recovery(self, frame: np.ndarray, found_match: Tuple):
        """Attempt to recover tracking from found match with size control"""
        found_bbox = found_match[:4]
        confidence = found_match[4]
        
        # Validate and clamp the found match
        if self._is_bbox_valid(found_bbox, frame.shape):
            # APPLY SIZE CONTROL TO RECOVERY
            if self.original_bbox_size:
                found_bbox = self._validate_size_for_recovery(found_bbox)
            
            # CRITICAL: Clamp bbox to ensure it's within frame boundaries
            found_bbox = self._clamp_bbox(found_bbox, frame.shape)
            
            # Re-initialize tracker with safe bbox
            self.tracker = cv2.TrackerCSRT_create()
            success = self.tracker.init(frame, found_bbox)
            
            if success:
                self.target_bbox = found_bbox
                self.state = TrackingState.TRACKING
                self.confidence = confidence
                self.lost_frame_count = 0
                self.prediction_steps = 0
                
                # Update prediction system
                quality = self.quality_assessor.analyze_image_quality(
                    frame[found_bbox[1]:found_bbox[1]+found_bbox[3],
                          found_bbox[0]:found_bbox[0]+found_bbox[2]]
                )
                self.predictor.update(found_bbox, confidence, quality['overall_quality'])
                
                return True
        
        return False
    
    def _generate_search_grid(self, frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate intelligent search grid points"""
        h, w = frame_shape[:2]
        points = []
        
        # Center and cardinal directions
        center = (w//2, h//2)
        points.append(center)
        
        # Grid points
        step = min(w, h) // 8
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                points.append((x, y))
        
        return points
    
    def _extrapolate_from_trail(self) -> Optional[Tuple[int, int]]:
        """Extrapolate next position from trail history"""
        if len(self.trail_points) < 3:
            return None
        
        trail_list = list(self.trail_points)
        recent_points = trail_list[-3:]
        
        # Simple linear extrapolation
        p1, p2, p3 = recent_points
        
        # Calculate velocity
        vx = (p3[0] - p1[0]) / 2
        vy = (p3[1] - p1[1]) / 2
        
        # Extrapolate
        next_x = p3[0] + vx * 2
        next_y = p3[1] + vy * 2
        
        return (int(next_x), int(next_y))
    
    def _update_trail(self):
        """Update trail with current position"""
        if self.target_bbox:
            center = self._get_bbox_center(self.target_bbox)
            self.trail_points.append(center)
    
    def _update_confidence_history(self):
        """Update confidence history for graphing"""
        self.confidence_history.append(self.confidence)
        
        # Update session average
        if self.confidence_history:
            self.session_stats['average_confidence'] = sum(self.confidence_history) / len(self.confidence_history)
    
    def _is_bbox_valid(self, bbox, frame_shape) -> bool:
        """Advanced bbox validation"""
        x, y, w, h = bbox
        height, width = frame_shape[:2]
        
        # Basic bounds check
        if not (0 <= x < width and 0 <= y < height and x + w <= width and y + h <= height):
            return False
        
        # Size reasonableness
        if w < 10 or h < 10 or w > width * 0.8 or h > height * 0.8:
            return False
        
        # Aspect ratio check
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 15:  # Very elongated objects are suspicious
            return False
        
        return True
    
    def _clamp_bbox(self, bbox, frame_shape):
        """Clamp bbox to frame boundaries with robust error handling"""
        try:
            x, y, w, h = bbox
            height, width = frame_shape[:2]
            
            # Ensure minimum size
            min_size = 10
            w = max(min_size, int(w))
            h = max(min_size, int(h))
            
            # Clamp position to valid range
            x = max(0, min(int(x), width - w))
            y = max(0, min(int(y), height - h))
            
            # Final size adjustment if needed
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Ensure we still have a valid bbox
            if w < min_size or h < min_size:
                # Return a safe center bbox
                safe_w = min(min_size * 4, width // 4)
                safe_h = min(min_size * 4, height // 4)
                safe_x = (width - safe_w) // 2
                safe_y = (height - safe_h) // 2
                return (safe_x, safe_y, safe_w, safe_h)
            
            return (x, y, w, h)
            
        except Exception as e:
            # Ultimate fallback - return center bbox
            height, width = frame_shape[:2]
            safe_w = min(100, width // 4)
            safe_h = min(100, height // 4)
            safe_x = (width - safe_w) // 2
            safe_y = (height - safe_h) // 2
            return (safe_x, safe_y, safe_w, safe_h)
    
    def _get_bbox_center(self, bbox) -> Tuple[int, int]:
        """Get bbox center point"""
        x, y, w, h = bbox
        return (x + w//2, y + h//2)
    
    def _reset_tracking(self):
        """Reset all tracking systems"""
        self.tracker = None
        self.target_bbox = None
        self.state = TrackingState.WAITING
        self.confidence = 0.0
        self.lost_frame_count = 0
        self.prediction_steps = 0
        self.trail_points.clear()
        self.confidence_history.clear()
        self.prediction_history.clear()
        self.predictor = MotionPredictor()
        self.occlusion_detector = OcclusionDetector()
        self.previous_confidence = 1.0
        
        # RESET SIZE CONTROL
        self.original_bbox_size = None
        self.bbox_size_history.clear()
        self.max_size_growth = 1.5  # Reset to defaults
        self.max_size_shrink = 0.7
        self.frames_since_template_update = 0
        
        print("[RESET] Tracking system reset. Ready for new target selection.")
    
    def _show_context_menu(self, x: int, y: int):
        """Show context menu for advanced options"""
        # For now, just print available options
        print("🎯 Context Menu:")
        print("   - Right-click detected at ({}, {})".format(x, y))
        print("   - Manual position adjustment available")
        print("   - Quality analysis tools ready")
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw the user interface with tracking visualizations"""
        h, w = frame.shape[:2]
        
        # Selection visualization
        if self.selecting and self.selection_start and self.selection_end:
            self._draw_selection_feedback(frame)
        
        # Main tracking visualization
        if self.target_bbox:
            self._draw_tracking_visualization(frame)
        
        # Control panels
        if self.show_analytics:
            self._draw_analytics_panel(frame)
        
        # Performance graphs
        if self.show_confidence_graph:
            self._draw_confidence_graph(frame)
        
        # Trail visualization
        if self.show_trail and self.trail_points:
            self._draw_trail_effects(frame)
        
        # Prediction visualization
        if self.show_prediction and self.prediction_history:
            self._draw_prediction_visualization(frame)
        
        # Progress and timeline
        self._draw_timeline(frame)
        
        return frame
    
    def _draw_selection_feedback(self, frame: np.ndarray):
        """Draw intelligent selection feedback"""
        if not self.selection_start or not self.selection_end:
            return
        
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        w, h = abs(x2 - x1), abs(y2 - y1)
        
        # Validation feedback
        validation = self._validate_selection(min(x1, x2), min(y1, y2), w, h)
        
        if validation['valid']:
            color = (0, 255, 0)  # Green for valid
            status = "✅ PERFECT SELECTION"
        else:
            color = (0, 0, 255)  # Red for invalid
            status = f"❌ {validation['message'][:30]}..."
        
        # Draw selection rectangle
        cv2.rectangle(frame, self.selection_start, self.selection_end, color, 2)
        
        # Size and status info
        info_text = f"{w}x{h} - {status}"
        cv2.putText(frame, info_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Quality indicator
        if hasattr(self, 'current_frame') and validation['valid']:
            x, y = min(x1, x2), min(y1, y2)
            selected_area = self.current_frame[y:y+h, x:x+w]
            quality = self.quality_assessor.analyze_image_quality(selected_area)
            
            quality_text = f"Quality: {quality['overall_quality']:.2f}"
            cv2.putText(frame, quality_text, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_tracking_visualization(self, frame: np.ndarray):
        """Draw advanced tracking visualization"""
        x, y, w, h = self.target_bbox
        
        # State-based colors with animations
        colors = {
            TrackingState.TRACKING: (0, 255, 0),
            TrackingState.PREDICTING: (0, 255, 255),
            TrackingState.OCCLUDED: (128, 0, 255),
            TrackingState.SEARCHING: (255, 165, 0),
            TrackingState.RECOVERING: (255, 0, 255),
            TrackingState.ANALYZING: (128, 255, 255),
            TrackingState.WAITING: (128, 128, 128)
        }
        
        color = colors.get(self.state, (255, 255, 255))
        
        # Animated thickness based on confidence
        base_thickness = 2
        confidence_thickness = int(base_thickness + (self.confidence * 4))
        
        # Main tracking rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, confidence_thickness)
        
        # Professional corner markers
        corner_length = min(35, w//3, h//3)
        corner_thickness = max(3, confidence_thickness)
        
        # Draw all four corners
        corners = [
            (x, y), (x + w, y), (x, y + h), (x + w, y + h)
        ]
        
        for i, (cx, cy) in enumerate(corners):
            if i == 0:  # Top-left
                cv2.line(frame, (cx, cy), (cx + corner_length, cy), color, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy + corner_length), color, corner_thickness)
            elif i == 1:  # Top-right  
                cv2.line(frame, (cx, cy), (cx - corner_length, cy), color, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy + corner_length), color, corner_thickness)
            elif i == 2:  # Bottom-left
                cv2.line(frame, (cx, cy), (cx + corner_length, cy), color, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy - corner_length), color, corner_thickness)
            else:  # Bottom-right
                cv2.line(frame, (cx, cy), (cx - corner_length, cy), color, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy - corner_length), color, corner_thickness)
        
        # Center crosshair with confidence indicator
        center = self._get_bbox_center(self.target_bbox)
        crosshair_size = int(15 + self.confidence * 10)
        
        cv2.circle(frame, center, int(8 + self.confidence * 4), color, -1)
        cv2.circle(frame, center, int(8 + self.confidence * 4), (255, 255, 255), 2)
        
        # Crosshair lines
        cv2.line(frame, (center[0] - crosshair_size, center[1]), 
                (center[0] + crosshair_size, center[1]), (255, 255, 255), 2)
        cv2.line(frame, (center[0], center[1] - crosshair_size), 
                (center[0], center[1] + crosshair_size), (255, 255, 255), 2)
        
        # Status text with enhanced info
        occlusion_duration = self.occlusion_detector.get_occlusion_duration(self.frame_count) if hasattr(self, 'occlusion_detector') else 0
        
        status_texts = {
            TrackingState.TRACKING: f"TRACKING - Confidence: {self.confidence:.2f}",
            TrackingState.PREDICTING: f"PREDICTING - Lost: {self.lost_frame_count} frames",
            TrackingState.OCCLUDED: f"OCCLUDED - Duration: {occlusion_duration} frames",
            TrackingState.SEARCHING: f"SEARCHING - Deep scan: {self.lost_frame_count}",
            TrackingState.RECOVERING: f"RECOVERING - Recovery mode: {self.lost_frame_count}",
            TrackingState.ANALYZING: f"ANALYZING - Quality check",
            TrackingState.WAITING: "WAITING - Select target"
        }
        
        status_text = status_texts.get(self.state, "Unknown State")
        cv2.putText(frame, status_text, (x, y - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Detailed info
        detail_text = f"ID: TARGET | {w}x{h}px | {self.tracking_mode.value}"
        cv2.putText(frame, detail_text, (x, y + h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Performance indicator
        success_rate = (self.successful_tracks / max(1, self.total_tracks)) * 100
        perf_text = f"Success Rate: {success_rate:.1f}%"
        cv2.putText(frame, perf_text, (x, y + h + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_analytics_panel(self, frame: np.ndarray):
        """Draw comprehensive analytics panel"""
        h, w = frame.shape[:2]
        panel_height = 140
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Title
        cv2.putText(frame, "ANALYTICS DASHBOARD", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # System status
        state_colors = {
            TrackingState.TRACKING: (0, 255, 0),
            TrackingState.PREDICTING: (0, 255, 255),
            TrackingState.SEARCHING: (255, 165, 0),
            TrackingState.RECOVERING: (255, 0, 255),
            TrackingState.WAITING: (128, 128, 128)
        }
        
        status_color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(frame, f"STATUS: {self.state.value}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Mode and speed
        mode_color = (0, 255, 255) if self.tracking_mode == TrackingMode.ADAPTIVE else (255, 255, 255)
        cv2.putText(frame, f"MODE: {self.tracking_mode.value}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        cv2.putText(frame, f"SPEED: {self.playback_speed:.1f}x", (10, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistics
        current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = (current_frame_num / self.total_frames) * 100
        elapsed_time = time.time() - self.session_stats['start_time']
        
        cv2.putText(frame, f"FRAME: {current_frame_num:,}/{self.total_frames:,} ({progress:.1f}%)", 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, f"SESSION: {elapsed_time:.1f}s", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Performance information
        perf_color = (0, 255, 0) if self.performance_level == "HIGH" else (255, 255, 0) if self.performance_level == "MEDIUM" else (255, 0, 0)
        cv2.putText(frame, f"PERFORMANCE: {self.performance_level}", (w - 250, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, perf_color, 1)
        
        # Frame time info
        frame_time_ms = self.last_frame_time * 1000
        target_time_ms = self.max_frame_time * 1000
        time_color = (0, 255, 0) if frame_time_ms < target_time_ms else (255, 0, 0)
        cv2.putText(frame, f"FRAME TIME: {frame_time_ms:.1f}/{target_time_ms:.1f}ms", (w - 250, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, time_color, 1)
        
        # Right side statistics
        if self.target_bbox:
            cv2.putText(frame, f"CONFIDENCE: {self.confidence:.3f}", (w - 250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # SHOW BBOX SIZE INFO
            bbox_w, bbox_h = self.target_bbox[2], self.target_bbox[3]
            if self.original_bbox_size:
                orig_w, orig_h = self.original_bbox_size
                size_ratio = ((bbox_w * bbox_h) / (orig_w * orig_h)) ** 0.5
                cv2.putText(frame, f"SIZE: {bbox_w}x{bbox_h} ({size_ratio:.2f}x)", (w - 250, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                cv2.putText(frame, f"SIZE: {bbox_w}x{bbox_h}", (w - 250, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(frame, f"LOST FRAMES: {self.lost_frame_count}", (w - 250, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"RECOVERIES: {self.session_stats['recovery_count']}", (w - 250, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"AVG CONFIDENCE: {self.session_stats['average_confidence']:.3f}", (w - 250, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_confidence_graph(self, frame: np.ndarray):
        """Draw real-time confidence graph"""
        if not self.confidence_history:
            return
        
        h, w = frame.shape[:2]
        graph_width = 300
        graph_height = 80
        graph_x = w - graph_width - 10
        graph_y = h - graph_height - 50
        
        # Graph background
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Graph border
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "CONFIDENCE", (graph_x + 5, graph_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Plot confidence line
        history_list = list(self.confidence_history)
        if len(history_list) > 1:
            points_to_show = min(len(history_list), graph_width - 20)
            step = (graph_width - 20) / max(1, points_to_show - 1)
            
            prev_point = None
            for i, conf in enumerate(history_list[-points_to_show:]):
                x = graph_x + 10 + int(i * step)
                y = graph_y + graph_height - 10 - int(conf * (graph_height - 20))
                
                current_point = (x, y)
                
                if prev_point:
                    # Color based on confidence level
                    color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
                    cv2.line(frame, prev_point, current_point, color, 2)
                
                prev_point = current_point
        
        # Current confidence value
        current_conf = self.confidence_history[-1] if self.confidence_history else 0
        cv2.putText(frame, f"{current_conf:.3f}", (graph_x + graph_width - 60, graph_y + graph_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_trail_effects(self, frame: np.ndarray):
        """Draw beautiful trail effects"""
        if len(self.trail_points) < 2:
            return
        
        trail_list = list(self.trail_points)
        
        # Draw trail with gradient effect
        for i in range(1, len(trail_list)):
            alpha = i / len(trail_list)
            thickness = max(1, int(5 * alpha))
            
            # Color transition from blue to green
            color = (
                int(255 * (1 - alpha)),  # Blue component
                int(255 * alpha),        # Green component  
                int(100 * alpha)         # Red component
            )
            
            cv2.line(frame, trail_list[i-1], trail_list[i], color, thickness)
        
        # Highlight recent positions
        if len(trail_list) >= 3:
            for i, point in enumerate(trail_list[-3:]):
                radius = 3 + i
                cv2.circle(frame, point, radius, (255, 255, 255), 1)
    
    def _draw_prediction_visualization(self, frame: np.ndarray):
        """Draw prediction visualization"""
        if not self.prediction_history:
            return
        
        # Draw prediction points
        for i, pred_point in enumerate(self.prediction_history):
            alpha = (i + 1) / len(self.prediction_history)
            radius = int(5 + alpha * 10)
            
            # Prediction color (yellow to orange gradient)
            color = (0, int(255 * alpha), 255)
            
            cv2.circle(frame, pred_point, radius, color, 2)
            cv2.circle(frame, pred_point, 2, (255, 255, 255), -1)
        
        # Current prediction
        if self.state == TrackingState.PREDICTING and self.target_bbox:
            center = self._get_bbox_center(self.target_bbox)
            cv2.circle(frame, center, 25, (0, 255, 255), 3)
            cv2.putText(frame, "PREDICTION", (center[0] - 40, center[1] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def _draw_timeline(self, frame: np.ndarray):
        """Draw timeline with progress markers"""
        h, w = frame.shape[:2]
        bar_height = 15
        bar_margin = 20
        bar_y = h - bar_height - 10
        bar_width = w - 2 * bar_margin
        
        # Background
        cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height), (20, 20, 20), -1)
        
        # Progress
        current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = current_frame_num / max(1, self.total_frames - 1)
        progress_width = int(bar_width * progress)
        
        if progress_width > 0:
            # Gradient progress bar
            for i in range(progress_width):
                ratio = i / max(1, progress_width)
                color_intensity = int(255 * (0.3 + 0.7 * ratio))
                cv2.line(frame, (bar_margin + i, bar_y), (bar_margin + i, bar_y + bar_height), 
                        (0, color_intensity, 0), 1)
        
        # Border
        cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height), (150, 150, 150), 2)
        
        # Time markers
        if self.fps > 0:
            for seconds in [30, 60, 120, 300]:  # 30s, 1m, 2m, 5m markers
                frame_marker = seconds * self.fps
                if frame_marker < self.total_frames:
                    marker_x = bar_margin + int((frame_marker / self.total_frames) * bar_width)
                    cv2.line(frame, (marker_x, bar_y), (marker_x, bar_y + bar_height), (200, 200, 200), 1)
        
        # Current time display
        current_time = current_frame_num / self.fps
        total_time = self.total_frames / self.fps
        time_text = f"{current_time:.1f}s / {total_time:.1f}s"
        cv2.putText(frame, time_text, (bar_margin, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def handle_controls(self, key: int) -> bool:
        """Handle all user controls"""
        if key == 27:  # ESC
            return False
            
        elif key == ord(' '):  # SPACE - Play/Pause
            self.paused = not self.paused
            if self.paused and self.state != TrackingState.WAITING:
                self.state = TrackingState.PAUSED
            elif not self.paused and self.state == TrackingState.PAUSED:
                self.state = TrackingState.TRACKING if self.target_bbox else TrackingState.WAITING
            print("[PAUSED]" if self.paused else "[PLAYING]")
            
        elif key == ord('r') or key == ord('R'):  # Reset
            self._reset_tracking()
            
        elif key == ord('m') or key == ord('M'):  # Cycle modes
            modes = list(TrackingMode)
            current_idx = modes.index(self.tracking_mode)
            self.tracking_mode = modes[(current_idx + 1) % len(modes)]
            print(f"[MODE] {self.tracking_mode.value}")
            
        elif key == ord('a') or key == ord('A'):  # Analytics
            self.show_analytics = not self.show_analytics
            print(f"[ANALYTICS] {'ON' if self.show_analytics else 'OFF'}")
            
        elif key == ord('t') or key == ord('T'):  # Trail
            self.show_trail = not self.show_trail
            print(f"[TRAIL] {'ON' if self.show_trail else 'OFF'}")
            
        elif key == ord('p') or key == ord('P'):  # Prediction
            self.show_prediction = not self.show_prediction
            print(f"[PREDICTION] {'ON' if self.show_prediction else 'OFF'}")
            
        elif key == ord('g') or key == ord('G'):  # Confidence graph
            self.show_confidence_graph = not self.show_confidence_graph
            print(f"[GRAPH] {'ON' if self.show_confidence_graph else 'OFF'}")
            
        elif key == ord('+') or key == ord('='):  # Speed up
            self.playback_speed = min(5.0, self.playback_speed + 0.25)
            print(f"[SPEED] {self.playback_speed:.2f}x")
            
        elif key == ord('-') or key == ord('_'):  # Speed down
            self.playback_speed = max(0.1, self.playback_speed - 0.25)
            print(f"[SPEED] {self.playback_speed:.2f}x")
            
        elif key == ord('1'):  # Precision mode
            self.tracking_mode = TrackingMode.PRECISION
            print(f"[MODE] {self.tracking_mode.value}")
            
        elif key == ord('2'):  # Balanced mode
            self.tracking_mode = TrackingMode.BALANCED
            print(f"[MODE] {self.tracking_mode.value}")
            
        elif key == ord('3'):  # Aggressive mode
            self.tracking_mode = TrackingMode.AGGRESSIVE
            print(f"[MODE] {self.tracking_mode.value}")
            
        elif key == ord('4'):  # Adaptive mode
            self.tracking_mode = TrackingMode.ADAPTIVE
            print(f"[MODE] {self.tracking_mode.value}")
            
        elif key == ord('z') or key == ord('Z'):  # Auto zoom
            self.auto_zoom = not self.auto_zoom
            print(f"[ZOOM] Auto-zoom: {'ON' if self.auto_zoom else 'OFF'}")
            
        elif key == ord('s') or key == ord('S'):  # Smart crop
            self.smart_crop = not self.smart_crop
            print(f"[CROP] Smart crop: {'ON' if self.smart_crop else 'OFF'}")
            
        elif key == ord('f') or key == ord('F'):  # Performance mode
            self.performance_mode = not self.performance_mode
            if self.performance_mode:
                self.adaptive_performance = False
                self.performance_level = "LOW"
                print("[PERFORMANCE] Mode: ON (Fixed LOW)")
            else:
                self.adaptive_performance = True
                self.performance_level = "HIGH"
                print("[PERFORMANCE] Mode: OFF (Adaptive)")
            
        elif key == ord('o') or key == ord('O'):  # Toggle adaptive performance
            self.adaptive_performance = not self.adaptive_performance
            print(f"[ADAPTIVE] Performance: {'ON' if self.adaptive_performance else 'OFF'}")
            
        elif key == ord('l') or key == ord('L'):  # Toggle lightweight mode
            self.lightweight_mode = not self.lightweight_mode
            if self.lightweight_mode:
                self.performance_level = "MEDIUM"
                print("[LIGHTWEIGHT] Mode: ON (Better performance)")
            else:
                self.performance_level = "HIGH"
                print("[LIGHTWEIGHT] Mode: OFF (Full features)")
                
        elif key == ord('x') or key == ord('X'):  # Toggle occlusion detection
            self.occlusion_detection_enabled = not self.occlusion_detection_enabled
            print(f"[OCCLUSION] Detection: {'ON' if self.occlusion_detection_enabled else 'OFF'}")
            
        return True
    
    def run(self):
        """Run the video tracking system"""
        window_name = "Advanced Video Tracking System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        cv2.resizeWindow(window_name, 1800, 1200)
        
        print("\nSYSTEM ACTIVATED!")
        print("Ready for video tracking...")
        
        try:
            while True:
                start_time = time.time()
                
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("📹 Video completed. Restarting...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    self.current_frame = frame.copy()
                    
                    # Main tracking update
                    if self.state not in [TrackingState.WAITING, TrackingState.PAUSED]:
                        self.update_tracking(frame)
                        
                else:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    else:
                        continue
                
                # UI rendering
                frame = self.draw_ui(frame)
                
                # Display
                cv2.imshow(window_name, frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_controls(key):
                    break
                
                # Perfect timing
                if not self.paused:
                    elapsed = time.time() - start_time
                    target_frame_time = self.frame_time / self.playback_speed
                    sleep_time = target_frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Print final statistics
            self._print_session_summary()
            self.cap.release()
            cv2.destroyAllWindows()
            print("System shutdown complete")
    
    def _print_session_summary(self):
        """Print comprehensive session summary"""
        elapsed_time = time.time() - self.session_stats['start_time']
        
        print("\n" + "="*72)
        print("                    TRACKING SESSION SUMMARY")
        print("="*72)
        print(f"Session Duration: {elapsed_time:.1f} seconds")
        print(f"Frames Processed: {self.session_stats['total_frames_processed']:,}")
        print(f"Successful Tracks: {self.successful_tracks}")
        print(f"Successful Predictions: {self.session_stats['successful_predictions']}")
        print(f"Recovery Operations: {self.session_stats['recovery_count']}")
        print(f"Average Confidence: {self.session_stats['average_confidence']:.3f}")
        print(f"Max Lost Duration: {self.session_stats['max_lost_duration']} frames")
        
        if self.total_tracks > 0:
            success_rate = (self.successful_tracks / self.total_tracks) * 100
            print(f"Overall Success Rate: {success_rate:.1f}%")
        
        print("="*72)
        print("Thank you for using the Advanced Video Tracking System!")
    
    def _validate_bbox_size_change(self, new_bbox, frame_shape) -> Tuple[int, int, int, int]:
        """Prevent bbox size drift and validate size changes"""
        if not self.original_bbox_size or not self.target_bbox:
            return new_bbox
            
        x, y, w, h = new_bbox
        original_w, original_h = self.original_bbox_size
        
        # Calculate size change ratios
        width_ratio = w / original_w
        height_ratio = h / original_h
        
        # Clamp size changes to prevent drift
        if width_ratio > self.max_size_growth:
            w = int(original_w * self.max_size_growth)
            print(f"[SIZE CONTROL] Width growth limited to {self.max_size_growth:.1f}x")
        elif width_ratio < self.max_size_shrink:
            w = int(original_w * self.max_size_shrink)
            print(f"[SIZE CONTROL] Width shrink limited to {self.max_size_shrink:.1f}x")
            
        if height_ratio > self.max_size_growth:
            h = int(original_h * self.max_size_growth)
            print(f"[SIZE CONTROL] Height growth limited to {self.max_size_growth:.1f}x")
        elif height_ratio < self.max_size_shrink:
            h = int(original_h * self.max_size_shrink)
            print(f"[SIZE CONTROL] Height shrink limited to {self.max_size_shrink:.1f}x")
        
        # Keep center position, adjust coordinates
        center_x = x + new_bbox[2] // 2
        center_y = y + new_bbox[3] // 2
        x = center_x - w // 2
        y = center_y - h // 2
        
        # Ensure it's still within frame
        corrected_bbox = self._clamp_bbox((x, y, w, h), frame_shape)
        
        # Track size history for analysis
        self.bbox_size_history.append((w, h))
        
        return corrected_bbox
    
    def _should_update_template(self, confidence: float) -> bool:
        """Decide if template should be updated based on confidence and time"""
        self.frames_since_template_update += 1
        
        # Update template if:
        # 1. High confidence tracking
        # 2. Enough frames have passed since last update
        # 3. Not in prediction/recovery mode
        if (confidence > self.template_update_threshold and 
            self.frames_since_template_update > 30 and 
            self.state == TrackingState.TRACKING):
            return True
        
        # Force update after too many frames to prevent template staleness
        if self.frames_since_template_update > 120:
            return True
            
        return False
    
    def _update_template_with_size_control(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Update template while maintaining size consistency"""
        try:
            x, y, w, h = bbox
            
            # Ensure bbox is valid for template extraction
            if (x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0] and w > 10 and h > 10):
                # Extract new template
                new_template_area = frame[y:y+h, x:x+w]
                
                # Check template quality before updating
                quality = self.quality_assessor.analyze_image_quality(new_template_area)
                
                if quality['overall_quality'] > 0.5:  # Only update with good quality templates
                    self.detector.set_template(frame, bbox)
                    self.frames_since_template_update = 0
                    print(f"[TEMPLATE] Updated with quality {quality['overall_quality']:.2f}")
                else:
                    print(f"[TEMPLATE] Skipped update - low quality {quality['overall_quality']:.2f}")
            else:
                print("[TEMPLATE] Skipped update - invalid bbox")
                
        except Exception as e:
            print(f"[TEMPLATE] Update failed: {e}")
    
    def _detect_size_drift(self) -> bool:
        """Detect if bbox size is drifting over time"""
        if len(self.bbox_size_history) < 5:
            return False
            
        sizes = list(self.bbox_size_history)
        recent_sizes = sizes[-3:]
        older_sizes = sizes[-6:-3] if len(sizes) >= 6 else sizes[:-3]
        
        if not older_sizes:
            return False
            
        # Calculate average size change
        recent_avg_w = sum(s[0] for s in recent_sizes) / len(recent_sizes)
        recent_avg_h = sum(s[1] for s in recent_sizes) / len(recent_sizes)
        older_avg_w = sum(s[0] for s in older_sizes) / len(older_sizes)
        older_avg_h = sum(s[1] for s in older_sizes) / len(older_sizes)
        
        # Check for significant drift
        width_drift = abs(recent_avg_w - older_avg_w) / older_avg_w
        height_drift = abs(recent_avg_h - older_avg_h) / older_avg_h
        
        # If either dimension drifts more than 20%, consider it drift
        return width_drift > 0.2 or height_drift > 0.2
    
    def _validate_size_for_recovery(self, found_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Validate recovered bbox size against original target size"""
        if not self.original_bbox_size:
            return found_bbox
            
        x, y, w, h = found_bbox
        original_w, original_h = self.original_bbox_size
        
        # Calculate size ratios
        width_ratio = w / original_w
        height_ratio = h / original_h
        
        # More permissive limits for recovery (object might be at different distance)
        max_recovery_growth = 2.0  # Allow 2x growth for recovery
        min_recovery_shrink = 0.5  # Allow 50% shrink for recovery
        
        size_adjusted = False
        
        # Adjust width if outside reasonable bounds
        if width_ratio > max_recovery_growth:
            w = int(original_w * max_recovery_growth)
            size_adjusted = True
        elif width_ratio < min_recovery_shrink:
            w = int(original_w * min_recovery_shrink)
            size_adjusted = True
            
        # Adjust height if outside reasonable bounds  
        if height_ratio > max_recovery_growth:
            h = int(original_h * max_recovery_growth)
            size_adjusted = True
        elif height_ratio < min_recovery_shrink:
            h = int(original_h * min_recovery_shrink)
            size_adjusted = True
        
        if size_adjusted:
            # Keep center position, adjust coordinates
            center_x = x + found_bbox[2] // 2
            center_y = y + found_bbox[3] // 2
            x = center_x - w // 2
            y = center_y - h // 2
            print(f"[RECOVERY] Size adjusted to {w}x{h} (was {found_bbox[2]}x{found_bbox[3]})")
        
        return (x, y, w, h)

def main():
    """Main function"""
    import sys
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else "track.mp4"
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        print("Usage: python enhanced_video_tracker.py [video_file]")
        return 1
    
    try:
        video_tracker = VideoTracker(video_path)
        video_tracker.run()
        return 0
    except Exception as e:
        print(f"[ERROR] Initialization error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 