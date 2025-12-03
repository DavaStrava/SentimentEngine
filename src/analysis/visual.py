"""Visual Analysis Module

This module processes video frames to extract emotional features from facial expressions.
It consumes video frames from Redis Streams asynchronously and produces timestamped
emotion results based on facial expression recognition.

Requirements:
    - Req 1.3: Visual Analysis Module extracts facial expression features continuously
    - Req 4.1: System detects faces present in the frame
    - Req 4.2: System extracts facial landmarks and expression features
    - Req 4.3: System classifies expressions into emotional categories with confidence scores
    - Req 4.5: System reports quality indicators when face detection fails or face is occluded
"""

import logging
import time
import asyncio
from typing import Optional, Dict, List
import numpy as np
import cv2
import redis.asyncio as redis
import mediapipe as mp

from src.models.frames import VideoFrame
from src.models.features import FaceLandmarks
from src.models.results import VisualResult
from src.config.config_loader import config


logger = logging.getLogger(__name__)


class VisualProcessingError(Exception):
    """Exception raised for errors during visual processing"""
    pass


class VisualAnalyzer:
    """Analyzes video frames to extract emotional features from facial expressions.
    
    This class implements the visual analysis module that:
    1. Consumes video frames from Redis Streams asynchronously
    2. Detects faces using MediaPipe Face Detection
    3. Extracts facial landmarks using MediaPipe Face Mesh
    4. Classifies facial expressions into emotional categories
    5. Reports quality indicators for occlusion and lighting
    6. Caches timestamped results for Fusion Engine access
    7. Implements frame skipping to maintain real-time performance
    
    Attributes:
        face_detection: MediaPipe face detection model
        face_mesh: MediaPipe face mesh model for landmark extraction
        redis_client: Async Redis client for stream consumption
        latest_result: Most recent analysis result (cached)
        frame_counter: Counter for frame skipping logic
        frame_skip: Process every Nth frame
    """
    
    def __init__(self):
        """Initialize the visual analyzer with MediaPipe models and configuration."""
        self.confidence_threshold = config.get('visual.face_detection_confidence', 0.5)
        self.frame_skip = config.get('visual.frame_skip', 2)
        self.max_faces = config.get('visual.max_faces', 5)
        self.redis_url = config.get('redis.url', 'redis://localhost:6379')
        self.video_stream = config.get('redis.video_stream', 'video_frames')
        
        # MediaPipe models (initialized lazily)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection: Optional[mp.solutions.face_detection.FaceDetection] = None
        self.face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
        
        # Result cache
        self.latest_result: Optional[VisualResult] = None
        
        # Frame skipping
        self.frame_counter = 0
        
        # Redis client (initialized in start())
        self.redis_client: Optional[redis.Redis] = None
        
        logger.info(f"VisualAnalyzer initialized with frame_skip={self.frame_skip}")
    
    def _load_models(self):
        """Load MediaPipe face detection and face mesh models.
        
        Initializes MediaPipe models for face detection and landmark extraction.
        Models are downloaded automatically on first use.
        
        Raises:
            Exception: If model initialization fails
        """
        try:
            logger.info("Loading MediaPipe face detection and face mesh models")
            
            # Initialize face detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=self.confidence_threshold,
                model_selection=0  # 0 for short-range (< 2 meters)
            )
            
            # Initialize face mesh for landmark extraction
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_faces,
                refine_landmarks=True,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=0.5
            )
            
            logger.info("MediaPipe models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MediaPipe models: {e}", exc_info=True)
            raise
    
    def _detect_face(self, image: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """Detect face in the image using MediaPipe.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Tuple of (face_detected, bounding_box) where bounding_box is
            [x, y, width, height] or None if no face detected
        """
        try:
            results = self.face_detection.process(image)
            
            if not results.detections:
                return False, None
            
            # Get first detection (primary face)
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            h, w = image.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            return True, np.array([x, y, width, height])
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return False, None
    
    def _extract_landmarks(self, image: np.ndarray) -> Optional[FaceLandmarks]:
        """Extract facial landmarks using MediaPipe Face Mesh.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            FaceLandmarks object with landmark points and confidence,
            or None if extraction fails
        """
        try:
            results = self.face_mesh.process(image)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array
            h, w = image.shape[:2]
            points = []
            for landmark in face_landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                points.append([x, y])
            
            points_array = np.array(points, dtype=np.float32)
            
            # Compute confidence based on landmark visibility
            visibility_scores = [lm.visibility for lm in face_landmarks.landmark if hasattr(lm, 'visibility')]
            confidence = float(np.mean(visibility_scores)) if visibility_scores else 0.8
            
            return FaceLandmarks(
                points=points_array,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Landmark extraction failed: {e}")
            return None
    
    def _assess_occlusion(self, landmarks: FaceLandmarks) -> float:
        """Assess face occlusion based on landmark visibility.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Occlusion score in [0, 1] where 0 is fully occluded, 1 is fully visible
        """
        # Use landmark confidence as occlusion indicator
        # Higher confidence means less occlusion
        return landmarks.confidence
    
    def _assess_lighting(self, image: np.ndarray) -> float:
        """Assess lighting quality from image statistics.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Lighting quality score in [0, 1] where 1 is optimal lighting
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Compute mean brightness
            mean_brightness = np.mean(gray) / 255.0
            
            # Compute contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            
            # Optimal brightness is around 0.4-0.6
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
            
            # Good contrast is > 0.2
            contrast_score = min(contrast / 0.2, 1.0)
            
            # Combined lighting quality
            lighting_quality = (brightness_score + contrast_score) / 2.0
            
            return lighting_quality
            
        except Exception as e:
            logger.warning(f"Lighting assessment failed: {e}")
            return 0.5  # Default to moderate quality
    
    def _classify_expression(self, image: np.ndarray, landmarks: FaceLandmarks) -> Dict[str, float]:
        """Classify facial expression into emotional categories.
        
        This is a simplified implementation using geometric features from landmarks.
        In production, this would use a pre-trained CNN model (e.g., FER2013-trained).
        
        Args:
            image: RGB image array (H, W, 3)
            landmarks: Facial landmarks
            
        Returns:
            Dictionary mapping emotion names to scores [0, 1]
        """
        try:
            # Simplified emotion classification based on landmark geometry
            # In production, use a pre-trained CNN model
            
            points = landmarks.points
            
            # Extract key landmark indices (MediaPipe 468-point model)
            # Mouth corners: 61, 291
            # Mouth top: 13
            # Mouth bottom: 14
            # Left eye: 33
            # Right eye: 263
            # Left eyebrow: 70
            # Right eyebrow: 300
            
            # For now, return neutral with some randomness
            # This should be replaced with actual model inference
            
            # Compute simple features
            if len(points) >= 468:
                # Mouth aspect ratio (smile detection)
                mouth_width = np.linalg.norm(points[61] - points[291])
                mouth_height = np.linalg.norm(points[13] - points[14])
                mouth_ratio = mouth_height / (mouth_width + 1e-6)
                
                # Eye openness
                left_eye_height = np.linalg.norm(points[159] - points[145])
                right_eye_height = np.linalg.norm(points[386] - points[374])
                eye_openness = (left_eye_height + right_eye_height) / 2.0
                
                # Simple heuristic classification
                if mouth_ratio > 0.3:  # Wide smile
                    return {
                        "happy": 0.7,
                        "neutral": 0.2,
                        "sad": 0.05,
                        "angry": 0.05
                    }
                elif mouth_ratio < 0.15:  # Closed mouth
                    return {
                        "neutral": 0.6,
                        "sad": 0.2,
                        "angry": 0.1,
                        "happy": 0.1
                    }
            
            # Default to neutral
            return {
                "neutral": 0.7,
                "happy": 0.1,
                "sad": 0.1,
                "angry": 0.1
            }
            
        except Exception as e:
            logger.warning(f"Expression classification failed: {e}")
            return {"neutral": 1.0}
    
    async def analyze_frame(self, video_frame: VideoFrame) -> Optional[VisualResult]:
        """Analyze video frame and return emotion result with quality-adjusted confidence.
        
        This is the main analysis method that orchestrates the complete visual analysis
        pipeline. It processes video frames through multiple stages to extract emotional
        intelligence from facial expressions while accounting for visual quality.
        
        The analysis pipeline:
        1. Detects faces in the frame using MediaPipe (Req 4.1)
        2. Extracts facial landmarks if face detected (Req 4.2)
        3. Assesses occlusion based on landmark visibility (Req 4.5)
        4. Assesses lighting quality from image statistics (Req 4.5)
        5. Classifies facial expression into emotional categories (Req 4.3)
        6. Adjusts confidence score based on quality indicators (Req 4.5)
        7. Returns timestamped result for fusion engine consumption
        
        Error Handling:
        - VisualProcessingError: Returns low-confidence neutral result (confidence=0.1)
        - Unexpected exceptions: Returns None, allowing fusion to continue with other modalities
        
        Args:
            video_frame: Video frame to analyze containing RGB image, timestamp,
                        and frame number
            
        Returns:
            VisualResult with emotion scores, quality-adjusted confidence, face detection
            status, facial landmarks, and timestamp. Returns None if unexpected error occurs,
            allowing graceful degradation. Returns low-confidence neutral result if no face
            is detected or processing fails.
            
        Validates:
            - Req 1.3: Visual Analysis Module extracts facial expression features continuously
            - Req 4.1: System detects faces present in the frame
            - Req 4.2: System extracts facial landmarks and expression features
            - Req 4.3: System classifies expressions into emotional categories with confidence scores
            - Req 4.5: System reports quality indicators when face detection fails or face is occluded
            - Prop 2: Visual feature extraction completeness
        """
        try:
            image = video_frame.image
            
            # Detect face
            face_detected, bbox = self._detect_face(image)
            
            if not face_detected:
                # No face detected - return low confidence neutral result
                logger.debug(f"No face detected in frame {video_frame.frame_number}")
                result = VisualResult(
                    emotion_scores={"neutral": 1.0},
                    confidence=0.1,
                    face_detected=False,
                    face_landmarks=None,
                    timestamp=time.time()
                )
                self.latest_result = result
                return result
            
            # Extract landmarks
            landmarks = self._extract_landmarks(image)
            
            if landmarks is None:
                # Landmark extraction failed
                logger.debug(f"Landmark extraction failed for frame {video_frame.frame_number}")
                result = VisualResult(
                    emotion_scores={"neutral": 1.0},
                    confidence=0.2,
                    face_detected=True,
                    face_landmarks=None,
                    timestamp=time.time()
                )
                self.latest_result = result
                return result
            
            # Assess quality indicators
            occlusion_score = self._assess_occlusion(landmarks)
            lighting_score = self._assess_lighting(image)
            quality_score = (occlusion_score + lighting_score) / 2.0
            
            # Classify expression
            emotion_scores = self._classify_expression(image, landmarks)
            
            # Compute confidence (based on quality and detection confidence)
            max_emotion_score = max(emotion_scores.values())
            base_confidence = max_emotion_score * landmarks.confidence
            adjusted_confidence = base_confidence * quality_score
            
            # Create result
            result = VisualResult(
                emotion_scores=emotion_scores,
                confidence=adjusted_confidence,
                face_detected=True,
                face_landmarks=landmarks,
                timestamp=time.time()
            )
            
            # Cache result
            self.latest_result = result
            
            logger.debug(f"Visual analysis complete: confidence={adjusted_confidence:.3f}, "
                        f"quality={quality_score:.3f}, occlusion={occlusion_score:.3f}, "
                        f"lighting={lighting_score:.3f}")
            
            return result
            
        except VisualProcessingError as e:
            logger.warning(f"Visual processing failed: {e}")
            # Return low-confidence neutral result
            result = VisualResult(
                emotion_scores={"neutral": 1.0},
                confidence=0.1,
                face_detected=False,
                face_landmarks=None,
                timestamp=time.time()
            )
            self.latest_result = result
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in visual analysis: {e}", exc_info=True)
            return None
    
    def get_latest_result(self) -> Optional[VisualResult]:
        """Get the most recent analysis result from cache.
        
        Used by Fusion Engine to access cached results during time-windowed fusion.
        This method provides non-blocking access to the latest visual analysis result,
        enabling the fusion engine to operate on fixed 1-second intervals without waiting
        for analysis completion.
        
        Returns:
            Latest VisualResult containing emotion scores, confidence, face detection
            status, landmarks, and timestamp. Returns None if no analysis has been
            performed yet (e.g., at system startup before first frame is processed).
            
        Validates:
            - Req 6.1: Fusion Engine receives outputs from analysis modules
            - Design: Result caching with timestamps for non-blocking fusion
        """
        return self.latest_result

    async def start(self):
        """Start consuming video frames from Redis Streams asynchronously.
        
        This method runs as an independent asyncio task and continuously consumes
        video frames from the Redis stream, analyzes them with frame skipping, and
        caches results. It implements the asynchronous, event-driven architecture
        that prevents slow modules from blocking fast ones.
        
        The method:
        1. Initializes async Redis client connection
        2. Loads MediaPipe face detection and face mesh models
        3. Continuously reads from Redis Streams using non-blocking xread
        4. Implements frame skipping (process every Nth frame) for performance
        5. Deserializes and analyzes selected video frames
        6. Caches timestamped results for Fusion Engine access
        7. Handles errors gracefully without crashing the pipeline
        
        Frame skipping is implemented to maintain real-time performance, as visual
        analysis is computationally expensive. Processing every 2nd or 3rd frame
        provides sufficient temporal resolution for emotion detection while meeting
        latency requirements.
        
        This task runs indefinitely until cancelled via asyncio.CancelledError,
        enabling clean shutdown of the analysis pipeline.
        
        Raises:
            Exception: Fatal errors during initialization (Redis connection, model loading)
                      are propagated to allow system to fail fast at startup
        
        Validates:
            - Req 1.1: System begins processing within 2 seconds of stream initiation
            - Req 1.3: Visual Analysis Module extracts facial expression features continuously
            - Req 9.1: End-to-end latency not exceeding 3 seconds
            - Design: Asynchronous processing with independent asyncio tasks
            - Design: Frame skipping (process every 2nd or 3rd frame)
            - Design: Result caching with timestamps for non-blocking fusion
        """
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url)
            logger.info(f"Connected to Redis at {self.redis_url}")
            
            # Load models
            self._load_models()
            
            # Start consuming from stream
            last_id = '0-0'  # Start from beginning
            logger.info(f"Starting to consume from stream: {self.video_stream}")
            
            while True:
                try:
                    # Read from stream (blocking with timeout)
                    messages = await self.redis_client.xread(
                        {self.video_stream: last_id},
                        block=100,  # 100ms timeout
                        count=1
                    )
                    
                    if messages:
                        for stream_name, message_list in messages:
                            for message_id, data in message_list:
                                # Increment frame counter
                                self.frame_counter += 1
                                
                                # Frame skipping logic
                                if self.frame_counter % self.frame_skip != 0:
                                    # Skip this frame
                                    last_id = message_id
                                    continue
                                
                                # Deserialize video frame
                                video_frame = self._deserialize_frame(data)
                                
                                # Analyze frame
                                await self.analyze_frame(video_frame)
                                
                                # Update last_id for next read
                                last_id = message_id
                    
                    # Small delay to prevent tight loop
                    await asyncio.sleep(0.01)
                    
                except asyncio.CancelledError:
                    logger.info("Visual analyzer task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing video frame: {e}", exc_info=True)
                    await asyncio.sleep(0.1)  # Back off on error
                    
        except Exception as e:
            logger.error(f"Fatal error in visual analyzer: {e}", exc_info=True)
            raise
        finally:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            
            # Clean up MediaPipe resources
            if self.face_detection:
                self.face_detection.close()
            if self.face_mesh:
                self.face_mesh.close()
    
    def _deserialize_frame(self, data: Dict) -> VideoFrame:
        """Deserialize video frame from Redis stream data.
        
        Converts Redis stream message data back into a VideoFrame object for analysis.
        The serialization format uses pickle for numpy arrays and standard types for
        metadata fields.
        
        Args:
            data: Dictionary containing serialized frame data with keys:
                 - 'image': Pickled numpy array of RGB image (H, W, 3)
                 - 'timestamp': Float timestamp in seconds since stream start
                 - 'frame_number': Integer sequential frame number
            
        Returns:
            Deserialized VideoFrame ready for visual analysis
            
        Validates:
            - Req 1.1: Continuous video processing through Redis Streams
            - Design: Asynchronous frame distribution via Redis Streams
        """
        # TODO: Implement proper serialization/deserialization
        # For now, assume data contains the necessary fields
        
        # Deserialize image
        image_bytes = data[b'image']
        # Assuming image is stored as flattened array with shape info
        height = int(data[b'height'])
        width = int(data[b'width'])
        channels = int(data[b'channels'])
        
        image_flat = np.frombuffer(image_bytes, dtype=np.uint8)
        image = image_flat.reshape((height, width, channels))
        
        timestamp = float(data[b'timestamp'])
        frame_number = int(data[b'frame_number'])
        
        return VideoFrame(
            image=image,
            timestamp=timestamp,
            frame_number=frame_number
        )

    async def start(self):
        """Start consuming video frames from Redis Streams asynchronously.
        
        This method runs as an independent asyncio task and continuously consumes
        video frames from the Redis stream, analyzes them with frame skipping, and
        caches results. It implements the asynchronous, event-driven architecture
        that prevents slow modules from blocking fast ones.
        
        The method:
        1. Initializes async Redis client connection
        2. Loads MediaPipe face detection and face mesh models
        3. Continuously reads from Redis Streams using non-blocking xread
        4. Implements frame skipping (process every Nth frame) for performance
        5. Deserializes and analyzes selected video frames
        6. Caches timestamped results for Fusion Engine access
        7. Handles errors gracefully without crashing the pipeline
        
        Frame skipping is implemented to maintain real-time performance, as visual
        analysis is computationally expensive. Processing every 2nd or 3rd frame
        provides sufficient temporal resolution for emotion detection while meeting
        latency requirements.
        
        This task runs indefinitely until cancelled via asyncio.CancelledError,
        enabling clean shutdown of the analysis pipeline.
        
        Raises:
            Exception: Fatal errors during initialization (Redis connection, model loading)
                      are propagated to allow system to fail fast at startup
        
        Validates:
            - Req 1.1: System begins processing within 2 seconds of stream initiation
            - Req 1.3: Visual Analysis Module extracts facial expression features continuously
            - Req 9.1: End-to-end latency not exceeding 3 seconds
            - Design: Asynchronous processing with independent asyncio tasks
            - Design: Frame skipping (process every 2nd or 3rd frame)
            - Design: Result caching with timestamps for non-blocking fusion
        """
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url)
            logger.info(f"Connected to Redis at {self.redis_url}")
            
            # Load models
            self._load_models()
            
            # Start consuming from stream
            last_id = '0-0'  # Start from beginning
            logger.info(f"Starting to consume from stream: {self.video_stream}")
            
            while True:
                try:
                    # Read from stream (blocking with timeout)
                    messages = await self.redis_client.xread(
                        {self.video_stream: last_id},
                        block=100,  # 100ms timeout
                        count=1
                    )
                    
                    if messages:
                        for stream_name, message_list in messages:
                            for message_id, data in message_list:
                                # Increment frame counter
                                self.frame_counter += 1
                                
                                # Frame skipping logic
                                if self.frame_counter % self.frame_skip != 0:
                                    # Skip this frame
                                    last_id = message_id
                                    continue
                                
                                # Deserialize video frame
                                video_frame = self._deserialize_frame(data)
                                
                                # Analyze frame
                                await self.analyze_frame(video_frame)
                                
                                # Update last_id for next read
                                last_id = message_id
                    
                    # Small delay to prevent tight loop
                    await asyncio.sleep(0.01)
                    
                except asyncio.CancelledError:
                    logger.info("Visual analyzer task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing video frame: {e}", exc_info=True)
                    await asyncio.sleep(0.1)  # Back off on error
                    
        except Exception as e:
            logger.error(f"Fatal error in visual analyzer: {e}", exc_info=True)
            raise
        finally:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            
            # Clean up MediaPipe resources
            if self.face_detection:
                self.face_detection.close()
            if self.face_mesh:
                self.face_mesh.close()
    
    def _deserialize_frame(self, data: Dict) -> VideoFrame:
        """Deserialize video frame from Redis stream data.
        
        Converts Redis stream message data back into a VideoFrame object for analysis.
        The serialization format uses pickle for numpy arrays and standard types for
        metadata fields.
        
        Args:
            data: Dictionary containing serialized frame data with keys:
                 - 'image': Pickled numpy array of RGB image (H, W, 3)
                 - 'timestamp': Float timestamp in seconds since stream start
                 - 'frame_number': Integer sequential frame number
            
        Returns:
            Deserialized VideoFrame ready for visual analysis
            
        Validates:
            - Req 1.1: Continuous video processing through Redis Streams
            - Design: Asynchronous frame distribution via Redis Streams
        """
        # TODO: Implement proper serialization/deserialization
        # For now, assume data contains the necessary fields
        
        # Deserialize image
        image_bytes = data[b'image']
        # Assuming image is stored as flattened array with shape info
        height = int(data[b'height'])
        width = int(data[b'width'])
        channels = int(data[b'channels'])
        
        image_flat = np.frombuffer(image_bytes, dtype=np.uint8)
        image = image_flat.reshape((height, width, channels))
        
        timestamp = float(data[b'timestamp'])
        frame_number = int(data[b'frame_number'])
        
        return VideoFrame(
            image=image,
            timestamp=timestamp,
            frame_number=frame_number
        )
