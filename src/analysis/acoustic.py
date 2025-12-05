"""Acoustic Analysis Module

This module processes audio frames to extract emotional features and classify emotions
from vocal tone characteristics. It consumes audio frames from Redis Streams asynchronously
and produces timestamped emotion results.

Requirements:
    - Req 1.2: Acoustic Analysis Module extracts vocal tone features continuously
    - Req 3.1: System extracts pitch, energy, speaking rate, and voice quality features
    - Req 3.2: System classifies tone into emotional categories with confidence scores
    - Req 3.3: System filters noise before feature extraction
    - Req 3.5: System reports quality indicators when audio quality is insufficient
"""

import logging
import time
import asyncio
from typing import Optional, Dict
import numpy as np
import librosa
import redis.asyncio as redis
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch

from src.models.frames import AudioFrame
from src.models.features import AcousticFeatures
from src.models.results import AcousticResult
from src.config.config_loader import config


logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Exception raised for errors during audio processing"""
    pass


class AcousticAnalyzer:
    """Analyzes audio frames to extract emotional features and classify emotions.
    
    This class implements the acoustic analysis module that:
    1. Consumes audio frames from Redis Streams asynchronously
    2. Extracts acoustic features (pitch, energy, speaking rate, spectral features)
    3. Applies noise filtering using spectral subtraction
    4. Classifies emotions using pre-trained wav2vec2 model
    5. Reports quality indicators and adjusts confidence accordingly
    6. Caches timestamped results for Fusion Engine access
    
    Attributes:
        model: Pre-trained wav2vec2 emotion recognition model
        processor: Audio processor for model input preparation
        redis_client: Async Redis client for stream consumption
        latest_result: Most recent analysis result (cached)
        confidence_threshold: Minimum confidence for valid results
    """
    
    def __init__(self):
        """Initialize the acoustic analyzer with model and configuration."""
        self.model_path = config.get('acoustic.model_path')
        self.confidence_threshold = config.get('acoustic.confidence_threshold', 0.05)
        self.sample_rate = config.get('audio.sample_rate', 16000)
        self.noise_reduction = config.get('audio.noise_reduction', True)
        self.redis_url = config.get('redis.url', 'redis://localhost:6379')
        self.audio_stream = config.get('redis.audio_stream', 'audio_frames')
        
        # Model and processor (loaded lazily)
        self.model: Optional[Wav2Vec2ForSequenceClassification] = None
        self.processor: Optional[Wav2Vec2Processor] = None
        self.device = "cuda" if config.get('performance.use_gpu', False) and torch.cuda.is_available() else "cpu"
        
        # Result cache
        self.latest_result: Optional[AcousticResult] = None
        
        # Redis client (initialized in start())
        self.redis_client: Optional[redis.Redis] = None
        
        logger.info(f"AcousticAnalyzer initialized with device: {self.device}")
    
    def _load_model(self):
        """Load the pre-trained emotion recognition model.
        
        Raises:
            FileNotFoundError: If model files are not found
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading acoustic model from {self.model_path}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Acoustic model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load acoustic model: {e}", exc_info=True)
            raise
    
    def _extract_features(self, audio_frame: AudioFrame) -> AcousticFeatures:
        """Extract acoustic features from audio frame.
        
        Extracts pitch, energy, speaking rate, and spectral features using librosa.
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            AcousticFeatures containing extracted features
            
        Raises:
            AudioProcessingError: If feature extraction fails
        """
        try:
            samples = audio_frame.samples.astype(np.float32)
            sr = audio_frame.sample_rate
            
            # Extract pitch (F0) using librosa's pyin algorithm
            f0, voiced_flag, voiced_probs = librosa.pyin(
                samples,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # Handle NaN values in pitch
            f0_valid = f0[~np.isnan(f0)]
            pitch_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
            pitch_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
            
            # Extract energy (RMS)
            rms = librosa.feature.rms(y=samples)[0]
            energy_mean = float(np.mean(rms))
            
            # Extract spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)[0]
            spectral_centroid_mean = float(np.mean(spectral_centroid))
            
            # Extract zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(samples)[0]
            zcr_mean = float(np.mean(zcr))
            
            # Estimate speaking rate (simplified: based on onset detection)
            onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            speaking_rate = len(onsets) / audio_frame.duration if audio_frame.duration > 0 else 0.0
            
            return AcousticFeatures(
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                energy_mean=energy_mean,
                speaking_rate=speaking_rate,
                spectral_centroid=spectral_centroid_mean,
                zero_crossing_rate=zcr_mean
            )
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise AudioProcessingError(f"Failed to extract acoustic features: {e}")
    
    def _apply_noise_reduction(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral subtraction for noise filtering.
        
        Args:
            samples: Audio samples
            sr: Sample rate
            
        Returns:
            Noise-reduced audio samples
        """
        try:
            # Compute STFT
            stft = librosa.stft(samples)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames (assuming initial silence/noise)
            noise_frames = min(10, magnitude.shape[1])
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            magnitude_clean = np.maximum(magnitude - noise_profile, 0.0)
            
            # Reconstruct signal
            stft_clean = magnitude_clean * np.exp(1j * phase)
            samples_clean = librosa.istft(stft_clean, length=len(samples))
            
            return samples_clean
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}, using original samples")
            return samples
    
    def _assess_audio_quality(self, audio_frame: AudioFrame, features: AcousticFeatures) -> float:
        """Assess audio quality and return quality indicator.
        
        Args:
            audio_frame: Audio frame to assess
            features: Extracted acoustic features
            
        Returns:
            Quality score in [0, 1] where 1 is highest quality
        """
        quality_score = 1.0
        
        # Check energy level (too low indicates poor quality)
        if features.energy_mean < 0.01:
            quality_score *= 0.5
            logger.debug("Low energy detected, reducing quality score")
        
        # Check for clipping (samples near max value)
        max_amplitude = np.max(np.abs(audio_frame.samples))
        if max_amplitude > 0.95:
            quality_score *= 0.7
            logger.debug("Clipping detected, reducing quality score")
        
        # Check zero crossing rate (very high indicates noise)
        if features.zero_crossing_rate > 0.5:
            quality_score *= 0.6
            logger.debug("High zero crossing rate, reducing quality score")
        
        return quality_score

    def _classify_emotion(self, audio_frame: AudioFrame) -> Dict[str, float]:
        """Classify emotion using pre-trained wav2vec2 model.
        
        Processes audio samples through the pre-trained emotion recognition model
        to generate emotion scores for each category. The model outputs logits that
        are converted to probabilities using softmax, representing the likelihood
        of each emotion being present in the audio.
        
        Args:
            audio_frame: Audio frame to classify containing PCM samples and metadata
            
        Returns:
            Dictionary mapping emotion names to probability scores in [0, 1] range.
            Example: {"angry": 0.1, "happy": 0.7, "sad": 0.05, "neutral": 0.15}
            
        Raises:
            AudioProcessingError: If classification fails due to model inference errors
                                 or audio preprocessing issues
        
        Validates:
            - Req 3.2: System classifies tone into emotional categories with confidence scores
            - Prop 1: Acoustic feature extraction completeness
        """
        try:
            # Ensure model is loaded
            if self.model is None or self.processor is None:
                self._load_model()
            
            # Prepare input for model
            samples = audio_frame.samples.astype(np.float32)
            
            # Resample if necessary
            if audio_frame.sample_rate != 16000:
                samples = librosa.resample(
                    samples,
                    orig_sr=audio_frame.sample_rate,
                    target_sr=16000
                )
            
            # Process audio
            inputs = self.processor(
                samples,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Convert to emotion scores
            # Assuming model outputs: [angry, happy, sad, neutral, fearful, disgust, surprised]
            emotion_labels = ["angry", "happy", "sad", "neutral", "fearful", "disgust", "surprised"]
            emotion_scores = {}
            
            probs_np = probs.cpu().numpy()[0]
            for i, label in enumerate(emotion_labels[:len(probs_np)]):
                emotion_scores[label] = float(probs_np[i])
            
            return emotion_scores
        except Exception as e:
            logger.error(f"Emotion classification failed: {e}")
            raise AudioProcessingError(f"Failed to classify emotion: {e}")
    
    async def analyze_audio(self, audio_frame: AudioFrame) -> Optional[AcousticResult]:
        """Analyze audio frame and return emotion result with quality-adjusted confidence.
        
        This is the main analysis method that orchestrates the complete acoustic analysis
        pipeline. It processes audio frames through multiple stages to extract emotional
        intelligence from vocal tone characteristics while accounting for audio quality.
        
        The analysis pipeline:
        1. Applies spectral subtraction noise reduction if enabled (Req 3.3)
        2. Extracts acoustic features: pitch, energy, speaking rate, spectral features (Req 3.1)
        3. Assesses audio quality based on energy, clipping, and noise indicators (Req 3.5)
        4. Classifies emotion using pre-trained wav2vec2 model (Req 3.2)
        5. Adjusts confidence score based on audio quality assessment (Req 3.5)
        6. Returns timestamped result for fusion engine consumption
        
        Error Handling:
        - AudioProcessingError: Returns low-confidence neutral result (confidence=0.1)
        - Unexpected exceptions: Returns None, allowing fusion to continue with other modalities
        
        Args:
            audio_frame: Audio frame to analyze containing PCM samples, sample rate,
                        timestamp, and duration
            
        Returns:
            AcousticResult with emotion scores, quality-adjusted confidence, extracted
            features, and timestamp. Returns None if unexpected error occurs, allowing
            graceful degradation. Returns low-confidence neutral result if processing
            fails but system should continue.
            
        Validates:
            - Req 1.2: Acoustic Analysis Module extracts vocal tone features continuously
            - Req 3.1: System extracts pitch, energy, speaking rate, and voice quality features
            - Req 3.2: System classifies tone into emotional categories with confidence scores
            - Req 3.3: System filters noise before feature extraction
            - Req 3.5: System reports quality indicators when audio quality is insufficient
            - Prop 1: Acoustic feature extraction completeness
        """
        try:
            # Apply noise reduction if enabled
            if self.noise_reduction:
                samples_clean = self._apply_noise_reduction(
                    audio_frame.samples,
                    audio_frame.sample_rate
                )
                audio_frame = AudioFrame(
                    samples=samples_clean,
                    sample_rate=audio_frame.sample_rate,
                    timestamp=audio_frame.timestamp,
                    duration=audio_frame.duration
                )
            
            # Extract features
            features = self._extract_features(audio_frame)
            
            # Assess audio quality (combine stream quality with analysis quality)
            analysis_quality = self._assess_audio_quality(audio_frame, features)
            stream_quality = audio_frame.quality_score
            # Combined quality is the minimum of both (conservative approach)
            quality_score = min(analysis_quality, stream_quality)
            
            # Classify emotion
            emotion_scores = self._classify_emotion(audio_frame)
            
            # Compute confidence (based on quality and model confidence)
            max_emotion_score = max(emotion_scores.values())
            base_confidence = max_emotion_score
            adjusted_confidence = base_confidence * quality_score
            
            # Create result
            result = AcousticResult(
                emotion_scores=emotion_scores,
                confidence=adjusted_confidence,
                features=features,
                timestamp=time.time()
            )
            
            # Cache result
            self.latest_result = result
            
            logger.debug(f"Acoustic analysis complete: confidence={adjusted_confidence:.3f}, "
                        f"quality={quality_score:.3f}")
            
            return result
            
        except AudioProcessingError as e:
            logger.warning(f"Audio processing failed: {e}")
            # Return low-confidence neutral result
            result = AcousticResult(
                emotion_scores={"neutral": 1.0},
                confidence=0.1,  # Very low confidence
                features=None,
                timestamp=time.time()
            )
            self.latest_result = result
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in acoustic analysis: {e}", exc_info=True)
            return None
    
    def get_latest_result(self) -> Optional[AcousticResult]:
        """Get the most recent analysis result from cache.
        
        Used by Fusion Engine to access cached results during time-windowed fusion.
        This method provides non-blocking access to the latest acoustic analysis result,
        enabling the fusion engine to operate on fixed 1-second intervals without waiting
        for analysis completion.
        
        Returns:
            Latest AcousticResult containing emotion scores, confidence, features, and
            timestamp. Returns None if no analysis has been performed yet (e.g., at
            system startup before first frame is processed).
            
        Validates:
            - Req 6.1: Fusion Engine receives outputs from analysis modules
            - Design: Result caching with timestamps for non-blocking fusion
        """
        return self.latest_result
    
    async def start(self):
        """Start consuming audio frames from Redis Streams asynchronously.
        
        This method runs as an independent asyncio task and continuously consumes
        audio frames from the Redis stream, analyzes them, and caches results. It
        implements the asynchronous, event-driven architecture that prevents slow
        modules from blocking fast ones.
        
        The method:
        1. Initializes async Redis client connection
        2. Loads the pre-trained emotion recognition model
        3. Continuously reads from Redis Streams using non-blocking xread
        4. Deserializes and analyzes each audio frame
        5. Caches timestamped results for Fusion Engine access
        6. Handles errors gracefully without crashing the pipeline
        
        This task runs indefinitely until cancelled via asyncio.CancelledError,
        enabling clean shutdown of the analysis pipeline.
        
        Raises:
            Exception: Fatal errors during initialization (Redis connection, model loading)
                      are propagated to allow system to fail fast at startup
        
        Validates:
            - Req 1.1: System begins processing within 2 seconds of stream initiation
            - Req 1.2: Acoustic Analysis Module extracts vocal tone features continuously
            - Req 9.1: End-to-end latency not exceeding 3 seconds
            - Design: Asynchronous processing with independent asyncio tasks
            - Design: Result caching with timestamps for non-blocking fusion
        """
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url)
            logger.info(f"Connected to Redis at {self.redis_url}")
            
            # Load model
            self._load_model()
            
            # Start consuming from stream
            last_id = '0-0'  # Start from beginning
            logger.info(f"Starting to consume from stream: {self.audio_stream}")
            
            while True:
                try:
                    # Read from stream (blocking with timeout)
                    messages = await self.redis_client.xread(
                        {self.audio_stream: last_id},
                        block=100,  # 100ms timeout
                        count=1
                    )
                    
                    if messages:
                        for stream_name, message_list in messages:
                            for message_id, data in message_list:
                                # Deserialize audio frame
                                audio_frame = self._deserialize_frame(data)
                                
                                # Analyze audio
                                await self.analyze_audio(audio_frame)
                                
                                # Update last_id for next read
                                last_id = message_id
                    
                    # Small delay to prevent tight loop
                    await asyncio.sleep(0.01)
                    
                except asyncio.CancelledError:
                    logger.info("Acoustic analyzer task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}", exc_info=True)
                    await asyncio.sleep(0.1)  # Back off on error
                    
        except Exception as e:
            logger.error(f"Fatal error in acoustic analyzer: {e}", exc_info=True)
            raise
        finally:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
    
    def _deserialize_frame(self, data: Dict) -> AudioFrame:
        """Deserialize audio frame from Redis stream data.
        
        Converts Redis stream message data back into an AudioFrame object for analysis.
        The serialization format uses pickle for numpy arrays and standard types for
        metadata fields.
        
        Args:
            data: Dictionary containing serialized frame data with keys:
                 - 'samples': Pickled numpy array of PCM audio samples
                 - 'sample_rate': Integer sample rate in Hz
                 - 'timestamp': Float timestamp in seconds since stream start
                 - 'duration': Float duration in seconds
            
        Returns:
            Deserialized AudioFrame ready for acoustic analysis
            
        Validates:
            - Req 1.1: Continuous audio processing through Redis Streams
            - Design: Asynchronous frame distribution via Redis Streams
        """
        # TODO: Implement proper serialization/deserialization
        # For now, assume data contains the necessary fields
        samples = np.frombuffer(data[b'samples'], dtype=np.float32)
        sample_rate = int(data[b'sample_rate'])
        timestamp = float(data[b'timestamp'])
        duration = float(data[b'duration'])
        quality_score = float(data.get(b'quality_score', b'1.0'))
        codec = data.get(b'codec', b'unknown').decode('utf-8')
        
        return AudioFrame(
            samples=samples,
            sample_rate=sample_rate,
            timestamp=timestamp,
            duration=duration,
            quality_score=quality_score,
            codec=codec
        )
