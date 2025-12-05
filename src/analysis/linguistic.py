"""Linguistic Analysis Module

This module processes audio frames to transcribe speech and analyze sentiment from
semantic content. It consumes audio frames from Redis Streams asynchronously and
produces timestamped emotion results based on linguistic analysis.

Requirements:
    - Req 1.4: Linguistic Analysis Module transcribes and analyzes text continuously
    - Req 5.1: System transcribes speech to text using automatic speech recognition
    - Req 5.2: System performs sentiment analysis on semantic content
    - Req 5.3: System identifies emotional polarity, intensity, and specific emotion categories
    - Req 5.4: System reports transcription confidence and adjusts linguistic confidence
    - Req 5.5: System applies context-aware sentiment interpretation for domain-specific terminology
"""

import logging
import time
import asyncio
from typing import Optional, Dict
import numpy as np
import redis.asyncio as redis
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from src.models.frames import AudioFrame
from src.models.results import LinguisticResult
from src.config.config_loader import config


logger = logging.getLogger(__name__)


class LinguisticProcessingError(Exception):
    """Exception raised for errors during linguistic processing"""
    pass


class LinguisticAnalyzer:
    """Analyzes audio frames to transcribe speech and classify sentiment.
    
    This class implements the linguistic analysis module that:
    1. Consumes audio frames from Redis Streams asynchronously
    2. Buffers audio for context-aware transcription (sliding window)
    3. Transcribes speech using Whisper
    4. Performs sentiment analysis using transformer model (DistilBERT)
    5. Reports transcription confidence and adjusts linguistic confidence
    6. Caches timestamped results for Fusion Engine access
    7. Processes at lower frequency (every 2-3 seconds) due to computational cost
    
    Attributes:
        whisper_model: Whisper model for speech-to-text
        sentiment_tokenizer: Tokenizer for sentiment model
        sentiment_model: Pre-trained sentiment analysis model
        redis_client: Async Redis client for stream consumption
        latest_result: Most recent analysis result (cached)
        audio_buffer: Sliding window buffer for context-aware transcription
        processing_interval: Time between processing cycles (seconds)
        last_process_time: Timestamp of last processing
    """
    
    def __init__(self):
        """Initialize the linguistic analyzer with models and configuration."""
        self.whisper_model_name = config.get('linguistic.whisper_model', 'base')
        self.sentiment_model_name = config.get('linguistic.sentiment_model', 
                                               'distilbert-base-uncased-finetuned-sst-2-english')
        self.buffer_duration = config.get('linguistic.buffer_duration', 3.0)
        self.processing_interval = config.get('linguistic.processing_interval', 2.0)
        self.redis_url = config.get('redis.url', 'redis://localhost:6379')
        self.audio_stream = config.get('redis.audio_stream', 'audio_frames')
        
        # Models (loaded lazily)
        self.whisper_model: Optional[whisper.Whisper] = None
        self.sentiment_tokenizer: Optional[AutoTokenizer] = None
        self.sentiment_model: Optional[AutoModelForSequenceClassification] = None
        self.device = "cuda" if config.get('performance.use_gpu', False) and torch.cuda.is_available() else "cpu"
        
        # Result cache
        self.latest_result: Optional[LinguisticResult] = None
        
        # Audio buffer for sliding window
        self.audio_buffer: list[AudioFrame] = []
        self.last_process_time = 0.0
        
        # Redis client (initialized in start())
        self.redis_client: Optional[redis.Redis] = None
        
        logger.info(f"LinguisticAnalyzer initialized with device: {self.device}, "
                   f"processing_interval: {self.processing_interval}s")
    
    def _load_models(self):
        """Load Whisper and sentiment analysis models.
        
        Initializes the Whisper speech-to-text model and DistilBERT sentiment analysis
        model, moving them to the appropriate device (CPU or GPU). This method is called
        lazily on first use to avoid loading models during initialization.
        
        The method:
        1. Loads Whisper model for speech transcription (Req 5.1)
        2. Loads DistilBERT tokenizer and model for sentiment analysis (Req 5.2)
        3. Moves models to configured device (CPU/GPU)
        4. Sets models to evaluation mode for inference
        
        Raises:
            Exception: If model loading fails (e.g., model files not found, insufficient
                      memory, CUDA errors). Fatal errors are propagated to fail fast at
                      startup rather than during stream processing.
        
        Validates:
            - Req 5.1: System transcribes speech to text using automatic speech recognition
            - Req 5.2: System performs sentiment analysis on semantic content
        """
        try:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_model = whisper.load_model(self.whisper_model_name, device=self.device)
            logger.info("Whisper model loaded successfully")
            
            logger.info(f"Loading sentiment model: {self.sentiment_model_name}")
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_model_name
            )
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load linguistic models: {e}", exc_info=True)
            raise
    
    def _buffer_audio(self, audio_frame: AudioFrame):
        """Add audio frame to buffer and maintain sliding window.
        
        Implements a sliding window buffer that maintains the most recent audio frames
        within the configured buffer duration (typically 3-5 seconds). This provides
        context for more accurate speech transcription by ensuring complete phrases
        and sentences are captured.
        
        The method:
        1. Appends the new audio frame to the buffer
        2. Removes frames older than buffer_duration from the current timestamp
        3. Maintains temporal ordering of frames
        
        Args:
            audio_frame: AudioFrame to add to buffer containing PCM samples, sample rate,
                        timestamp, and duration
        
        Validates:
            - Req 5.1: System transcribes speech to text using automatic speech recognition
            - Design: Audio buffering with sliding window for context-aware transcription
        """
        self.audio_buffer.append(audio_frame)
        
        # Remove old frames outside buffer duration
        current_time = audio_frame.timestamp
        self.audio_buffer = [
            frame for frame in self.audio_buffer
            if current_time - frame.timestamp <= self.buffer_duration
        ]
    
    def _get_buffered_audio(self) -> Optional[np.ndarray]:
        """Get concatenated audio from buffer.
        
        Concatenates all audio frames currently in the sliding window buffer into a
        single continuous audio array suitable for Whisper transcription. This provides
        the context needed for accurate speech recognition by including complete phrases.
        
        The method:
        1. Checks if buffer contains any frames
        2. Extracts samples from each buffered AudioFrame
        3. Concatenates samples into a single numpy array
        4. Converts to float32 format required by Whisper
        
        Returns:
            np.ndarray: Concatenated audio samples as float32 array, or None if buffer
                       is empty (e.g., at system startup before first frame arrives)
        
        Validates:
            - Req 5.1: System transcribes speech to text using automatic speech recognition
            - Design: Audio buffering with sliding window for context-aware transcription
        """
        if not self.audio_buffer:
            return None
        
        # Concatenate all audio frames in buffer
        audio_segments = [frame.samples for frame in self.audio_buffer]
        concatenated = np.concatenate(audio_segments)
        
        return concatenated.astype(np.float32)
    
    def _transcribe_speech(self, audio_samples: np.ndarray) -> tuple[str, float]:
        """Transcribe speech using Whisper.
        
        Processes buffered audio through the Whisper automatic speech recognition model
        to generate text transcription. The method also computes transcription confidence
        based on Whisper's internal no_speech_prob metric, which is used to adjust the
        final linguistic confidence score.
        
        The method:
        1. Ensures Whisper model is loaded (lazy loading)
        2. Transcribes audio using Whisper with appropriate precision (fp16 for GPU)
        3. Extracts transcription text from result
        4. Computes average confidence from segment-level no_speech_prob scores
        5. Returns both transcription and confidence for downstream processing
        
        Args:
            audio_samples: np.ndarray of PCM audio samples (float32) to transcribe,
                          typically 3-5 seconds of buffered audio for context
            
        Returns:
            tuple[str, float]: A tuple containing:
                - transcription (str): Transcribed text, stripped of leading/trailing whitespace
                - confidence (float): Transcription confidence in [0, 1] range, computed as
                                     1.0 - avg(no_speech_prob) across segments
            
        Raises:
            LinguisticProcessingError: If transcription fails due to model errors, invalid
                                      audio format, or insufficient memory
        
        Validates:
            - Req 5.1: System transcribes speech to text using automatic speech recognition
            - Req 5.4: System reports transcription confidence and adjusts linguistic confidence
        """
        try:
            # Ensure model is loaded
            if self.whisper_model is None:
                self._load_models()
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(
                audio_samples,
                fp16=(self.device == "cuda"),
                language='en'  # Can be made configurable
            )
            
            transcription = result['text'].strip()
            
            # Compute average confidence from segments
            segments = result.get('segments', [])
            if segments:
                confidences = []
                for segment in segments:
                    # Whisper doesn't directly provide confidence, use no_speech_prob as proxy
                    no_speech_prob = segment.get('no_speech_prob', 0.5)
                    confidence = 1.0 - no_speech_prob
                    confidences.append(confidence)
                avg_confidence = float(np.mean(confidences))
            else:
                avg_confidence = 0.5  # Default moderate confidence
            
            logger.debug(f"Transcription: '{transcription}' (confidence: {avg_confidence:.3f})")
            
            return transcription, avg_confidence
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise LinguisticProcessingError(f"Failed to transcribe speech: {e}")
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of transcribed text.
        
        Processes transcribed text through a DistilBERT sentiment analysis model to
        classify emotional content. The model outputs positive/negative sentiment scores
        which are then mapped to specific emotion categories (happy, sad, angry, etc.)
        for consistency with acoustic and visual analysis outputs.
        
        The method:
        1. Ensures sentiment model and tokenizer are loaded (lazy loading)
        2. Handles empty text by returning neutral sentiment
        3. Tokenizes text with truncation and padding for model input
        4. Runs inference through DistilBERT model
        5. Applies softmax to convert logits to probabilities
        6. Maps binary sentiment (negative/positive) to emotion categories
        7. Normalizes emotion scores to sum to 1.0
        
        The mapping from DistilBERT SST-2 output to emotions:
        - Negative score → sad (40%), angry (30%), fearful (20%), disgust (10%)
        - Positive score → happy (70%), surprised (20%), neutral (10%)
        
        Args:
            text: str containing transcribed speech to analyze for sentiment and emotion
            
        Returns:
            Dict[str, float]: Dictionary mapping emotion names to probability scores in
                             [0, 1] range. Scores sum to approximately 1.0. Example:
                             {"happy": 0.6, "sad": 0.2, "neutral": 0.2}
            
        Raises:
            LinguisticProcessingError: If sentiment analysis fails due to model errors,
                                      tokenization issues, or device errors
        
        Validates:
            - Req 5.2: System performs sentiment analysis on semantic content
            - Req 5.3: System identifies emotional polarity, intensity, and specific emotion categories
        """
        try:
            # Ensure model is loaded
            if self.sentiment_model is None or self.sentiment_tokenizer is None:
                self._load_models()
            
            # Handle empty text
            if not text or len(text.strip()) == 0:
                return {"neutral": 1.0}
            
            # Tokenize text
            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Convert to emotion scores
            # DistilBERT SST-2 outputs: [negative, positive]
            # Map to emotion categories
            probs_np = probs.cpu().numpy()[0]
            
            negative_score = float(probs_np[0])
            positive_score = float(probs_np[1])
            
            # Map to emotion categories
            emotion_scores = {
                "sad": negative_score * 0.4,
                "angry": negative_score * 0.3,
                "fearful": negative_score * 0.2,
                "disgust": negative_score * 0.1,
                "happy": positive_score * 0.7,
                "surprised": positive_score * 0.2,
                "neutral": positive_score * 0.1
            }
            
            # Normalize to sum to 1.0
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v / total for k, v in emotion_scores.items()}
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise LinguisticProcessingError(f"Failed to analyze sentiment: {e}")
    
    def _apply_domain_adaptation(self, text: str, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Apply domain-specific sentiment interpretation.
        
        Adjusts sentiment scores based on domain-specific terminology that may not be
        correctly interpreted by general-purpose sentiment models. For example, financial
        terms like "bear market" or "crisis" carry strong negative sentiment in financial
        contexts that may not be fully captured by standard sentiment analysis.
        
        This MVP implementation focuses on financial domain terminology, but the approach
        can be extended to other domains (medical, legal, etc.) by adding domain-specific
        term dictionaries.
        
        The method:
        1. Defines domain-specific negative terms (crisis, crash, recession, etc.)
        2. Defines domain-specific positive terms (bull market, growth, rally, etc.)
        3. Scans transcribed text for these terms (case-insensitive)
        4. Boosts relevant emotion scores when domain terms are detected
        5. Re-normalizes scores to maintain probability distribution
        
        Adjustment strategy:
        - Negative terms: Boost sad (+0.2) and fearful (+0.1) emotions
        - Positive terms: Boost happy (+0.2) emotion
        
        Args:
            text: str containing transcribed text to scan for domain-specific terms
            emotion_scores: Dict[str, float] containing initial emotion scores from
                           sentiment analysis, will be adjusted based on domain terms
            
        Returns:
            Dict[str, float]: Adjusted emotion scores with domain-specific boosts applied
                             and re-normalized to sum to 1.0
        
        Validates:
            - Req 5.5: System applies context-aware sentiment interpretation for domain-specific terminology
            - Prop 15: Domain-specific sentiment interpretation (if implemented)
        """
        # Domain-specific terms (can be expanded)
        negative_financial_terms = [
            'crisis', 'crash', 'bear market', 'recession', 'systemic risk',
            'default', 'bankruptcy', 'collapse', 'downturn'
        ]
        
        positive_financial_terms = [
            'bull market', 'growth', 'rally', 'surge', 'boom',
            'profit', 'gains', 'recovery', 'expansion'
        ]
        
        text_lower = text.lower()
        
        # Check for negative terms
        for term in negative_financial_terms:
            if term in text_lower:
                # Boost negative emotions
                emotion_scores['sad'] = min(emotion_scores.get('sad', 0) + 0.2, 1.0)
                emotion_scores['fearful'] = min(emotion_scores.get('fearful', 0) + 0.1, 1.0)
                logger.debug(f"Domain term detected: '{term}', boosting negative sentiment")
                break
        
        # Check for positive terms
        for term in positive_financial_terms:
            if term in text_lower:
                # Boost positive emotions
                emotion_scores['happy'] = min(emotion_scores.get('happy', 0) + 0.2, 1.0)
                logger.debug(f"Domain term detected: '{term}', boosting positive sentiment")
                break
        
        # Normalize
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}
        
        return emotion_scores
    
    async def analyze_audio(self, audio_frame: AudioFrame) -> Optional[LinguisticResult]:
        """Analyze buffered audio and return linguistic result with transcription confidence.
        
        This is the main analysis method that orchestrates the complete linguistic analysis
        pipeline. It processes buffered audio through multiple stages to extract emotional
        intelligence from semantic content while accounting for transcription quality.
        
        The analysis pipeline:
        1. Buffers audio frames in sliding window for context (Req 5.1)
        2. Checks if processing interval has elapsed (lower frequency processing)
        3. Transcribes buffered speech using Whisper (Req 5.1)
        4. Performs sentiment analysis on transcription using DistilBERT (Req 5.2, 5.3)
        5. Applies domain-specific sentiment interpretation (Req 5.5)
        6. Adjusts confidence based on transcription quality (Req 5.4)
        7. Returns timestamped result for fusion engine consumption
        
        Error Handling:
        - LinguisticProcessingError: Returns low-confidence neutral result (confidence=0.1)
        - Unexpected exceptions: Returns None, allowing fusion to continue with other modalities
        
        Args:
            audio_frame: Audio frame to buffer and potentially analyze
            
        Returns:
            LinguisticResult with transcription, emotion scores, confidence, transcription
            confidence, and timestamp. Returns None if processing interval hasn't elapsed
            or if unexpected error occurs. Returns low-confidence neutral result if
            processing fails but system should continue.
            
        Validates:
            - Req 1.4: Linguistic Analysis Module transcribes and analyzes text continuously
            - Req 5.1: System transcribes speech to text using automatic speech recognition
            - Req 5.2: System performs sentiment analysis on semantic content
            - Req 5.3: System identifies emotional polarity, intensity, and specific emotion categories
            - Req 5.4: System reports transcription confidence and adjusts linguistic confidence
            - Req 5.5: System applies context-aware sentiment interpretation
            - Prop 3: Linguistic analysis completeness
        """
        try:
            # Buffer audio frame
            self._buffer_audio(audio_frame)
            
            # Check if processing interval has elapsed
            current_time = time.time()
            if current_time - self.last_process_time < self.processing_interval:
                # Not time to process yet
                return None
            
            self.last_process_time = current_time
            
            # Get buffered audio
            audio_samples = self._get_buffered_audio()
            if audio_samples is None or len(audio_samples) < 1600:  # Minimum 0.1s at 16kHz
                logger.debug("Insufficient audio in buffer, skipping processing")
                return None
            
            # Transcribe speech
            transcription, transcription_confidence = self._transcribe_speech(audio_samples)
            
            # Handle empty transcription
            if not transcription or len(transcription.strip()) == 0:
                logger.debug("Empty transcription, returning neutral result")
                result = LinguisticResult(
                    transcription="",
                    emotion_scores={"neutral": 1.0},
                    confidence=0.2,
                    transcription_confidence=transcription_confidence,
                    timestamp=current_time
                )
                self.latest_result = result
                return result
            
            # Analyze sentiment
            emotion_scores = self._analyze_sentiment(transcription)
            
            # Apply domain adaptation
            emotion_scores = self._apply_domain_adaptation(transcription, emotion_scores)
            
            # Compute confidence (based on transcription confidence, sentiment strength, and stream quality)
            max_emotion_score = max(emotion_scores.values())
            base_confidence = max_emotion_score
            
            # Calculate average stream quality from buffered frames
            if self.audio_buffer:
                avg_stream_quality = np.mean([frame.quality_score for frame in self.audio_buffer])
            else:
                avg_stream_quality = 1.0
            
            # Combine all quality factors
            adjusted_confidence = base_confidence * transcription_confidence * avg_stream_quality
            
            # Create result
            result = LinguisticResult(
                transcription=transcription,
                emotion_scores=emotion_scores,
                confidence=adjusted_confidence,
                transcription_confidence=transcription_confidence,
                timestamp=current_time
            )
            
            # Cache result
            self.latest_result = result
            
            logger.debug(f"Linguistic analysis complete: '{transcription}' "
                        f"confidence={adjusted_confidence:.3f}, "
                        f"transcription_conf={transcription_confidence:.3f}")
            
            return result
            
        except LinguisticProcessingError as e:
            logger.warning(f"Linguistic processing failed: {e}")
            # Return low-confidence neutral result
            result = LinguisticResult(
                transcription="",
                emotion_scores={"neutral": 1.0},
                confidence=0.1,
                transcription_confidence=0.1,
                timestamp=time.time()
            )
            self.latest_result = result
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in linguistic analysis: {e}", exc_info=True)
            return None
    
    def get_latest_result(self) -> Optional[LinguisticResult]:
        """Get the most recent analysis result from cache.
        
        Used by Fusion Engine to access cached results during time-windowed fusion.
        This method provides non-blocking access to the latest linguistic analysis result,
        enabling the fusion engine to operate on fixed 1-second intervals without waiting
        for analysis completion.
        
        Returns:
            Latest LinguisticResult containing transcription, emotion scores, confidence,
            transcription confidence, and timestamp. Returns None if no analysis has been
            performed yet (e.g., at system startup before first processing cycle).
            
        Validates:
            - Req 6.1: Fusion Engine receives outputs from analysis modules
            - Design: Result caching with timestamps for non-blocking fusion
        """
        return self.latest_result
    
    async def start(self):
        """Start consuming audio frames from Redis Streams asynchronously.
        
        This method runs as an independent asyncio task and continuously consumes
        audio frames from the Redis stream, buffers them, and periodically analyzes
        them. It implements the asynchronous, event-driven architecture that prevents
        slow modules from blocking fast ones.
        
        The method:
        1. Initializes async Redis client connection
        2. Loads Whisper and sentiment analysis models
        3. Continuously reads from Redis Streams using non-blocking xread
        4. Buffers audio frames in sliding window
        5. Periodically processes buffered audio (every 2-3 seconds)
        6. Caches timestamped results for Fusion Engine access
        7. Handles errors gracefully without crashing the pipeline
        
        Lower-frequency processing (every 2-3 seconds) is implemented because linguistic
        analysis is computationally expensive (Whisper transcription + transformer sentiment).
        This maintains real-time performance while still providing valuable semantic context.
        
        This task runs indefinitely until cancelled via asyncio.CancelledError,
        enabling clean shutdown of the analysis pipeline.
        
        Raises:
            Exception: Fatal errors during initialization (Redis connection, model loading)
                      are propagated to allow system to fail fast at startup
        
        Validates:
            - Req 1.1: System begins processing within 2 seconds of stream initiation
            - Req 1.4: Linguistic Analysis Module transcribes and analyzes text continuously
            - Req 9.1: End-to-end latency not exceeding 3 seconds
            - Design: Asynchronous processing with independent asyncio tasks
            - Design: Lower-frequency processing (every 2-3 seconds) for linguistic analysis
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
                                
                                # Analyze audio (will buffer and process periodically)
                                await self.analyze_audio(audio_frame)
                                
                                # Update last_id for next read
                                last_id = message_id
                    
                    # Small delay to prevent tight loop
                    await asyncio.sleep(0.01)
                    
                except asyncio.CancelledError:
                    logger.info("Linguistic analyzer task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}", exc_info=True)
                    await asyncio.sleep(0.1)  # Back off on error
                    
        except Exception as e:
            logger.error(f"Fatal error in linguistic analyzer: {e}", exc_info=True)
            raise
        finally:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
    
    def _deserialize_frame(self, data: Dict) -> AudioFrame:
        """Deserialize audio frame from Redis stream data.
        
        Converts Redis stream message data back into an AudioFrame object for buffering
        and analysis. The serialization format uses pickle for numpy arrays and standard
        types for metadata fields.
        
        Args:
            data: Dictionary containing serialized frame data with keys:
                 - 'samples': Pickled numpy array of PCM audio samples
                 - 'sample_rate': Integer sample rate in Hz
                 - 'timestamp': Float timestamp in seconds since stream start
                 - 'duration': Float duration in seconds
            
        Returns:
            Deserialized AudioFrame ready for buffering and linguistic analysis
            
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
