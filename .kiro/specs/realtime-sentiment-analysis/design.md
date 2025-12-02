# Design Document: Real-Time Multimedia Sentiment Analysis Engine

## Overview

The Real-Time Multimedia Sentiment Analysis Engine is a multi-modal AI system that processes live video streams to extract emotional intelligence. The system operates as a pipeline that ingests multimedia streams, processes them through three parallel analysis modules (acoustic, visual, and linguistic), and fuses the results into a unified sentiment score. The architecture prioritizes low latency, modularity, and extensibility to support real-time decision-making in time-sensitive contexts.

The system is designed as a proof-of-concept that demonstrates the feasibility of real-time emotional analysis from multimedia sources. The MVP focuses on core analysis capabilities with a simple visualization interface, deferring advanced features like alerting and external API integration to future iterations.

## Architecture

### High-Level Architecture

The system follows an asynchronous, event-driven architecture with parallel processing streams to meet real-time latency requirements:

```
                                    ┌→ [Acoustic Analysis] → [Result Cache]
                                    │                              ↓
[Stream Input] → [Stream Decoder] → ├→ [Visual Analysis] → [Result Cache] → [Fusion Engine] → [Sentiment Score] → [UI Display]
                                    │                              ↑              ↑
                                    └→ [Linguistic Analysis] → [Result Cache]    │
                                                                                  │
                                                              [Timer: 1 second intervals]
```

**Key Architectural Decisions**:
- **Asynchronous Processing**: Analysis modules run independently and asynchronously to prevent slow modules from blocking fast ones
- **Result Caching**: Each module writes timestamped results to a local cache that the Fusion Engine reads from
- **Time-Windowed Fusion**: Fusion Engine operates on a fixed 1-second timer, reading the latest available results from each module
- **Non-Blocking Pipeline**: If a module hasn't produced a result in the current window, Fusion uses the last known value or reduces that modality's weight

### Component Layers

1. **Input Layer**: Handles stream ingestion, protocol negotiation, and decoding
2. **Distribution Layer**: Distributes frames to analysis modules asynchronously
3. **Analysis Layer**: Three independent modules for acoustic, visual, and linguistic processing (run in parallel)
4. **Caching Layer**: Stores timestamped results from each analysis module
5. **Fusion Layer**: Combines multi-modal signals with quality-aware weighting on fixed time intervals
6. **Output Layer**: Provides real-time visualization and data access

### Technology Stack

- **Language**: Python 3.10+ (for ML/AI library ecosystem)
- **Async Framework**: asyncio for asynchronous task management
- **Message Queue**: Redis Streams for asynchronous frame distribution (lightweight, fast, Python-friendly)
- **Stream Processing**: OpenCV for video, PyAV for audio/video decoding
- **Acoustic Analysis**: librosa for audio feature extraction, pre-trained emotion recognition models
- **Visual Analysis**: MediaPipe or OpenCV for face detection, pre-trained facial expression recognition models
- **Linguistic Analysis**: Whisper for speech-to-text, transformers library for sentiment analysis
- **Fusion**: NumPy for numerical processing
- **UI**: Streamlit for rapid prototyping of visualization interface
- **Testing**: pytest for unit tests, Hypothesis for property-based testing

## Components and Interfaces

### 1. Stream Input Manager

**Responsibility**: Ingests multimedia streams, decodes them, and publishes frames to the message queue for asynchronous processing.

**Interface**:
```python
class StreamInputManager:
    def connect(self, stream_url: str, protocol: StreamProtocol) -> StreamConnection
    async def start_streaming(self) -> None  # Continuously publishes frames to queue
    def is_active(self) -> bool
    def disconnect(self) -> None
    
    # Internal methods
    async def _publish_audio_frame(self, frame: AudioFrame) -> None
    async def _publish_video_frame(self, frame: VideoFrame) -> None
```

**Key Behaviors**:
- Supports RTMP, HLS, WebRTC, and local file input
- Decodes streams into raw audio (PCM) and video (RGB frames)
- Publishes frames to Redis Streams with timestamps for asynchronous consumption
- Handles reconnection on stream interruption
- Buffers frames to smooth out network jitter
- Runs in a dedicated asyncio task to avoid blocking

### 2. Acoustic Analysis Module

**Responsibility**: Consumes audio frames from the queue, extracts emotional features, and writes timestamped results to cache.

**Interface**:
```python
class AcousticAnalyzer:
    async def start(self) -> None  # Starts consuming from queue
    async def analyze_audio(self, audio_frame: AudioFrame) -> AcousticResult
    def get_latest_result(self) -> Optional[AcousticResult]  # For Fusion Engine
    
class AcousticResult:
    emotion_scores: Dict[str, float]  # e.g., {"happy": 0.7, "sad": 0.1, ...}
    confidence: float
    features: AcousticFeatures  # pitch, energy, speaking_rate, etc.
    timestamp: float  # When this result was generated
```

**Key Behaviors**:
- Consumes audio frames from Redis Streams asynchronously
- Extracts acoustic features: pitch (F0), energy (RMS), speaking rate, spectral features
- Applies noise filtering using spectral subtraction
- Classifies emotion using pre-trained model (e.g., fine-tuned wav2vec2)
- Detects speaker changes using voice activity detection and speaker diarization
- Reports quality indicators when audio is degraded
- Writes timestamped results to local cache for Fusion Engine access
- Runs in a dedicated asyncio task

### 3. Visual Analysis Module

**Responsibility**: Consumes video frames from the queue, extracts emotional features from facial expressions, and writes timestamped results to cache.

**Interface**:
```python
class VisualAnalyzer:
    async def start(self) -> None  # Starts consuming from queue
    async def analyze_frame(self, video_frame: VideoFrame) -> VisualResult
    def get_latest_result(self) -> Optional[VisualResult]  # For Fusion Engine
    
class VisualResult:
    emotion_scores: Dict[str, float]
    confidence: float
    face_detected: bool
    face_landmarks: Optional[FaceLandmarks]
    timestamp: float  # When this result was generated
```

**Key Behaviors**:
- Consumes video frames from Redis Streams asynchronously
- Detects faces using MediaPipe Face Detection or MTCNN
- Extracts facial landmarks (68-point or 478-point model)
- Classifies expressions using pre-trained model (e.g., FER2013-trained CNN)
- Identifies primary speaker using audio-visual synchronization component (lip movement correlation with audio)
- Handles occlusion and poor lighting with quality indicators
- Writes timestamped results to local cache for Fusion Engine access
- Runs in a dedicated asyncio task
- May skip frames (process every 2nd or 3rd frame) to maintain real-time performance

### 4. Linguistic Analysis Module

**Responsibility**: Consumes audio frames from the queue, transcribes speech, analyzes sentiment, and writes timestamped results to cache.

**Interface**:
```python
class LinguisticAnalyzer:
    async def start(self) -> None  # Starts consuming from queue
    async def analyze_audio(self, audio_frame: AudioFrame) -> LinguisticResult
    def get_latest_result(self) -> Optional[LinguisticResult]  # For Fusion Engine
    
class LinguisticResult:
    transcription: str
    emotion_scores: Dict[str, float]
    confidence: float
    transcription_confidence: float
    timestamp: float  # When this result was generated
```

**Key Behaviors**:
- Consumes audio frames from Redis Streams asynchronously
- Transcribes speech using Whisper (base or small model for speed)
- Performs sentiment analysis using transformer model (e.g., DistilBERT fine-tuned on emotion)
- Buffers audio for context-aware transcription (sliding window of 3-5 seconds)
- Reports transcription confidence to weight linguistic contribution
- Handles domain-specific terminology through vocabulary adaptation
- Writes timestamped results to local cache for Fusion Engine access
- Runs in a dedicated asyncio task
- May process at lower frequency than visual/acoustic (e.g., every 2-3 seconds) due to computational cost

### 5. Fusion Engine

**Responsibility**: Combines multi-modal signals into a unified sentiment score on fixed time intervals.

**Interface**:
```python
class FusionEngine:
    async def start(self) -> None  # Starts fusion timer (1 second intervals)
    def fuse_latest_results(self) -> SentimentScore
    def get_latest_score(self) -> Optional[SentimentScore]
    
class SentimentScore:
    score: float  # -1.0 to 1.0
    confidence: float
    modality_contributions: Dict[str, float]
    emotion_breakdown: Dict[str, float]
    timestamp: float
```

**Key Behaviors**:
- Operates on a fixed 1-second timer (asyncio periodic task)
- Queries latest timestamped results from each analysis module's cache
- Handles missing modality data gracefully:
  - If a modality hasn't produced a result in the current window, uses last known value with reduced weight
  - If a modality has never produced a result, excludes it from fusion
- Computes confidence-weighted fusion using refined formula:
  - `weight_m = modality_confidence_m * baseline_weight_m`
  - `score = Σ(weight_m * modality_score_m) / Σ(weight_m)`
- Applies conflict resolution rules:
  - When two modalities agree and one disagrees, reduce the outlier's weight by 50%
  - When all three disagree significantly, report low confidence (<0.5)
- Dynamically adjusts weights based on confidence scores from each modality
- Normalizes scores to [-1, 1] range (negative = negative sentiment, positive = positive sentiment)
- Applies temporal smoothing using exponential moving average (α=0.3) to reduce noise while preserving large shifts
- Maintains emotion category breakdown (happy, sad, angry, neutral, etc.)
- Writes sentiment scores to output queue for UI consumption

### 6. Audio-Visual Synchronization Component

**Responsibility**: Identifies the primary speaker in multi-face scenarios by correlating lip movement with audio signals.

**Interface**:
```python
class AudioVisualSync:
    def identify_primary_speaker(self, faces: List[FaceRegion], 
                                audio_frame: AudioFrame) -> Optional[int]
    
class FaceRegion:
    bounding_box: Tuple[int, int, int, int]
    landmarks: FaceLandmarks
    face_id: int
```

**Key Behaviors**:
- Extracts lip movement features from facial landmarks (mouth region)
- Computes correlation between lip movement and audio energy/phonemes
- Returns the face_id with highest audio-visual correlation
- Handles cases where no face correlates with audio (returns None)
- Used by Visual Analysis Module to filter to primary speaker before emotion classification

### 7. Sentiment Display Interface

**Responsibility**: Visualizes real-time sentiment scores and analysis details.

**Interface**:
```python
class SentimentDisplay:
    def update_score(self, sentiment: SentimentScore) -> None
    def show_history(self, duration_seconds: int) -> None
    def show_modality_breakdown(self) -> None
```

**Key Behaviors**:
- Displays current sentiment score as a gauge or line chart
- Shows individual modality contributions (acoustic, visual, linguistic)
- Plots sentiment history over time
- Highlights significant emotional shifts
- Provides session summary on stream end

## Data Models

### AudioFrame
```python
@dataclass
class AudioFrame:
    samples: np.ndarray  # PCM audio samples
    sample_rate: int     # e.g., 16000 Hz
    timestamp: float     # seconds since stream start
    duration: float      # frame duration in seconds
```

### VideoFrame
```python
@dataclass
class VideoFrame:
    image: np.ndarray    # RGB image (H, W, 3)
    timestamp: float     # seconds since stream start
    frame_number: int
```

### AcousticFeatures
```python
@dataclass
class AcousticFeatures:
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    speaking_rate: float  # syllables per second
    spectral_centroid: float
    zero_crossing_rate: float
```

### FaceLandmarks
```python
@dataclass
class FaceLandmarks:
    points: np.ndarray   # (N, 2) array of (x, y) coordinates
    confidence: float
```

### StreamConnection
```python
@dataclass
class StreamConnection:
    url: str
    protocol: StreamProtocol
    is_active: bool
    audio_codec: str
    video_codec: str
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Acoustic feature extraction completeness
*For any* audio frame with valid audio content, the Acoustic Analysis Module should produce an AcousticResult containing all required features (pitch, energy, speaking rate, spectral features) and emotion scores.
**Validates: Requirements 1.2, 3.1, 3.2**

### Property 2: Visual feature extraction completeness
*For any* video frame, the Visual Analysis Module should attempt face detection and, when a face is detected, produce a VisualResult containing facial landmarks and emotion scores.
**Validates: Requirements 1.3, 4.1, 4.2, 4.3**

### Property 3: Linguistic analysis completeness
*For any* audio frame containing speech, the Linguistic Analysis Module should produce a LinguisticResult containing transcription text, emotion scores, and confidence values.
**Validates: Requirements 1.4, 5.1, 5.2, 5.3**

### Property 4: Multi-speaker audio handling
*For any* audio sequence containing multiple speakers, the Acoustic Analysis Module should detect speaker transitions and maintain analysis continuity across the transition.
**Validates: Requirements 3.4**

### Property 5: Multi-face video handling
*For any* video frame containing multiple faces, the Visual Analysis Module should identify and analyze the primary speaker without failing or producing invalid results.
**Validates: Requirements 4.4**

### Property 6: Fusion score normalization
*For any* combination of acoustic, visual, and linguistic results, the Fusion Engine should produce a sentiment score in the range [-1.0, 1.0].
**Validates: Requirements 6.4**

### Property 7: Quality-weighted fusion
*For any* two modalities A and B where confidence_A ≥ 2 × confidence_B (and all other inputs are neutral), the final sentiment score should reflect at least 70% of Modality A's score contribution.
**Validates: Requirements 6.1, 6.2**

### Property 8: Conflict resolution in fusion
*For any* set of modality results with conflicting sentiment (e.g., positive acoustic, negative visual), the Fusion Engine should produce a valid sentiment score and report a confidence level that reflects the disagreement.
**Validates: Requirements 6.3**

### Property 9: Temporal smoothing preservation
*For any* sequence of sentiment scores with small random fluctuations, applying temporal smoothing should reduce the variance while preserving the overall trend and genuine large shifts.
**Validates: Requirements 6.5**

### Property 10: Display output structure
*For any* sentiment score generated by the Fusion Engine, the display output should contain the score, timestamp, and individual modality contributions (acoustic, visual, linguistic).
**Validates: Requirements 7.1, 7.2**

### Property 11: Historical data retrieval
*For any* sequence of sentiment scores generated during a session, requesting historical data should return all scores in chronological order with no missing entries.
**Validates: Requirements 7.3**

### Property 12: Stream format decoding
*For any* supported stream format and codec combination, the Stream Input Manager should successfully decode the stream and produce valid audio and video frames.
**Validates: Requirements 8.2**

### Property 13: Adaptive quality processing
*For any* stream with varying quality levels, the system should continue producing sentiment scores without crashing, and quality indicators should reflect the degradation.
**Validates: Requirements 8.3**

### Property 14: State reset on source change
*For any* active analysis session, changing the stream source should clear all accumulated state (history, temporal context) and begin fresh analysis with the new source.
**Validates: Requirements 8.5**

### Property 15: Domain-specific sentiment interpretation
*For any* known domain-specific term with strong sentiment (e.g., "systemic risk" in finance, "bear market"), the linguistic analysis should produce a sentiment score that reflects the domain-specific meaning regardless of surrounding neutral context.
**Validates: Requirements 5.5**

## Error Handling

### Stream Input Errors
- **Connection Failure**: Retry connection with exponential backoff (max 3 attempts), then report error to user
- **Decode Errors**: Log codec information, skip corrupted frames, continue processing
- **Stream Interruption**: Attempt reconnection, buffer recent frames to resume smoothly

### Analysis Module Errors
- **Acoustic Analysis Failures**: Report low confidence, continue with visual and linguistic only
- **Visual Analysis Failures**: Report low confidence when no face detected, continue with acoustic and linguistic
- **Linguistic Analysis Failures**: Report low confidence when transcription fails, continue with acoustic and visual
- **Model Loading Errors**: Fail fast at startup with clear error message about missing models

### Fusion Errors
- **Missing Modality Data**: Compute fusion with available modalities, adjust weights accordingly
- **Invalid Score Values**: Clamp scores to valid range [-1, 1], log warning
- **Confidence Calculation Errors**: Default to low confidence (0.3), continue processing

### General Error Handling Principles
- Fail gracefully: Never crash the entire pipeline due to single-frame errors
- Degrade gracefully: Continue with reduced functionality when modules fail
- Log comprehensively: Record all errors with context for debugging
- Report transparently: Communicate quality issues to users through confidence scores

## Testing Strategy

### Unit Testing Approach

Unit tests will verify specific behaviors and edge cases for individual components:

- **Stream Input Manager**: Test connection to different protocols, codec handling, frame extraction
- **Acoustic Analyzer**: Test feature extraction with clean audio, noisy audio, silence
- **Visual Analyzer**: Test face detection with single face, multiple faces, no faces, occlusions
- **Linguistic Analyzer**: Test transcription with clear speech, accented speech, background noise
- **Fusion Engine**: Test score computation with balanced inputs, conflicting inputs, missing modalities
- **Display Interface**: Test rendering with valid data, edge case scores (exactly -1, 0, 1)

Unit tests will use pytest and focus on concrete examples that demonstrate correct behavior.

### Property-Based Testing Approach

Property-based tests will verify universal properties across many randomly generated inputs using the Hypothesis library. Each correctness property defined above will be implemented as a property-based test.

**Property-Based Testing Configuration**:
- Minimum 100 iterations per property test to ensure thorough coverage
- Each property test will be tagged with a comment referencing the design document property
- Tag format: `# Feature: realtime-sentiment-analysis, Property {number}: {property_text}`

**Property Test Examples**:

1. **Property 1 Test**: Generate random audio frames (varying length, sample rate, content), verify AcousticResult contains all required fields
2. **Property 6 Test**: Generate random modality results with varying scores, verify fusion output is always in [-1, 1]
3. **Property 7 Test**: Generate modality results with one high-confidence and others low-confidence, verify high-confidence modality dominates
4. **Property 9 Test**: Generate noisy score sequences, verify smoothing reduces variance while preserving large shifts

**Test Data Generation**:
- Audio frames: Random PCM samples, varying durations (0.1s to 5s), sample rates (8kHz to 48kHz)
- Video frames: Random RGB images, varying resolutions, with/without synthetic faces
- Modality results: Random emotion scores, confidence values, quality indicators
- Score sequences: Random walks with controlled noise and trend

### Integration Testing

Integration tests will verify end-to-end pipeline behavior:

- Process a complete test video file and verify sentiment scores are generated
- Test pipeline with real-world video samples (news clips, presentations)
- Verify latency requirements with timed integration tests
- Test error recovery by simulating stream interruptions

### Performance Testing

Performance tests will verify latency requirements:

- Measure end-to-end latency from frame ingestion to sentiment output
- Profile each module to identify bottlenecks
- Test with varying stream qualities and resolutions
- Verify target latency of 1 second under normal conditions

## Implementation Notes

### Model Selection

For the MVP, we'll use pre-trained models to accelerate development:

- **Acoustic Emotion Recognition**: Use a pre-trained wav2vec2 model fine-tuned on emotion datasets (e.g., RAVDESS, IEMOCAP)
- **Facial Expression Recognition**: Use a pre-trained CNN model (e.g., FER2013-trained model or EfficientNet)
- **Speech-to-Text**: Use OpenAI Whisper (base or small model for speed/accuracy balance)
- **Text Sentiment Analysis**: Use a pre-trained transformer (e.g., DistilBERT fine-tuned on emotion datasets)

### Performance Optimization

To meet latency requirements:

- **Asynchronous Architecture**: Use asyncio and Redis Streams to prevent slow modules from blocking fast ones
- **Time-Windowed Fusion**: Fusion operates on fixed 1-second intervals, using latest available results rather than waiting for all modalities
- **GPU Acceleration**: Use GPU for model inference when available (especially for Whisper and visual models)
- **Frame Skipping**: Visual analysis processes every 2nd or 3rd frame to reduce computational load
- **Model Selection**: Use smaller, faster model variants (e.g., Whisper base, DistilBERT)
- **Adaptive Processing**: Linguistic analysis runs at lower frequency (every 2-3 seconds) due to computational cost
- **Result Caching**: Each module maintains a local cache of latest results for fast Fusion Engine access
- **Graceful Degradation**: Missing modality data doesn't block fusion; system continues with available modalities

### Extensibility Considerations

The architecture supports future enhancements:

- **Alert System (P1)**: Add AlertManager component that monitors score changes and triggers notifications
- **API Layer (P1)**: Add FastAPI REST/WebSocket endpoints wrapping the core pipeline
- **Additional Modalities**: Architecture supports adding new analysis modules (e.g., body language, text overlays)
- **Custom Models**: Interface design allows swapping pre-trained models with custom-trained versions
- **Multi-Stream Support**: Architecture can be extended to process multiple streams in parallel

