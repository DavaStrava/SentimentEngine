# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create Python project with virtual environment
  - Install core dependencies: asyncio, redis, opencv-python, pyav, librosa, transformers, whisper, mediapipe, numpy, streamlit, pytest, hypothesis
  - Set up project directory structure: src/, tests/, models/, config/
  - Create configuration file for model paths and parameters
  - _Requirements: All_

- [x] 2. Implement data models and core interfaces
  - Define AudioFrame, VideoFrame, AcousticFeatures, FaceLandmarks dataclasses
  - Define AcousticResult, VisualResult, LinguisticResult, SentimentScore dataclasses
  - Define StreamConnection and StreamProtocol enums
  - Create base interfaces for analysis modules
  - _Requirements: 1.2, 1.3, 1.4, 3.1, 4.1, 5.1, 6.1_

- [x] 2.1 Write property test for data model serialization
  - **Property: Data model round-trip consistency**
  - **Validates: Requirements 1.2, 1.3, 1.4**

- [x] 3. Implement Stream Input Manager
  - Create StreamInputManager class with connection handling
  - Implement stream decoding for local video files (MVP scope)
  - Implement audio frame extraction (PCM format, 16kHz sample rate)
  - Implement video frame extraction (RGB format)
  - Add frame timestamping
  - _Requirements: 1.1, 8.1, 8.2_

- [x] 3.1 Add Redis Streams integration for frame publishing
  - Set up Redis connection and stream creation
  - Implement async audio frame publishing to Redis
  - Implement async video frame publishing to Redis
  - Add frame serialization for Redis storage
  - _Requirements: 1.1, 9.1_

- [x] 3.2 Write unit tests for Stream Input Manager
  - Test connection to local video file
  - Test audio frame extraction and format
  - Test video frame extraction and format
  - Test frame timestamping accuracy
  - _Requirements: 1.1, 8.2_

- [x] 4. Implement Acoustic Analysis Module
  - Create AcousticAnalyzer class with Redis consumer
  - Implement audio feature extraction using librosa (pitch, energy, speaking rate, spectral features)
  - Integrate pre-trained emotion recognition model (wav2vec2 or similar)
  - Implement emotion classification with confidence scoring
  - Add result caching with timestamps
  - _Requirements: 1.2, 3.1, 3.2_

- [x] 4.1 Add noise filtering and quality indicators
  - Implement spectral subtraction for noise filtering
  - Add audio quality assessment
  - Report quality indicators in AcousticResult
  - Adjust confidence based on audio quality
  - _Requirements: 3.3, 3.5_

- [x] 4.2 Write property test for acoustic feature extraction
  - **Property 1: Acoustic feature extraction completeness**
  - **Validates: Requirements 1.2, 3.1, 3.2**

- [x] 4.3 Write property test for acoustic emotion classification
  - **Property: Acoustic emotion scores structure**
  - **Validates: Requirements 3.2**

- [x] 5. Implement Visual Analysis Module
  - Create VisualAnalyzer class with Redis consumer
  - Implement face detection using MediaPipe or OpenCV
  - Implement facial landmark extraction
  - Integrate pre-trained facial expression recognition model
  - Implement emotion classification with confidence scoring
  - Add result caching with timestamps
  - Implement frame skipping (process every 2nd or 3rd frame)
  - _Requirements: 1.3, 4.1, 4.2, 4.3_

- [x] 5.1 Add quality indicators for visual analysis
  - Detect face occlusion using landmark visibility
  - Assess lighting quality from image statistics
  - Report quality indicators in VisualResult
  - Adjust confidence based on visual quality
  - _Requirements: 4.5_

- [x] 5.2 Write property test for visual feature extraction
  - **Property 2: Visual feature extraction completeness**
  - **Validates: Requirements 1.3, 4.1, 4.2, 4.3**

- [ ] 6. Implement Audio-Visual Synchronization Component
  - Create AudioVisualSync class
  - Extract lip movement features from facial landmarks
  - Compute audio energy and phoneme features
  - Implement correlation algorithm between lip movement and audio
  - Return primary speaker face_id
  - _Requirements: 4.4_

- [ ] 6.1 Integrate audio-visual sync into Visual Analysis Module
  - Call AudioVisualSync when multiple faces detected
  - Filter to primary speaker before emotion classification
  - Handle cases where no face correlates with audio
  - _Requirements: 4.4_

- [ ]* 6.2 Write property test for multi-face handling
  - **Property 5: Multi-face video handling**
  - **Validates: Requirements 4.4**

- [x] 7. Implement Linguistic Analysis Module
  - Create LinguisticAnalyzer class with Redis consumer
  - Implement audio buffering (sliding window of 3-5 seconds)
  - Integrate Whisper for speech-to-text transcription
  - Integrate transformer model for sentiment analysis (DistilBERT)
  - Implement emotion classification with confidence scoring
  - Add result caching with timestamps
  - Implement lower-frequency processing (every 2-3 seconds)
  - _Requirements: 1.4, 5.1, 5.2, 5.3_

- [x] 7.1 Add transcription quality indicators
  - Report transcription confidence from Whisper
  - Adjust linguistic sentiment confidence based on transcription quality
  - Handle low-confidence transcriptions gracefully
  - _Requirements: 5.4_

- [ ]* 7.2 Write property test for linguistic analysis completeness
  - **Property 3: Linguistic analysis completeness**
  - **Validates: Requirements 1.4, 5.1, 5.2, 5.3**

- [ ]* 7.3 Write property test for domain-specific sentiment
  - **Property 15: Domain-specific sentiment interpretation**
  - **Validates: Requirements 5.5**

- [ ] 8. Checkpoint - Ensure all analysis modules work independently
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Fusion Engine
  - Create FusionEngine class with 1-second timer
  - Implement result querying from analysis module caches
  - Implement confidence-weighted fusion formula
  - Handle missing modality data (use last known value with reduced weight)
  - Normalize scores to [-1, 1] range
  - _Requirements: 1.5, 6.1, 6.4_

- [x] 9.1 Add conflict resolution and temporal smoothing
  - Implement conflict resolution rules (reduce outlier weight when two modalities agree)
  - Apply exponential moving average for temporal smoothing (α=0.3)
  - Report confidence levels that reflect modality agreement
  - Maintain emotion category breakdown
  - _Requirements: 6.2, 6.3, 6.5_

- [ ]* 9.2 Write property test for fusion score normalization
  - **Property 6: Fusion score normalization**
  - **Validates: Requirements 6.4**

- [ ]* 9.3 Write property test for quality-weighted fusion
  - **Property 7: Quality-weighted fusion**
  - **Validates: Requirements 6.1, 6.2**

- [ ]* 9.4 Write property test for conflict resolution
  - **Property 8: Conflict resolution in fusion**
  - **Validates: Requirements 6.3**

- [ ]* 9.5 Write property test for temporal smoothing
  - **Property 9: Temporal smoothing preservation**
  - **Validates: Requirements 6.5**

- [x] 10. Implement Sentiment Display Interface
  - Create SentimentDisplay class using Streamlit
  - Display current sentiment score as gauge or line chart
  - Show individual modality contributions (acoustic, visual, linguistic)
  - Plot sentiment history over time
  - Display emotion category breakdown
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 10.1 Add session summary functionality
  - Store sentiment scores throughout session
  - Generate summary report on stream end
  - Identify significant emotional shifts in summary
  - Display summary in UI
  - _Requirements: 7.5_

- [ ]* 10.2 Write property test for display output structure
  - **Property 10: Display output structure**
  - **Validates: Requirements 7.1, 7.2**

- [ ]* 10.3 Write property test for historical data retrieval
  - **Property 11: Historical data retrieval**
  - **Validates: Requirements 7.3**

- [x] 11. Implement main orchestration and async coordination
  - Create main application entry point
  - Initialize Redis connection
  - Start Stream Input Manager in asyncio task
  - Start all three analysis modules in separate asyncio tasks
  - Start Fusion Engine timer task
  - Start Sentiment Display in main thread
  - Handle graceful shutdown
  - _Requirements: 1.1, 1.5, 9.1_

- [x] 11.1 Add error handling and reconnection logic
  - Implement stream reconnection on interruption
  - Handle analysis module failures gracefully (continue with other modalities)
  - Log errors comprehensively
  - Add performance monitoring and latency logging
  - _Requirements: 8.4, 9.3_

- [ ]* 11.2 Write property test for state reset on source change
  - **Property 14: State reset on source change**
  - **Validates: Requirements 8.5**

- [ ] 12. Add stream format support and adaptive processing
  - Extend Stream Input Manager to support multiple codecs
  - Implement adaptive processing for varying stream quality
  - Add quality indicators throughout pipeline
  - _Requirements: 8.2, 8.3_

- [ ]* 12.1 Write property test for stream format decoding
  - **Property 12: Stream format decoding**
  - **Validates: Requirements 8.2**

- [ ]* 12.2 Write property test for adaptive quality processing
  - **Property 13: Adaptive quality processing**
  - **Validates: Requirements 8.3**

- [ ] 13. Integration testing and performance validation
  - Test end-to-end pipeline with sample video files
  - Measure and validate latency requirements (target: 1 second, max: 3 seconds)
  - Test with varying video qualities and formats
  - Verify all modalities contribute to final score
  - Test error recovery scenarios
  - _Requirements: 9.1, 9.4_

- [ ] 14. Final checkpoint - Ensure all tests pass and system meets requirements
  - Ensure all tests pass, ask the user if questions arise.
