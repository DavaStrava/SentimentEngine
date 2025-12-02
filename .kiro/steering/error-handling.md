# Error Handling Patterns

## Core Principles

1. **Fail Gracefully**: Never crash the entire pipeline due to single-frame errors
2. **Degrade Gracefully**: Continue with reduced functionality when modules fail
3. **Log Comprehensively**: Record all errors with context for debugging
4. **Report Transparently**: Communicate quality issues through confidence scores

## Analysis Module Error Pattern

All analysis modules should handle errors without blocking the pipeline:

```python
class AcousticAnalyzer:
    async def analyze_audio(self, audio_frame: AudioFrame) -> Optional[AcousticResult]:
        try:
            # Extract features
            features = self._extract_features(audio_frame)
            
            # Classify emotion
            emotion_scores = self._classify_emotion(features)
            
            return AcousticResult(
                emotion_scores=emotion_scores,
                confidence=self._compute_confidence(features),
                features=features,
                timestamp=time.time()
            )
        except AudioProcessingError as e:
            logger.warning(f"Audio processing failed: {e}")
            # Return low-confidence neutral result
            return AcousticResult(
                emotion_scores={"neutral": 1.0},
                confidence=0.1,  # Very low confidence
                features=None,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Unexpected error in acoustic analysis: {e}", exc_info=True)
            return None  # Fusion will handle missing modality
```

## Fusion Engine Error Pattern

The Fusion Engine must handle missing or invalid modality data:

```python
def fuse(
    self,
    acoustic: Optional[AcousticResult],
    visual: Optional[VisualResult],
    linguistic: Optional[LinguisticResult]
) -> SentimentScore:
    """Fuse modality results, handling missing data gracefully"""
    
    # Collect available modalities
    available = {}
    if acoustic and acoustic.confidence > 0.05:
        available['acoustic'] = acoustic
    if visual and visual.confidence > 0.05:
        available['visual'] = visual
    if linguistic and linguistic.confidence > 0.05:
        available['linguistic'] = linguistic
    
    # Handle case where no modalities are available
    if not available:
        logger.warning("No modalities available for fusion")
        return SentimentScore(
            score=0.0,
            confidence=0.0,
            modality_contributions={},
            emotion_breakdown={"neutral": 1.0},
            timestamp=time.time()
        )
    
    # Compute fusion with available modalities
    weights = self._compute_weights(available)
    score = self._weighted_average(available, weights)
    
    # Clamp score to valid range (defensive programming)
    score = max(-1.0, min(1.0, score))
    
    return SentimentScore(
        score=score,
        confidence=self._compute_confidence(weights),
        modality_contributions=weights,
        emotion_breakdown=self._merge_emotions(available),
        timestamp=time.time()
    )
```

## Stream Input Error Pattern

Handle connection and decoding errors with retry logic:

```python
class StreamInputManager:
    async def start_streaming(self) -> None:
        """Start streaming with automatic reconnection"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                await self._stream_loop()
                break  # Successful completion
            except ConnectionError as e:
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Connection failed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries reached, giving up")
                    raise
            except DecodeError as e:
                logger.warning(f"Frame decode error: {e}, skipping frame")
                continue  # Skip corrupted frame, continue processing
```

## Logging Standards

Use structured logging with appropriate levels:

```python
import logging

logger = logging.getLogger(__name__)

# ERROR: System failures that require attention
logger.error("Failed to load emotion model", exc_info=True)

# WARNING: Degraded functionality, but system continues
logger.warning(f"Low audio quality detected, confidence reduced to {confidence}")

# INFO: Normal operational messages
logger.info(f"Started acoustic analysis module")

# DEBUG: Detailed diagnostic information
logger.debug(f"Extracted features: pitch={pitch_mean}, energy={energy_mean}")
```

## Exception Hierarchy

Define custom exceptions for different error types:

```python
class SentimentEngineError(Exception):
    """Base exception for all sentiment engine errors"""
    pass

class StreamError(SentimentEngineError):
    """Errors related to stream input"""
    pass

class AnalysisError(SentimentEngineError):
    """Errors during analysis processing"""
    pass

class AudioProcessingError(AnalysisError):
    """Errors in acoustic analysis"""
    pass

class VisualProcessingError(AnalysisError):
    """Errors in visual analysis"""
    pass

class LinguisticProcessingError(AnalysisError):
    """Errors in linguistic analysis"""
    pass

class FusionError(SentimentEngineError):
    """Errors during fusion"""
    pass
```

## Model Loading Errors

Fail fast at startup for critical errors:

```python
def load_models():
    """Load all required models at startup"""
    try:
        acoustic_model = load_acoustic_model()
        visual_model = load_visual_model()
        linguistic_model = load_linguistic_model()
        return acoustic_model, visual_model, linguistic_model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.error("Please download required models before starting")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1)
```

## Confidence Score Guidelines

Use confidence scores to communicate quality issues:

- **1.0 - 0.8**: High quality, reliable result
- **0.8 - 0.5**: Moderate quality, usable result
- **0.5 - 0.2**: Low quality, use with caution
- **< 0.2**: Very low quality, likely error condition
- **0.0**: Complete failure, no valid result

The Fusion Engine should weight modalities based on these confidence levels.
