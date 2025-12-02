# Fusion Logic Structure

## Pure Function Design

The FusionEngine core logic MUST be implemented as pure functions that accept only data model inputs and return data model outputs:

```python
class FusionEngine:
    def fuse(
        self,
        acoustic: Optional[AcousticResult],
        visual: Optional[VisualResult],
        linguistic: Optional[LinguisticResult]
    ) -> SentimentScore:
        """Pure function: no I/O, no Redis, no side effects"""
        # Collect available modalities
        available = self._collect_available(acoustic, visual, linguistic)
        
        # Compute weighted fusion
        weights = self._compute_weights(available)
        raw_score = self._weighted_average(available, weights)
        
        # Apply conflict resolution
        resolved_score = self._resolve_conflicts(
            raw_score, acoustic, visual, linguistic
        )
        
        # Apply temporal smoothing
        smoothed_score = self._apply_smoothing(resolved_score)
        
        return SentimentScore(
            score=smoothed_score,
            confidence=self._compute_confidence(weights),
            modality_contributions=weights,
            emotion_breakdown=self._merge_emotions(available),
            timestamp=time.time()
        )
    
    def _compute_weights(self, ...) -> Dict[str, float]:
        """Pure function for weight calculation"""
        pass
    
    def _resolve_conflicts(self, ...) -> float:
        """Pure function for conflict resolution"""
        pass
    
    def _apply_smoothing(self, score: float) -> float:
        """Pure function for temporal smoothing"""
        # Uses self._history for EMA, but doesn't perform I/O
        pass
```

## Separation of Concerns

**Core Fusion Logic** (pure functions):
- `fuse()` - Main fusion algorithm
- `_compute_weights()` - Quality-aware weight calculation
- `_resolve_conflicts()` - Conflict resolution rules
- `_apply_smoothing()` - Temporal smoothing (EMA)

**Infrastructure Logic** (separate methods):
- `start()` - Starts the 1-second timer task
- `_fetch_latest_results()` - Queries Redis/cache for latest results
- `_publish_score()` - Publishes result to output queue

## Testing Benefits

Pure functions enable easy property-based testing:

```python
from hypothesis import given, strategies as st

# Feature: realtime-sentiment-analysis, Property 7: Quality-weighted fusion
@given(
    confidence_a=st.floats(min_value=0.8, max_value=1.0),
    confidence_b=st.floats(min_value=0.1, max_value=0.4)
)
def test_quality_weighted_fusion(confidence_a, confidence_b):
    acoustic = AcousticResult(
        emotion_scores={"positive": 0.9},
        confidence=confidence_a,
        timestamp=0.0
    )
    visual = VisualResult(
        emotion_scores={"neutral": 0.5},
        confidence=confidence_b,
        timestamp=0.0
    )
    
    engine = FusionEngine()
    score = engine.fuse(acoustic, visual, None)
    
    # High-confidence modality should dominate
    assert score.score > 0.6  # Reflects acoustic's positive sentiment
```

## Critical Rules

1. **No I/O in fusion logic** - Keep Redis, file access, and network calls separate
2. **Accept data models only** - All inputs should be typed dataclasses
3. **Return data models only** - Output should be SentimentScore dataclass
4. **Maintain state minimally** - Only store smoothing history, not raw frames

## Rationale

This ensures the core fusion logic (Task 9) remains testable (as required by Properties 7, 8, 9) and decoupled from the pipeline's infrastructure. Pure functions are easier to test, debug, and reason about, which is critical for the complex multi-modal fusion algorithm.
