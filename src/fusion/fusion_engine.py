"""Fusion Engine

This module combines multi-modal signals (acoustic, visual, linguistic) into a unified
sentiment score. It operates on fixed time intervals, applying quality-aware weighting,
conflict resolution, and temporal smoothing.

Requirements:
    - Req 1.5: Fusion Engine generates unified Sentiment Score at least once per second
    - Req 6.1: System computes weighted combination based on signal quality
    - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
    - Req 6.3: System applies conflict resolution rules and reports confidence levels
    - Req 6.4: System normalizes score to consistent range [-1, 1]
    - Req 6.5: System applies smoothing to reduce noise while preserving emotional shifts
"""

import logging
import time
import asyncio
from typing import Optional, Dict, List
import numpy as np

from src.models.results import AcousticResult, VisualResult, LinguisticResult, SentimentScore
from src.config.config_loader import config


logger = logging.getLogger(__name__)


class FusionError(Exception):
    """Exception raised for errors during fusion"""
    pass


class FusionEngine:
    """Combines multi-modal signals into unified sentiment scores.
    
    This class implements the fusion engine that:
    1. Operates on fixed 1-second timer intervals
    2. Queries latest results from analysis module caches
    3. Computes quality-aware weighted fusion
    4. Applies conflict resolution when modalities disagree
    5. Applies temporal smoothing using exponential moving average
    6. Normalizes scores to [-1, 1] range
    7. Maintains emotion category breakdown
    
    The core fusion logic is implemented as pure functions that accept only
    data model inputs and return data model outputs, enabling easy testing
    and reasoning about the fusion algorithm.
    
    Attributes:
        baseline_weights: Default weights for each modality
        smoothing_alpha: EMA smoothing factor (0.3 = 30% new, 70% history)
        conflict_threshold: Threshold for detecting conflicting modalities
        outlier_weight_reduction: Factor to reduce outlier weight (0.5 = 50% reduction)
        score_history: History of scores for temporal smoothing
        latest_score: Most recent sentiment score (cached)
    """
    
    def __init__(self, acoustic_analyzer=None, visual_analyzer=None, linguistic_analyzer=None):
        """Initialize the fusion engine with configuration and analyzer references.
        
        Args:
            acoustic_analyzer: Reference to AcousticAnalyzer for result access
            visual_analyzer: Reference to VisualAnalyzer for result access
            linguistic_analyzer: Reference to LinguisticAnalyzer for result access
        """
        # Configuration
        self.timer_interval = config.get('fusion.timer_interval', 1.0)
        self.baseline_weights = config.get('fusion.baseline_weights', {
            'acoustic': 0.33,
            'visual': 0.33,
            'linguistic': 0.34
        })
        self.smoothing_alpha = config.get('fusion.smoothing_alpha', 0.3)
        self.conflict_threshold = config.get('fusion.conflict_threshold', 0.5)
        self.outlier_weight_reduction = config.get('fusion.outlier_weight_reduction', 0.5)
        
        # Analyzer references
        self.acoustic_analyzer = acoustic_analyzer
        self.visual_analyzer = visual_analyzer
        self.linguistic_analyzer = linguistic_analyzer
        
        # State
        self.score_history: List[float] = []
        self.latest_score: Optional[SentimentScore] = None
        
        logger.info(f"FusionEngine initialized with timer_interval={self.timer_interval}s, "
                   f"smoothing_alpha={self.smoothing_alpha}")
    
    def _collect_available(
        self,
        acoustic: Optional[AcousticResult],
        visual: Optional[VisualResult],
        linguistic: Optional[LinguisticResult]
    ) -> Dict[str, tuple]:
        """Collect available modality results with sufficient confidence.
        
        Pure function that filters modalities based on confidence threshold (0.05).
        This implements graceful degradation by excluding low-quality modality results
        from the fusion process, allowing the system to continue with remaining
        high-quality modalities.
        
        Args:
            acoustic: Acoustic analysis result or None if not available or failed
            visual: Visual analysis result or None if not available or failed
            linguistic: Linguistic analysis result or None if not available or failed
            
        Returns:
            Dictionary mapping modality names (str) to (result, confidence) tuples.
            Only includes modalities with confidence > 0.05. Empty dict if no
            modalities meet the threshold.
            Example: {"acoustic": (AcousticResult(...), 0.85), "visual": (VisualResult(...), 0.72)}
            
        Validates:
            - Req 6.1: System computes weighted combination based on signal quality
            - Design: Graceful degradation when modalities fail or produce low-quality results
        """
        available = {}
        
        if acoustic and acoustic.confidence > 0.05:
            available['acoustic'] = (acoustic, acoustic.confidence)
        
        if visual and visual.confidence > 0.05:
            available['visual'] = (visual, visual.confidence)
        
        if linguistic and linguistic.confidence > 0.05:
            available['linguistic'] = (linguistic, linguistic.confidence)
        
        return available
    
    def _compute_weights(self, available: Dict[str, tuple]) -> Dict[str, float]:
        """Compute quality-aware weights for each modality.
        
        Pure function that computes weights based on confidence and baseline weights.
        Formula: weight_m = modality_confidence_m * baseline_weight_m
        
        Args:
            available: Dictionary of available modalities with (result, confidence) tuples
            
        Returns:
            Dictionary mapping modality names to normalized weights
            
        Validates:
            - Req 6.1: System computes weighted combination based on signal quality
            - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
        """
        if not available:
            return {}
        
        # Compute raw weights
        raw_weights = {}
        for modality, (result, confidence) in available.items():
            baseline = self.baseline_weights.get(modality, 0.33)
            raw_weights[modality] = confidence * baseline
        
        # Normalize weights to sum to 1.0
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in raw_weights.items()}
        else:
            # Fallback to equal weights
            n = len(available)
            normalized_weights = {k: 1.0 / n for k in available.keys()}
        
        return normalized_weights
    
    def _emotion_to_score(self, emotion_scores: Dict[str, float]) -> float:
        """Convert emotion scores to sentiment score in [-1, 1] range.
        
        Pure function that maps emotion categories to sentiment polarity using
        a predefined polarity mapping. This enables consistent sentiment scoring
        across all three modalities (acoustic, visual, linguistic) by converting
        their emotion classifications to a unified sentiment scale.
        
        Polarity mapping:
        - Positive emotions: happy (1.0), surprised (0.5)
        - Neutral: neutral (0.0)
        - Negative emotions: fearful (-0.3), disgust (-0.6), sad (-0.7), angry (-0.8)
        
        The final score is computed as a weighted average of polarities using the
        emotion scores as weights, then clamped to [-1, 1] range.
        
        Args:
            emotion_scores: Dictionary mapping emotion names (str) to probability
                           scores (float in [0, 1]). Example: {"happy": 0.7, "sad": 0.2, "neutral": 0.1}
            
        Returns:
            Sentiment score in [-1, 1] range where -1 is very negative, 0 is neutral,
            and 1 is very positive. Computed as weighted average of emotion polarities.
            
        Validates:
            - Req 6.4: System normalizes score to consistent range [-1, 1]
            - Prop 6: Fusion score normalization
        """
        # Emotion polarity mapping
        polarity_map = {
            'happy': 1.0,
            'surprised': 0.5,
            'neutral': 0.0,
            'fearful': -0.3,
            'sad': -0.7,
            'angry': -0.8,
            'disgust': -0.6
        }
        
        # Compute weighted average of polarities
        score = 0.0
        for emotion, emotion_score in emotion_scores.items():
            polarity = polarity_map.get(emotion, 0.0)
            score += polarity * emotion_score
        
        # Clamp to [-1, 1]
        score = max(-1.0, min(1.0, score))
        
        return score
    
    def _weighted_average(
        self,
        available: Dict[str, tuple],
        weights: Dict[str, float]
    ) -> float:
        """Compute weighted average of modality scores.
        
        Pure function that combines modality scores using quality-aware weights.
        Each modality's emotion scores are first converted to a sentiment score
        in [-1, 1] range, then combined using the normalized weights computed
        from confidence levels and baseline weights.
        
        Formula: score = Σ(weight_m * sentiment_score_m) for all modalities m
        
        Args:
            available: Dictionary of available modalities with (result, confidence) tuples.
                      Keys are modality names ("acoustic", "visual", "linguistic").
                      Values are tuples of (analysis_result, confidence_score).
            weights: Dictionary of normalized weights for each modality (sum to 1.0).
                    Keys match modality names in available dict.
                    Example: {"acoustic": 0.4, "visual": 0.35, "linguistic": 0.25}
            
        Returns:
            Weighted average sentiment score in [-1, 1] range, clamped to ensure
            valid output. Returns 0.0 if no modalities available or weights empty.
            
        Validates:
            - Req 6.1: System computes weighted combination based on signal quality
            - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
            - Req 6.4: System normalizes score to consistent range [-1, 1]
            - Prop 7: Quality-weighted fusion
        """
        if not available or not weights:
            return 0.0
        
        weighted_sum = 0.0
        for modality, (result, confidence) in available.items():
            weight = weights.get(modality, 0.0)
            
            # Convert emotion scores to sentiment score
            modality_score = self._emotion_to_score(result.emotion_scores)
            
            weighted_sum += weight * modality_score
        
        # Clamp to [-1, 1]
        weighted_sum = max(-1.0, min(1.0, weighted_sum))
        
        return weighted_sum
    
    def _resolve_conflicts(
        self,
        raw_score: float,
        acoustic: Optional[AcousticResult],
        visual: Optional[VisualResult],
        linguistic: Optional[LinguisticResult]
    ) -> float:
        """Apply conflict resolution rules when modalities disagree.
        
        Pure function that detects conflicts and adjusts the score accordingly.
        When two modalities agree and one disagrees, reduce the outlier's weight.
        
        Args:
            raw_score: Initial weighted average score
            acoustic: Acoustic analysis result or None
            visual: Visual analysis result or None
            linguistic: Linguistic analysis result or None
            
        Returns:
            Conflict-resolved sentiment score
            
        Validates:
            - Req 6.3: System applies conflict resolution rules
        """
        # Get individual modality scores
        scores = []
        if acoustic and acoustic.confidence > 0.05:
            scores.append(('acoustic', self._emotion_to_score(acoustic.emotion_scores)))
        if visual and visual.confidence > 0.05:
            scores.append(('visual', self._emotion_to_score(visual.emotion_scores)))
        if linguistic and linguistic.confidence > 0.05:
            scores.append(('linguistic', self._emotion_to_score(linguistic.emotion_scores)))
        
        # Need at least 3 modalities for conflict detection
        if len(scores) < 3:
            return raw_score
        
        # Check for conflicts (one modality significantly different from others)
        score_values = [s[1] for s in scores]
        mean_score = np.mean(score_values)
        std_score = np.std(score_values)
        
        # If standard deviation is high, there's conflict
        if std_score > self.conflict_threshold:
            # Find outlier (furthest from mean)
            outlier_idx = np.argmax([abs(s - mean_score) for s in score_values])
            outlier_name = scores[outlier_idx][0]
            
            logger.debug(f"Conflict detected: {outlier_name} is outlier (std={std_score:.3f})")
            
            # Recompute score with reduced outlier weight
            available = {}
            weights = {}
            total_weight = 0.0
            
            for name, score in scores:
                if name == outlier_name:
                    # Reduce outlier weight
                    weight = self.baseline_weights.get(name, 0.33) * self.outlier_weight_reduction
                else:
                    weight = self.baseline_weights.get(name, 0.33)
                
                weights[name] = weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Recompute weighted average
            resolved_score = sum(weights.get(name, 0.0) * score for name, score in scores)
            resolved_score = max(-1.0, min(1.0, resolved_score))
            
            return resolved_score
        
        return raw_score
    
    def _apply_smoothing(self, score: float) -> float:
        """Apply exponential moving average for temporal smoothing.
        
        Pure function (with minimal state) that smooths scores over time.
        Formula: smoothed = alpha * new_score + (1 - alpha) * previous_score
        
        Args:
            score: New sentiment score
            
        Returns:
            Smoothed sentiment score
            
        Validates:
            - Req 6.5: System applies smoothing to reduce noise while preserving shifts
        """
        if not self.score_history:
            # First score, no smoothing
            self.score_history.append(score)
            return score
        
        # Get previous smoothed score
        previous_score = self.score_history[-1]
        
        # Apply EMA: alpha * new + (1 - alpha) * old
        smoothed_score = self.smoothing_alpha * score + (1 - self.smoothing_alpha) * previous_score
        
        # Clamp to [-1, 1]
        smoothed_score = max(-1.0, min(1.0, smoothed_score))
        
        # Update history
        self.score_history.append(smoothed_score)
        
        # Keep history bounded (last 100 scores)
        if len(self.score_history) > 100:
            self.score_history = self.score_history[-100:]
        
        return smoothed_score
    
    def _compute_confidence(
        self,
        weights: Dict[str, float],
        acoustic: Optional[AcousticResult] = None,
        visual: Optional[VisualResult] = None,
        linguistic: Optional[LinguisticResult] = None
    ) -> float:
        """Compute overall confidence based on modality weights and agreement.
        
        Pure function that computes confidence from weight distribution and
        sentiment agreement across modalities.
        
        Args:
            weights: Dictionary of normalized weights for each modality
            acoustic: Acoustic analysis result or None
            visual: Visual analysis result or None
            linguistic: Linguistic analysis result or None
            
        Returns:
            Confidence score in [0, 1] range
            
        Validates:
            - Req 6.3: System reports confidence levels that reflect disagreement
        """
        if not weights:
            return 0.0
        
        # Confidence is higher when:
        # 1. More modalities are available
        # 2. Weights are more evenly distributed (less conflict)
        # 3. Modalities agree on sentiment direction
        
        num_modalities = len(weights)
        max_modalities = 3
        
        # Modality availability factor
        availability_factor = num_modalities / max_modalities
        
        # Weight distribution factor (entropy-based)
        # Higher entropy = more even distribution = higher confidence
        weight_values = list(weights.values())
        entropy = -sum(w * np.log(w + 1e-10) for w in weight_values)
        max_entropy = np.log(num_modalities) if num_modalities > 0 else 1.0
        distribution_factor = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Agreement factor: Check if modalities agree on sentiment direction
        # Get sentiment scores from each modality
        scores = []
        if acoustic and acoustic.confidence > 0.05:
            scores.append(self._emotion_to_score(acoustic.emotion_scores))
        if visual and visual.confidence > 0.05:
            scores.append(self._emotion_to_score(visual.emotion_scores))
        if linguistic and linguistic.confidence > 0.05:
            scores.append(self._emotion_to_score(linguistic.emotion_scores))
        
        # Compute agreement factor based on standard deviation of scores
        if len(scores) >= 2:
            std_score = np.std(scores)
            # High std = low agreement = low confidence
            # std ranges from 0 (perfect agreement) to ~1.0 (extreme disagreement)
            # Map std to agreement factor: 0 std -> 1.0 agreement, 1.0 std -> 0.0 agreement
            agreement_factor = max(0.0, 1.0 - std_score)
        else:
            # Only one modality, no disagreement possible
            agreement_factor = 1.0
        
        # Combined confidence with agreement factor weighted more heavily
        # When modalities disagree significantly, confidence should be low
        confidence = (
            0.3 * availability_factor +
            0.2 * distribution_factor +
            0.5 * agreement_factor
        )
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _merge_emotions(self, available: Dict[str, tuple]) -> Dict[str, float]:
        """Merge emotion breakdowns from all modalities.
        
        Pure function that combines emotion scores across modalities using
        confidence-weighted averaging. Each modality's emotion scores are weighted
        by its confidence level, then normalized to produce a unified emotion
        breakdown that represents the combined assessment from all modalities.
        
        This provides transparency into which emotions are driving the overall
        sentiment score, enabling users to understand the emotional composition
        beyond just the unified sentiment value.
        
        Args:
            available: Dictionary of available modalities with (result, confidence) tuples.
                      Keys are modality names ("acoustic", "visual", "linguistic").
                      Values are tuples of (analysis_result, confidence_score).
                      Each result contains emotion_scores dict.
            
        Returns:
            Dictionary mapping emotion names (str) to merged probability scores (float).
            Scores are normalized to sum to 1.0, representing a probability distribution
            over emotion categories. Returns {"neutral": 1.0} if no modalities available.
            Example: {"happy": 0.6, "neutral": 0.25, "sad": 0.15}
            
        Validates:
            - Req 7.2: System shows individual contributions from acoustic, visual, and linguistic analysis
            - Design: Emotion category breakdown for transparency
        """
        if not available:
            return {"neutral": 1.0}
        
        # Collect all emotion scores weighted by modality confidence
        merged_emotions = {}
        total_confidence = sum(conf for _, conf in available.values())
        
        for modality, (result, confidence) in available.items():
            weight = confidence / total_confidence if total_confidence > 0 else 0.0
            
            for emotion, score in result.emotion_scores.items():
                if emotion not in merged_emotions:
                    merged_emotions[emotion] = 0.0
                merged_emotions[emotion] += weight * score
        
        # Normalize to sum to 1.0
        total = sum(merged_emotions.values())
        if total > 0:
            merged_emotions = {k: v / total for k, v in merged_emotions.items()}
        
        return merged_emotions
    
    def fuse(
        self,
        acoustic: Optional[AcousticResult],
        visual: Optional[VisualResult],
        linguistic: Optional[LinguisticResult]
    ) -> SentimentScore:
        """Fuse multi-modal signals into unified sentiment score.
        
        This is the main fusion method that orchestrates the complete fusion pipeline.
        It is implemented as a pure function that accepts only data model inputs and
        returns a data model output, enabling easy testing and reasoning.
        
        The fusion pipeline:
        1. Collects available modalities with sufficient confidence
        2. Computes quality-aware weights for each modality (Req 6.1, 6.2)
        3. Computes weighted average of modality scores
        4. Applies conflict resolution when modalities disagree (Req 6.3)
        5. Applies temporal smoothing using EMA (Req 6.5)
        6. Normalizes score to [-1, 1] range (Req 6.4)
        7. Computes overall confidence
        8. Merges emotion breakdowns
        
        Args:
            acoustic: Acoustic analysis result or None
            visual: Visual analysis result or None
            linguistic: Linguistic analysis result or None
            
        Returns:
            SentimentScore with unified score, confidence, modality contributions,
            emotion breakdown, and timestamp
            
        Validates:
            - Req 6.1: System computes weighted combination based on signal quality
            - Req 6.2: System adjusts weights dynamically to favor higher-quality signals
            - Req 6.3: System applies conflict resolution rules and reports confidence
            - Req 6.4: System normalizes score to consistent range [-1, 1]
            - Req 6.5: System applies smoothing to reduce noise while preserving shifts
            - Prop 6: Fusion score normalization
            - Prop 7: Quality-weighted fusion
            - Prop 8: Conflict resolution in fusion
            - Prop 9: Temporal smoothing preservation
        """
        try:
            # Collect available modalities
            available = self._collect_available(acoustic, visual, linguistic)
            
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
            
            # Compute weights
            weights = self._compute_weights(available)
            
            # Compute weighted average
            raw_score = self._weighted_average(available, weights)
            
            # Apply conflict resolution
            resolved_score = self._resolve_conflicts(raw_score, acoustic, visual, linguistic)
            
            # Apply temporal smoothing
            smoothed_score = self._apply_smoothing(resolved_score)
            
            # Compute confidence (considering agreement across modalities)
            confidence = self._compute_confidence(weights, acoustic, visual, linguistic)
            
            # Merge emotion breakdowns
            emotion_breakdown = self._merge_emotions(available)
            
            # Create sentiment score
            sentiment_score = SentimentScore(
                score=smoothed_score,
                confidence=confidence,
                modality_contributions=weights,
                emotion_breakdown=emotion_breakdown,
                timestamp=time.time()
            )
            
            # Cache result
            self.latest_score = sentiment_score
            
            logger.debug(f"Fusion complete: score={smoothed_score:.3f}, "
                        f"confidence={confidence:.3f}, "
                        f"modalities={list(weights.keys())}")
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Fusion error: {e}", exc_info=True)
            # Return neutral score on error
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                modality_contributions={},
                emotion_breakdown={"neutral": 1.0},
                timestamp=time.time()
            )
    
    def get_latest_score(self) -> Optional[SentimentScore]:
        """Get the most recent sentiment score from cache.
        
        Provides non-blocking access to the latest fused sentiment score for
        consumption by the UI layer or external APIs. This method is called
        by the display interface to retrieve scores without waiting for the
        next fusion cycle to complete.
        
        Returns:
            Latest SentimentScore containing unified score, confidence, modality
            contributions, emotion breakdown, and timestamp. Returns None if no
            fusion has been performed yet (e.g., at system startup before first
            fusion cycle completes).
            
        Validates:
            - Req 7.1: System displays sentiment score with timestamp
            - Design: Result caching with timestamps for non-blocking access
        """
        return self.latest_score
    
    async def start(self):
        """Start the fusion timer task.
        
        This method runs as an independent asyncio task and periodically (every 1 second)
        queries the latest results from all analysis modules, fuses them, and caches
        the result. It implements the time-windowed fusion architecture that enables
        non-blocking operation.
        
        The method:
        1. Runs on fixed 1-second timer intervals
        2. Fetches latest results from acoustic, visual, and linguistic analyzers
        3. Calls fuse() to combine modality signals
        4. Caches result for UI/API access
        5. Handles missing modality data gracefully
        
        This task runs indefinitely until cancelled via asyncio.CancelledError,
        enabling clean shutdown of the fusion pipeline.
        
        Validates:
            - Req 1.5: Fusion Engine generates unified Sentiment Score at least once per second
            - Req 9.1: End-to-end latency not exceeding 3 seconds
            - Design: Time-windowed fusion on fixed 1-second intervals
            - Design: Non-blocking operation using cached results
        """
        try:
            logger.info(f"Starting fusion timer with interval={self.timer_interval}s")
            
            while True:
                try:
                    # Fetch latest results from analyzers
                    acoustic_result = self.acoustic_analyzer.get_latest_result() if self.acoustic_analyzer else None
                    visual_result = self.visual_analyzer.get_latest_result() if self.visual_analyzer else None
                    linguistic_result = self.linguistic_analyzer.get_latest_result() if self.linguistic_analyzer else None
                    
                    # Fuse results
                    sentiment_score = self.fuse(acoustic_result, visual_result, linguistic_result)
                    
                    # Wait for next interval
                    await asyncio.sleep(self.timer_interval)
                    
                except asyncio.CancelledError:
                    logger.info("Fusion engine task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in fusion cycle: {e}", exc_info=True)
                    await asyncio.sleep(self.timer_interval)
                    
        except Exception as e:
            logger.error(f"Fatal error in fusion engine: {e}", exc_info=True)
            raise
