"""Base interfaces for analysis modules"""

from abc import ABC, abstractmethod
from typing import Optional
from src.models.frames import AudioFrame, VideoFrame
from src.models.results import AcousticResult, VisualResult, LinguisticResult


class AnalysisModule(ABC):
    """Base interface for all analysis modules"""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the analysis module (begin consuming from queue)"""
        pass
    
    @abstractmethod
    def get_latest_result(self):
        """Get the most recent analysis result
        
        Returns:
            Latest result or None if no result available
        """
        pass


class AcousticAnalyzerInterface(AnalysisModule):
    """Interface for acoustic analysis module"""
    
    @abstractmethod
    async def analyze_audio(self, audio_frame: AudioFrame) -> Optional[AcousticResult]:
        """Analyze an audio frame
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            Acoustic analysis result or None on failure
        """
        pass


class VisualAnalyzerInterface(AnalysisModule):
    """Interface for visual analysis module"""
    
    @abstractmethod
    async def analyze_frame(self, video_frame: VideoFrame) -> Optional[VisualResult]:
        """Analyze a video frame
        
        Args:
            video_frame: Video frame to analyze
            
        Returns:
            Visual analysis result or None on failure
        """
        pass


class LinguisticAnalyzerInterface(AnalysisModule):
    """Interface for linguistic analysis module"""
    
    @abstractmethod
    async def analyze_audio(self, audio_frame: AudioFrame) -> Optional[LinguisticResult]:
        """Analyze audio for linguistic content
        
        Args:
            audio_frame: Audio frame to transcribe and analyze
            
        Returns:
            Linguistic analysis result or None on failure
        """
        pass
