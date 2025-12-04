"""Main Application Entry Point

This module orchestrates the complete sentiment analysis pipeline, coordinating
all analysis modules, fusion engine, and display interface.

Requirements:
    - Req 1.1: System begins processing within 2 seconds of stream initiation
    - Req 1.5: Fusion Engine generates unified Sentiment Score at least once per second
    - Req 9.1: End-to-end latency not exceeding 3 seconds
    - Req 8.4: System handles stream connection interruption
    - Req 9.3: System logs errors comprehensively
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from src.input.stream_manager import StreamInputManager
from src.analysis.acoustic import AcousticAnalyzer
from src.analysis.visual import VisualAnalyzer
from src.analysis.linguistic import LinguisticAnalyzer
from src.fusion.fusion_engine import FusionEngine
from src.ui.display import SentimentDisplay
from src.config.config_loader import config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sentiment_engine.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class SentimentEngine:
    """Main orchestrator for the sentiment analysis pipeline.
    
    This class coordinates all components of the system:
    1. Stream Input Manager (ingests and distributes frames)
    2. Acoustic Analysis Module (analyzes audio)
    3. Visual Analysis Module (analyzes video)
    4. Linguistic Analysis Module (transcribes and analyzes speech)
    5. Fusion Engine (combines modality signals)
    6. Sentiment Display (visualizes results)
    
    All analysis modules run as independent asyncio tasks to prevent blocking.
    The fusion engine operates on fixed 1-second intervals.
    
    Attributes:
        stream_manager: Manages stream input and frame distribution
        acoustic_analyzer: Analyzes audio frames
        visual_analyzer: Analyzes video frames
        linguistic_analyzer: Analyzes speech and text
        fusion_engine: Combines multi-modal signals
        display: Visualizes sentiment scores
        tasks: List of running asyncio tasks
    """
    
    def __init__(self):
        """Initialize the sentiment engine with all components."""
        logger.info("Initializing SentimentEngine...")
        
        # Initialize components
        self.stream_manager = StreamInputManager()
        self.acoustic_analyzer = AcousticAnalyzer()
        self.visual_analyzer = VisualAnalyzer()
        self.linguistic_analyzer = LinguisticAnalyzer()
        
        # Initialize fusion engine with analyzer references
        self.fusion_engine = FusionEngine(
            acoustic_analyzer=self.acoustic_analyzer,
            visual_analyzer=self.visual_analyzer,
            linguistic_analyzer=self.linguistic_analyzer
        )
        
        # Initialize display with fusion engine reference
        self.display = SentimentDisplay(fusion_engine=self.fusion_engine)
        
        # Task management
        self.tasks = []
        self.shutdown_event = asyncio.Event()
        
        logger.info("SentimentEngine initialized successfully")
    
    async def start_analysis_modules(self):
        """Start all analysis modules as independent asyncio tasks.
        
        Launches acoustic, visual, and linguistic analyzers as concurrent tasks
        that consume frames from Redis Streams and produce timestamped results.
        
        Validates:
            - Req 1.1: System begins processing within 2 seconds
            - Design: Asynchronous processing with independent tasks
        """
        logger.info("Starting analysis modules...")
        
        # Start acoustic analyzer
        acoustic_task = asyncio.create_task(
            self.acoustic_analyzer.start(),
            name="acoustic_analyzer"
        )
        self.tasks.append(acoustic_task)
        logger.info("Acoustic analyzer started")
        
        # Start visual analyzer
        visual_task = asyncio.create_task(
            self.visual_analyzer.start(),
            name="visual_analyzer"
        )
        self.tasks.append(visual_task)
        logger.info("Visual analyzer started")
        
        # Start linguistic analyzer
        linguistic_task = asyncio.create_task(
            self.linguistic_analyzer.start(),
            name="linguistic_analyzer"
        )
        self.tasks.append(linguistic_task)
        logger.info("Linguistic analyzer started")
        
        logger.info("All analysis modules started successfully")
    
    async def start_fusion_engine(self):
        """Start fusion engine timer task.
        
        Launches fusion engine that operates on fixed 1-second intervals,
        combining modality signals into unified sentiment scores.
        
        Validates:
            - Req 1.5: Fusion Engine generates scores at least once per second
            - Design: Time-windowed fusion on fixed intervals
        """
        logger.info("Starting fusion engine...")
        
        fusion_task = asyncio.create_task(
            self.fusion_engine.start(),
            name="fusion_engine"
        )
        self.tasks.append(fusion_task)
        
        logger.info("Fusion engine started successfully")
    
    async def start_stream_input(self, video_path: str):
        """Start stream input manager.
        
        Launches stream input manager that ingests video file and distributes
        frames to analysis modules via Redis Streams.
        
        Args:
            video_path: Path to video file to process
            
        Validates:
            - Req 1.1: System begins processing within 2 seconds
            - Req 8.1: System supports common streaming protocols
        """
        logger.info(f"Starting stream input for: {video_path}")
        
        # Connect to video file
        from src.models.enums import StreamProtocol
        connection = self.stream_manager.connect(video_path, StreamProtocol.FILE)
        
        if not connection.is_active:
            logger.error(f"Failed to connect to video file: {video_path}")
            raise RuntimeError(f"Failed to connect to video file: {video_path}")
        
        logger.info(f"Connected to video file: {video_path}")
        
        # Start streaming
        stream_task = asyncio.create_task(
            self.stream_manager.start_streaming(),
            name="stream_manager"
        )
        self.tasks.append(stream_task)
        
        logger.info("Stream input started successfully")
    
    async def monitor_tasks(self):
        """Monitor running tasks and handle failures.
        
        Monitors all asyncio tasks and logs errors if any task fails.
        Implements graceful degradation by allowing system to continue
        even if individual modules fail.
        
        Validates:
            - Req 8.4: System handles module failures gracefully
            - Req 9.3: System logs errors comprehensively
        """
        while not self.shutdown_event.is_set():
            # Check for completed tasks
            for task in self.tasks:
                if task.done():
                    task_name = task.get_name()
                    try:
                        # Check if task raised an exception
                        exception = task.exception()
                        if exception:
                            logger.error(f"Task {task_name} failed with exception: {exception}",
                                       exc_info=exception)
                        else:
                            logger.info(f"Task {task_name} completed successfully")
                    except asyncio.CancelledError:
                        logger.info(f"Task {task_name} was cancelled")
            
            # Wait before next check
            await asyncio.sleep(1.0)
    
    async def shutdown(self):
        """Gracefully shutdown all components.
        
        Cancels all running tasks and cleans up resources.
        
        Validates:
            - Design: Graceful shutdown of analysis pipeline
        """
        logger.info("Shutting down SentimentEngine...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Disconnect stream
        if self.stream_manager.is_active():
            self.stream_manager.disconnect()
        
        logger.info("SentimentEngine shutdown complete")
    
    async def run(self, video_path: str):
        """Run the complete sentiment analysis pipeline.
        
        This is the main execution method that:
        1. Starts stream input manager
        2. Starts all analysis modules
        3. Starts fusion engine
        4. Monitors tasks for errors
        5. Handles graceful shutdown
        
        Args:
            video_path: Path to video file to process
            
        Validates:
            - Req 1.1: System begins processing within 2 seconds
            - Req 1.5: Fusion Engine generates scores at least once per second
            - Req 9.1: End-to-end latency not exceeding 3 seconds
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting Real-Time Sentiment Analysis Engine")
            logger.info("=" * 60)
            
            # Start stream input
            await self.start_stream_input(video_path)
            
            # Small delay to let stream initialize
            await asyncio.sleep(0.5)
            
            # Start analysis modules
            await self.start_analysis_modules()
            
            # Small delay to let analyzers initialize
            await asyncio.sleep(0.5)
            
            # Start fusion engine
            await self.start_fusion_engine()
            
            logger.info("=" * 60)
            logger.info("All components started successfully")
            logger.info("Processing video stream...")
            logger.info("=" * 60)
            
            # Monitor tasks
            monitor_task = asyncio.create_task(self.monitor_tasks())
            
            # Wait for shutdown signal or stream completion
            await self.shutdown_event.wait()
            
            # Cancel monitor task
            monitor_task.cancel()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            await self.shutdown()


def setup_signal_handlers(engine: SentimentEngine):
    """Setup signal handlers for graceful shutdown.
    
    Args:
        engine: SentimentEngine instance to shutdown
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(engine.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main_async():
    """Async main entry point."""
    # Get video path from config or command line
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default test video (user should provide)
        video_path = "test_video.mp4"
        logger.warning(f"No video path provided, using default: {video_path}")
    
    # Check if video exists
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        logger.info("Usage: python -m src.main <video_path>")
        return
    
    # Create engine
    engine = SentimentEngine()
    
    # Setup signal handlers
    setup_signal_handlers(engine)
    
    # Run engine
    await engine.run(video_path)


def main():
    """Main entry point."""
    try:
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Run async main
        asyncio.run(main_async())
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
