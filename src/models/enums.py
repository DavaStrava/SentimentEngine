"""Enumerations for stream protocols and connection types"""

from enum import Enum


class StreamProtocol(Enum):
    """Supported streaming protocols"""
    RTMP = "rtmp"
    HLS = "hls"
    WEBRTC = "webrtc"
    FILE = "file"  # Local file input


class StreamConnection:
    """Stream connection information
    
    Attributes:
        url: Stream URL or file path
        protocol: Streaming protocol
        is_active: Whether connection is currently active
        audio_codec: Audio codec name (e.g., "aac", "mp3")
        video_codec: Video codec name (e.g., "h264", "vp9")
    """
    
    def __init__(
        self,
        url: str,
        protocol: StreamProtocol,
        is_active: bool = False,
        audio_codec: str = "",
        video_codec: str = ""
    ):
        self.url = url
        self.protocol = protocol
        self.is_active = is_active
        self.audio_codec = audio_codec
        self.video_codec = video_codec
    
    def __repr__(self) -> str:
        return (
            f"StreamConnection(url='{self.url}', protocol={self.protocol.value}, "
            f"is_active={self.is_active}, audio_codec='{self.audio_codec}', "
            f"video_codec='{self.video_codec}')"
        )
