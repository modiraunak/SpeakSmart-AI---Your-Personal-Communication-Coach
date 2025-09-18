"""
File Handler Module for SpeakSmart AI
Handles audio/video file processing, validation, and management

Authors: Team SpeakSmart (Group 256)
- Vedant Singh (Backend & Data Management)
- Raunak Kumar Modi (Team Lead & Integration)
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import mimetypes
from datetime import datetime
import hashlib

# Import numpy with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Audio/Video processing imports
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

try:
    import cv2
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class FileValidator:
    """File validation and security checks"""
    
    # Supported file formats
    SUPPORTED_AUDIO_FORMATS = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac',
        '.aac': 'audio/aac',
        '.ogg': 'audio/ogg',
        '.wma': 'audio/x-ms-wma'
    }
    
    SUPPORTED_VIDEO_FORMATS = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
        '.wmv': 'video/x-ms-wmv',
        '.flv': 'video/x-flv'
    }
    
    # File size limits (in bytes)
    MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
    
    # Duration limits (in seconds)
    MAX_AUDIO_DURATION = 600  # 10 minutes
    MAX_VIDEO_DURATION = 900  # 15 minutes
    
    @staticmethod
    def validate_file_upload(file_path: str, file_type: str) -> Dict[str, Any]:
        """Validate uploaded file for security and format compliance"""
        
        validation_result = {
            'valid': False,
            'file_type': None,
            'size': 0,
            'duration': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            if not os.path.exists(file_path):
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_extension = Path(file_path).suffix.lower()
            
            validation_result['size'] = file_size
            validation_result['file_type'] = file_type
            
            # Validate file extension and MIME type
            if file_type == 'audio':
                if file_extension not in FileValidator.SUPPORTED_AUDIO_FORMATS:
                    validation_result['errors'].append(f"Unsupported audio format: {file_extension}")
                    return validation_result
                
                if file_size > FileValidator.MAX_AUDIO_SIZE:
                    validation_result['errors'].append(f"Audio file too large: {file_size / (1024*1024):.1f}MB > {FileValidator.MAX_AUDIO_SIZE / (1024*1024):.0f}MB")
                    return validation_result
            
            elif file_type == 'video':
                if file_extension not in FileValidator.SUPPORTED_VIDEO_FORMATS:
                    validation_result['errors'].append(f"Unsupported video format: {file_extension}")
                    return validation_result
                
                if file_size > FileValidator.MAX_VIDEO_SIZE:
                    validation_result['errors'].append(f"Video file too large: {file_size / (1024*1024):.1f}MB > {FileValidator.MAX_VIDEO_SIZE / (1024*1024):.0f}MB")
                    return validation_result
            
            # Verify MIME type
            detected_mime, _ = mimetypes.guess_type(file_path)
            expected_formats = (FileValidator.SUPPORTED_AUDIO_FORMATS if file_type == 'audio' 
                              else FileValidator.SUPPORTED_VIDEO_FORMATS)
            
            if detected_mime and detected_mime not in expected_formats.values():
                validation_result['warnings'].append(f"MIME type mismatch: detected {detected_mime}")
            
            # Validate file content and duration
            duration = FileValidator._get_file_duration(file_path, file_type)
            validation_result['duration'] = duration
            
            if duration <= 0:
                validation_result['errors'].append("Cannot determine file duration or file is empty")
                return validation_result
            
            max_duration = (FileValidator.MAX_AUDIO_DURATION if file_type == 'audio' 
                          else FileValidator.MAX_VIDEO_DURATION)
            
            if duration > max_duration:
                validation_result['errors'].append(f"File too long: {duration:.1f}s > {max_duration}s")
                return validation_result
            
            # File is valid
            validation_result['valid'] = True
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    @staticmethod
    def _get_file_duration(file_path: str, file_type: str) -> float:
        """Get file duration in seconds"""
        
        try:
            if file_type == 'audio' and AUDIO_PROCESSING_AVAILABLE:
                # Use librosa for audio files
                duration = librosa.get_duration(filename=file_path)
                return duration
            
            elif file_type == 'video':
                if MOVIEPY_AVAILABLE:
                    # Use moviepy for video files
                    with VideoFileClip(file_path) as clip:
                        return clip.duration
                
                elif VIDEO_PROCESSING_AVAILABLE:
                    # Fallback to OpenCV
                    cap = cv2.VideoCapture(file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    
                    if fps > 0:
                        return frame_count / fps
            
            return 0
            
        except Exception as e:
            logger.warning(f"Could not determine duration for {file_path}: {str(e)}")
            return 0
    
    @staticmethod
    def scan_for_malware(file_path: str) -> bool:
        """Basic malware scanning (placeholder for actual implementation)"""
        
        try:
            # Basic file signature checks
            with open(file_path, 'rb') as f:
                header = f.read(512)
            
            # Check for suspicious patterns (this is a simplified example)
            suspicious_patterns = [
                b'\x4D\x5A',  # PE executable header
                b'\x7F\x45\x4C\x46',  # ELF executable header
            ]
            
            for pattern in suspicious_patterns:
                if pattern in header[:50]:  # Check only beginning of file
                    logger.warning(f"Suspicious file signature detected in {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Malware scan error: {str(e)}")
            return False


class AudioFileProcessor:
    """Audio file processing and conversion"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning("Audio processing libraries not available")
    
    def load_audio_file(self, file_path: str, target_sr: int = 44100) -> Tuple[Optional[Union['np.ndarray', list]], int]:
        """Load and preprocess audio file"""
        
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.error("Audio processing not available - install librosa and soundfile")
            return None, 0
        
        if not NUMPY_AVAILABLE:
            logger.error("Numpy not available - install numpy")
            return None, 0
        
        try:
            # Load audio with librosa
            audio_data, sample_rate = librosa.load(
                file_path, 
                sr=target_sr,
                mono=True,  # Convert to mono
                dtype=np.float32
            )
            
            logger.info(f"Loaded audio: {len(audio_data)/sample_rate:.2f}s at {sample_rate}Hz")
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            return None, 0
    
    def convert_audio_format(self, input_path: str, output_format: str = 'wav') -> Optional[str]:
        """Convert audio file to specified format"""
        
        if not AUDIO_PROCESSING_AVAILABLE:
            return None
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(input_path, sr=None)
            
            # Create output filename
            input_name = Path(input_path).stem
            output_path = os.path.join(self.temp_dir, f"{input_name}.{output_format}")
            
            # Save in new format
            sf.write(output_path, audio_data, sample_rate)
            
            logger.info(f"Converted {input_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio conversion error: {str(e)}")
            return None
    
    def extract_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract audio file metadata"""
        
        metadata = {
            'filename': Path(file_path).name,
            'size_bytes': os.path.getsize(file_path),
            'format': Path(file_path).suffix.lower(),
            'duration': 0,
            'sample_rate': 0,
            'channels': 0,
            'bitrate': None
        }
        
        if not AUDIO_PROCESSING_AVAILABLE:
            return metadata
        
        try:
            # Get basic info with librosa
            info = sf.info(file_path)
            
            metadata.update({
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'format_info': str(info.format_info)
            })
            
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {str(e)}")
        
        return metadata
    
    def preprocess_audio(self, audio_data, sample_rate: int):
        """Apply preprocessing to audio data"""
        
        if not NUMPY_AVAILABLE:
            logger.error("Numpy not available for audio preprocessing")
            return audio_data
        
        try:
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Remove DC component
            audio_data = audio_data - np.mean(audio_data)
            
            # Apply gentle high-pass filter to remove low-frequency noise
            try:
                from scipy.signal import butter, filtfilt
                
                nyquist = sample_rate / 2
                low_cutoff = 80 / nyquist  # 80 Hz high-pass
                b, a = butter(2, low_cutoff, btype='high')
                
                audio_data = filtfilt(b, a, audio_data)
            except ImportError:
                logger.warning("Scipy not available - skipping audio filtering")
            
            return audio_data.astype(np.float32) if hasattr(audio_data, 'astype') else audio_data
            
        except Exception as e:
            logger.warning(f"Audio preprocessing error: {str(e)}")
            return audio_data


class VideoFileProcessor:
    """Video file processing and frame extraction"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.warning("Video processing libraries not available")
    
    def load_video_file(self, file_path: str) -> Optional[cv2.VideoCapture]:
        """Load video file for processing"""
        
        if not VIDEO_PROCESSING_AVAILABLE:
            logger.error("Video processing not available - install opencv-python")
            return None
        
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video file: {file_path}")
                return None
            
            # Verify video has frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                logger.error(f"Video file has no frames: {file_path}")
                cap.release()
                return None
            
            logger.info(f"Loaded video: {frame_count} frames")
            return cap
            
        except Exception as e:
            logger.error(f"Error loading video file {file_path}: {str(e)}")
            return None
    
    def extract_frames(self, video_path: str, max_frames: int = 1000, 
                      frame_skip: int = 1) -> List:
        """Extract frames from video file"""
        
        frames = []
        
        cap = self.load_video_file(video_path)
        if cap is None:
            return frames
        
        try:
            frame_count = 0
            extracted_count = 0
            
            while extracted_count < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_count % frame_skip == 0:
                    frames.append(frame)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            
        except Exception as e:
            logger.error(f"Frame extraction error: {str(e)}")
            cap.release()
        
        return frames
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio track from video file"""
        
        if not MOVIEPY_AVAILABLE:
            logger.warning("MoviePy not available - cannot extract audio from video")
            return None
        
        try:
            # Generate output path
            video_name = Path(video_path).stem
            audio_path = os.path.join(self.temp_dir, f"{video_name}_extracted.wav")
            
            # Extract audio
            with VideoFileClip(video_path) as video:
                if video.audio is not None:
                    video.audio.write_audiofile(
                        audio_path,
                        verbose=False,
                        logger=None  # Suppress moviepy logs
                    )
                    
                    logger.info(f"Extracted audio to {audio_path}")
                    return audio_path
                else:
                    logger.warning("Video file has no audio track")
                    return None
            
        except Exception as e:
            logger.error(f"Audio extraction error: {str(e)}")
            return None
    
    def get_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract video file metadata"""
        
        metadata = {
            'filename': Path(file_path).name,
            'size_bytes': os.path.getsize(file_path),
            'format': Path(file_path).suffix.lower(),
            'duration': 0,
            'fps': 0,
            'width': 0,
            'height': 0,
            'frame_count': 0,
            'has_audio': False
        }
        
        if not VIDEO_PROCESSING_AVAILABLE:
            return metadata
        
        try:
            cap = cv2.VideoCapture(file_path)
            
            if cap.isOpened():
                metadata.update({
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                })
                
                if metadata['fps'] > 0:
                    metadata['duration'] = metadata['frame_count'] / metadata['fps']
                
                cap.release()
            
            # Check for audio track
            if MOVIEPY_AVAILABLE:
                try:
                    with VideoFileClip(file_path) as clip:
                        metadata['has_audio'] = clip.audio is not None
                        if metadata['duration'] == 0:
                            metadata['duration'] = clip.duration
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Could not extract video metadata from {file_path}: {str(e)}")
        
        return metadata


class FileManager:
    """Central file management system"""
    
    def __init__(self, base_temp_dir: Optional[str] = None):
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self.session_temp_dir = None
        self.processed_files = {}
        
        # Initialize processors
        self.validator = FileValidator()
        self.audio_processor = AudioFileProcessor(self.base_temp_dir)
        self.video_processor = VideoFileProcessor(self.base_temp_dir)
    
    def create_session_workspace(self) -> str:
        """Create temporary workspace for current session"""
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_temp_dir = os.path.join(self.base_temp_dir, f"speaksmart_{session_id}")
        
        os.makedirs(self.session_temp_dir, exist_ok=True)
        logger.info(f"Created session workspace: {self.session_temp_dir}")
        
        return self.session_temp_dir
    
    def process_uploaded_file(self, uploaded_file, file_type: str) -> Dict[str, Any]:
        """Process uploaded file from Streamlit"""
        
        result = {
            'success': False,
            'file_path': None,
            'metadata': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Create session workspace if needed
            if not self.session_temp_dir:
                self.create_session_workspace()
            
            # Save uploaded file to temporary location
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
            temp_filename = f"{file_hash}_{uploaded_file.name}"
            temp_path = os.path.join(self.session_temp_dir, temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Validate file
            validation = self.validator.validate_file_upload(temp_path, file_type)
            
            if not validation['valid']:
                result['errors'] = validation['errors']
                result['warnings'] = validation['warnings']
                # Clean up invalid file
                os.remove(temp_path)
                return result
            
            # Scan for malware
            if not self.validator.scan_for_malware(temp_path):
                result['errors'].append("File failed security scan")
                os.remove(temp_path)
                return result
            
            # Extract metadata
            if file_type == 'audio':
                metadata = self.audio_processor.extract_audio_metadata(temp_path)
            elif file_type == 'video':
                metadata = self.video_processor.get_video_metadata(temp_path)
            else:
                metadata = {}
            
            # Store file information
            file_info = {
                'original_name': uploaded_file.name,
                'temp_path': temp_path,
                'file_type': file_type,
                'metadata': metadata,
                'processed_at': datetime.now().isoformat()
            }
            
            self.processed_files[file_hash] = file_info
            
            result.update({
                'success': True,
                'file_path': temp_path,
                'metadata': metadata,
                'file_id': file_hash,
                'warnings': validation['warnings']
            })
            
            logger.info(f"Successfully processed {file_type} file: {uploaded_file.name}")
            
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            result['errors'].append(f"Processing error: {str(e)}")
        
        return result
    
    def get_processed_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed file"""
        return self.processed_files.get(file_id)
    
    def cleanup_session_files(self):
        """Clean up temporary files from current session"""
        
        if self.session_temp_dir and os.path.exists(self.session_temp_dir):
            try:
                shutil.rmtree(self.session_temp_dir)
                logger.info(f"Cleaned up session workspace: {self.session_temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up session files: {str(e)}")
        
        self.processed_files.clear()
        self.session_temp_dir = None
    
    def export_session_data(self, output_path: str) -> bool:
        """Export processed file information"""
        
        try:
            export_data = {
                'session_timestamp': datetime.now().isoformat(),
                'processed_files': self.processed_files,
                'session_workspace': self.session_temp_dir
            }
            
            with open(output_path, 'w') as f:
                import json
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported session data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            return False


# Utility functions
def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    return os.path.getsize(file_path) / (1024 * 1024)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def is_supported_format(filename: str, file_type: str) -> bool:
    """Check if file format is supported"""
    extension = Path(filename).suffix.lower()
    
    if file_type == 'audio':
        return extension in FileValidator.SUPPORTED_AUDIO_FORMATS
    elif file_type == 'video':
        return extension in FileValidator.SUPPORTED_VIDEO_FORMATS
    
    return False


# Context manager for file operations
class TemporaryFileManager:
    """Context manager for temporary file operations"""
    
    def __init__(self, cleanup_on_exit: bool = True):
        self.cleanup_on_exit = cleanup_on_exit
        self.temp_files = []
        self.temp_dirs = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            self.cleanup()
    
    def create_temp_file(self, suffix: str = '', prefix: str = 'speaksmart_') -> str:
        """Create a temporary file"""
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)  # Close the file descriptor
        
        self.temp_files.append(path)
        return path
    
    def create_temp_dir(self, prefix: str = 'speaksmart_') -> str:
        """Create a temporary directory"""
        path = tempfile.mkdtemp(prefix=prefix)
        self.temp_dirs.append(path)
        return path
    
    def cleanup(self):
        """Clean up all temporary files and directories"""
        
        # Remove temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")
        
        # Remove temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Could not remove temp dir {temp_dir}: {str(e)}")
        
        self.temp_files.clear()
        self.temp_dirs.clear()


if __name__ == "__main__":
    # Test file handler functionality
    file_manager = FileManager()
    print("FileManager initialized successfully")
    
    # Test validation
    validator = FileValidator()
    print("Available audio formats:", list(validator.SUPPORTED_AUDIO_FORMATS.keys()))
    print("Available video formats:", list(validator.SUPPORTED_VIDEO_FORMATS.keys()))