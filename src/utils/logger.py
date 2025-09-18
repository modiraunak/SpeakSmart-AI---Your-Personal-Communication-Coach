"""
Logging System for SpeakSmart AI
Comprehensive logging, monitoring, and debugging utilities

Authors: Team SpeakSmart (Group 256)
- Vedant Singh (Backend & Data Management)
- Raunak Kumar Modi (Team Lead & Integration)
"""

import logging
import logging.handlers
import sys
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import threading
from collections import deque, defaultdict
import time

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors"""
        
        # Add color to levelname
        if record.levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


class PerformanceLogger:
    """Performance monitoring and timing utilities"""
    
    def __init__(self):
        self.timers = {}
        self.performance_data = defaultdict(list)
        self.lock = threading.Lock()
    
    def start_timer(self, name: str):
        """Start a performance timer"""
        with self.lock:
            self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a performance timer and return elapsed time"""
        with self.lock:
            if name in self.timers:
                elapsed = time.time() - self.timers[name]
                self.performance_data[name].append({
                    'elapsed_time': elapsed,
                    'timestamp': datetime.now().isoformat()
                })
                del self.timers[name]
                return elapsed
            return 0.0
    
    def get_performance_stats(self, name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific timer"""
        with self.lock:
            if name not in self.performance_data:
                return {}
            
            times = [entry['elapsed_time'] for entry in self.performance_data[name]]
            
            if not times:
                return {}
            
            return {
                'count': len(times),
                'total_time': sum(times),
                'average_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'last_time': times[-1] if times else 0
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all timers"""
        stats = {}
        for name in self.performance_data.keys():
            stats[name] = self.get_performance_stats(name)
        return stats


class SessionLogger:
    """Session-specific logging for user interactions"""
    
    def __init__(self, session_id: str, max_entries: int = 1000):
        self.session_id = session_id
        self.max_entries = max_entries
        self.session_logs = deque(maxlen=max_entries)
        self.session_start = datetime.now()
        self.lock = threading.Lock()
    
    def log_event(self, event_type: str, event_data: Dict[str, Any], 
                  level: str = 'INFO'):
        """Log a session event"""
        
        with self.lock:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'event_type': event_type,
                'level': level,
                'data': event_data,
                'session_duration': (datetime.now() - self.session_start).total_seconds()
            }
            
            self.session_logs.append(entry)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session activity"""
        
        with self.lock:
            if not self.session_logs:
                return {'session_id': self.session_id, 'events': 0}
            
            event_counts = defaultdict(int)
            level_counts = defaultdict(int)
            
            for entry in self.session_logs:
                event_counts[entry['event_type']] += 1
                level_counts[entry['level']] += 1
            
            return {
                'session_id': self.session_id,
                'session_start': self.session_start.isoformat(),
                'session_duration': (datetime.now() - self.session_start).total_seconds(),
                'total_events': len(self.session_logs),
                'event_types': dict(event_counts),
                'log_levels': dict(level_counts),
                'last_event': self.session_logs[-1]['timestamp'] if self.session_logs else None
            }
    
    def export_session_logs(self) -> List[Dict[str, Any]]:
        """Export all session logs"""
        with self.lock:
            return list(self.session_logs)


class ApplicationLogger:
    """Main application logger with comprehensive features"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_performance: bool = True):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_performance = enable_performance
        
        # Initialize components
        self.loggers = {}
        self.performance_logger = PerformanceLogger() if enable_performance else None
        self.session_loggers = {}
        
        # Setup root logger
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup the root logger configuration"""
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file:
            log_file = self.log_dir / "speaksmart.log"
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            
            file_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Error file handler (separate file for errors)
        if self.enable_file:
            error_file = self.log_dir / "speaksmart_errors.log"
            
            error_handler = logging.handlers.RotatingFileHandler(
                filename=error_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            
            error_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s\n%(exc_info)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            error_handler.setFormatter(error_formatter)
            root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name"""
        
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def create_session_logger(self, session_id: str) -> SessionLogger:
        """Create a session-specific logger"""
        
        if session_id not in self.session_loggers:
            self.session_loggers[session_id] = SessionLogger(session_id)
        
        return self.session_loggers[session_id]
    
    def get_session_logger(self, session_id: str) -> Optional[SessionLogger]:
        """Get existing session logger"""
        return self.session_loggers.get(session_id)
    
    def log_exception(self, logger_name: str, exception: Exception, 
                     context: Optional[Dict[str, Any]] = None):
        """Log an exception with full traceback and context"""
        
        logger = self.get_logger(logger_name)
        
        # Prepare exception info
        exc_info = {
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        # Log the exception
        logger.error(
            f"Exception occurred: {type(exception).__name__}: {str(exception)}",
            extra={'exc_info': exc_info}
        )
        
        # Also log to error file with full context
        error_logger = self.get_logger('error_handler')
        error_logger.error(
            f"EXCEPTION DETAILS:\n"
            f"Type: {exc_info['exception_type']}\n"
            f"Message: {exc_info['exception_message']}\n"
            f"Context: {json.dumps(exc_info['context'], indent=2, default=str)}\n"
            f"Traceback:\n{exc_info['traceback']}"
        )
    
    def log_performance(self, operation: str, elapsed_time: float, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Log performance data"""
        
        if not self.performance_logger:
            return
        
        perf_logger = self.get_logger('performance')
        
        log_data = {
            'operation': operation,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            log_data.update(metadata)
        
        perf_logger.info(f"Performance: {operation} took {elapsed_time:.3f}s", extra=log_data)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        if not self.performance_logger:
            return {}
        
        return self.performance_logger.get_all_stats()
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleaned_files = []
        
        try:
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    cleaned_files.append(str(log_file))
            
            if cleaned_files:
                logger = self.get_logger('system')
                logger.info(f"Cleaned up {len(cleaned_files)} old log files")
                
        except Exception as e:
            logger = self.get_logger('system')
            logger.error(f"Error cleaning up logs: {str(e)}")
        
        return cleaned_files
    
    def export_logs(self, start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None,
                   log_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export logs within specified parameters"""
        
        # This is a simplified version - in practice you'd parse log files
        exported_logs = []
        
        try:
            log_files = list(self.log_dir.glob("speaksmart.log*"))
            
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            # Parse log line (simplified)
                            if ' | ' in line:
                                parts = line.strip().split(' | ')
                                if len(parts) >= 4:
                                    timestamp_str = parts[0]
                                    level = parts[1]
                                    logger_name = parts[2]
                                    message = ' | '.join(parts[3:])
                                    
                                    # Filter by level if specified
                                    if log_level and level != log_level:
                                        continue
                                    
                                    log_entry = {
                                        'timestamp': timestamp_str,
                                        'level': level,
                                        'logger': logger_name,
                                        'message': message
                                    }
                                    
                                    exported_logs.append(log_entry)
                        
                        except Exception:
                            continue  # Skip malformed lines
            
        except Exception as e:
            logger = self.get_logger('system')
            logger.error(f"Error exporting logs: {str(e)}")
        
        return exported_logs
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system and logging statistics"""
        
        stats = {
            'log_directory': str(self.log_dir),
            'log_level': logging.getLevelName(self.log_level),
            'active_loggers': len(self.loggers),
            'active_sessions': len(self.session_loggers),
            'performance_enabled': self.enable_performance,
            'uptime': datetime.now().isoformat()
        }
        
        # Add file system stats
        try:
            log_files = list(self.log_dir.glob("*.log*"))
            total_size = sum(f.stat().st_size for f in log_files)
            
            stats.update({
                'log_files_count': len(log_files),
                'total_log_size_mb': total_size / (1024 * 1024),
                'log_files': [f.name for f in log_files]
            })
        
        except Exception:
            pass
        
        # Add performance stats if available
        if self.performance_logger:
            stats['performance_stats'] = self.get_performance_stats()
        
        return stats


# Performance timer context manager
class PerformanceTimer:
    """Context manager for performance timing"""
    
    def __init__(self, app_logger: ApplicationLogger, operation_name: str, 
                 logger_name: str = 'performance', metadata: Optional[Dict[str, Any]] = None):
        self.app_logger = app_logger
        self.operation_name = operation_name
        self.logger_name = logger_name
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.app_logger.performance_logger:
            self.app_logger.performance_logger.start_timer(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            
            if self.app_logger.performance_logger:
                self.app_logger.performance_logger.end_timer(self.operation_name)
            
            self.app_logger.log_performance(self.operation_name, elapsed, self.metadata)
            
            # Log any exceptions that occurred
            if exc_type:
                self.app_logger.log_exception(
                    self.logger_name, 
                    exc_val, 
                    {'operation': self.operation_name, 'elapsed_time': elapsed}
                )


# Global logger instance
_app_logger_instance = None

def setup_logger(name: str = "speaksmart", 
                log_dir: str = "logs",
                log_level: str = "INFO") -> logging.Logger:
    """Setup and return application logger"""
    
    global _app_logger_instance
    
    if _app_logger_instance is None:
        _app_logger_instance = ApplicationLogger(
            log_dir=log_dir,
            log_level=log_level
        )
    
    return _app_logger_instance.get_logger(name)


def get_app_logger() -> Optional[ApplicationLogger]:
    """Get the global application logger instance"""
    return _app_logger_instance


def log_function_call(logger_name: str = 'function_calls'):
    """Decorator to automatically log function calls"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            app_logger = get_app_logger()
            if not app_logger:
                return func(*args, **kwargs)
            
            logger = app_logger.get_logger(logger_name)
            
            # Log function entry
            logger.debug(f"Entering {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
            
            try:
                with PerformanceTimer(app_logger, f"{func.__module__}.{func.__name__}"):
                    result = func(*args, **kwargs)
                
                logger.debug(f"Exiting {func.__name__} successfully")
                return result
            
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {str(e)}")
                app_logger.log_exception(logger_name, e, {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })
                raise
        
        return wrapper
    return decorator


# Utility functions
def log_system_info():
    """Log system information for debugging"""
    
    logger = setup_logger('system')
    
    try:
        import platform
        import psutil
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
        }
        
        logger.info(f"System Info: {json.dumps(system_info, indent=2)}")
        
    except ImportError:
        logger.warning("psutil not available - limited system info logging")
    except Exception as e:
        logger.error(f"Error logging system info: {str(e)}")


if __name__ == "__main__":
    # Test the logging system
    logger = setup_logger("test")
    
    logger.info("Testing SpeakSmart AI logging system")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance logging
    app_logger = get_app_logger()
    if app_logger:
        with PerformanceTimer(app_logger, "test_operation"):
            time.sleep(0.1)  # Simulate work
        
        print("Performance stats:", app_logger.get_performance_stats())
        print("System stats:", app_logger.get_system_stats())