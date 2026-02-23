"""Utility functions for automatic speech segmentation."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig


def setup_logging(level: str = "INFO", log_dir: Optional[str] = None) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files (optional)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "segmentation.log") if log_dir else logging.NullHandler(),
        ],
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps")
        
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Loaded configuration
    """
    from omegaconf import OmegaConf
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        output_path: Output file path
    """
    from omegaconf import OmegaConf
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


def create_directories(paths: List[Union[str, Path]]) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def anonymize_filename(filename: str) -> str:
    """Anonymize filename by removing potentially identifying information.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Anonymized filename
    """
    # Remove common identifying patterns
    import re
    
    # Remove timestamps, dates, and common ID patterns
    filename = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', filename)
    filename = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIME', filename)
    filename = re.sub(r'[A-Z]{2,}\d{3,}', 'ID', filename)
    
    return filename


def remove_pii_from_text(text: str) -> str:
    """Remove potentially identifying information from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with PII removed
    """
    import re
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    
    # Remove SSN patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    return text


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def compute_rtf(processing_time: float, audio_duration: float) -> float:
    """Compute Real-Time Factor (RTF).
    
    Args:
        processing_time: Time taken to process audio (seconds)
        audio_duration: Duration of audio (seconds)
        
    Returns:
        float: RTF value
    """
    if audio_duration == 0:
        return float('inf')
    return processing_time / audio_duration


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.start_time, 'record'):
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.end_time.record()
            torch.cuda.synchronize()
            elapsed = self.start_time.elapsed_time(self.end_time) / 1000.0  # Convert to seconds
        else:
            import time
            elapsed = time.time() - self.start_time
        
        print(f"{self.name} took {elapsed:.4f} seconds")
        return False
