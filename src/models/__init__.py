"""Neural network models for speech segmentation."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        """Initialize TCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation rate
            dropout: Dropout rate
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        out = out + residual
        
        return out


class TCNBasedSegmentation(nn.Module):
    """TCN-based speech segmentation model."""
    
    def __init__(self, config: DictConfig):
        """Initialize TCN segmentation model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        input_dim = config.data.n_mels
        hidden_dim = config.model.hidden_dim
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn_blocks.append(
                TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation, dropout=dropout)
            )
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram (batch_size, n_mels, time)
            
        Returns:
            torch.Tensor: Segmentation probabilities (batch_size, 1, time)
        """
        # Input projection
        x = self.input_proj(x)
        
        # TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        
        # Output projection
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        return x


class CNNBasedSegmentation(nn.Module):
    """CNN-based speech segmentation model."""
    
    def __init__(self, config: DictConfig):
        """Initialize CNN segmentation model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        input_dim = config.data.n_mels
        hidden_dim = config.model.hidden_dim
        dropout = config.model.dropout
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ])
        
        # Calculate output size after convolutions
        # Assuming input is (batch_size, 1, n_mels, time)
        # After 3 maxpool operations with stride 2, dimensions are reduced by 8
        self.fc_input_size = 128 * (input_dim // 8) * 1  # Adjust based on actual output size
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram (batch_size, n_mels, time)
            
        Returns:
            torch.Tensor: Segmentation probabilities (batch_size, 1, time)
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, n_mels, time)
        
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        # Reshape to match input time dimension
        batch_size = x.size(0)
        time_dim = self.config.data.n_mels  # This needs to be adjusted based on actual input
        x = x.unsqueeze(-1).expand(batch_size, 1, time_dim)
        
        return x


class TransformerBasedSegmentation(nn.Module):
    """Transformer-based speech segmentation model."""
    
    def __init__(self, config: DictConfig):
        """Initialize Transformer segmentation model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        input_dim = config.data.n_mels
        hidden_dim = config.model.hidden_dim
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim))  # Max sequence length
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram (batch_size, n_mels, time)
            
        Returns:
            torch.Tensor: Segmentation probabilities (batch_size, 1, time)
        """
        batch_size, n_mels, seq_len = x.shape
        
        # Transpose to (batch_size, time, n_mels)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(0):
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Output projection
        x = self.output_proj(x)
        x = self.sigmoid(x)
        
        # Transpose back to (batch_size, 1, time)
        x = x.transpose(1, 2)
        
        return x


class SegmentationModelFactory:
    """Factory class for creating segmentation models."""
    
    @staticmethod
    def create_model(model_type: str, config: DictConfig) -> nn.Module:
        """Create segmentation model based on type.
        
        Args:
            model_type: Type of model ("tcn", "cnn", "transformer")
            config: Configuration object
            
        Returns:
            Segmentation model
        """
        if model_type == "tcn":
            return TCNBasedSegmentation(config)
        elif model_type == "cnn":
            return CNNBasedSegmentation(config)
        elif model_type == "transformer":
            return TransformerBasedSegmentation(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class BoundaryDetector:
    """Boundary detection using segmentation model predictions."""
    
    def __init__(self, config: DictConfig):
        """Initialize boundary detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.boundary_threshold = config.segmentation.boundary_threshold
        self.min_segment_duration = config.segmentation.min_segment_duration
        self.max_segment_duration = config.segmentation.max_segment_duration
        
    def detect_boundaries(self, predictions: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Detect speech segment boundaries from model predictions.
        
        Args:
            predictions: Model predictions (probabilities)
            sample_rate: Sample rate of audio
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        # Convert probabilities to binary decisions
        boundaries = (predictions > self.boundary_threshold).astype(int)
        
        # Find segment starts and ends
        segments = self._find_segments(boundaries)
        
        # Convert frame indices to time
        hop_length = self.config.data.hop_length
        frame_duration = hop_length / sample_rate
        
        time_segments = []
        for start_frame, end_frame in segments:
            start_time = start_frame * frame_duration
            end_time = (end_frame + 1) * frame_duration
            
            # Apply duration constraints
            duration = end_time - start_time
            if duration >= self.min_segment_duration and duration <= self.max_segment_duration:
                time_segments.append((start_time, end_time))
        
        return time_segments
    
    def _find_segments(self, boundaries: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments of speech.
        
        Args:
            boundaries: Binary boundary array
            
        Returns:
            List of (start, end) frame tuples
        """
        segments = []
        start = None
        
        for i, boundary in enumerate(boundaries):
            if boundary == 1 and start is None:
                start = i
            elif boundary == 0 and start is not None:
                segments.append((start, i - 1))
                start = None
        
        if start is not None:
            segments.append((start, len(boundaries) - 1))
        
        return segments
