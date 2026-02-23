"""Voice Activity Detection (VAD) implementations for speech segmentation."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EnergyBasedVAD:
    """Energy-based Voice Activity Detection."""
    
    def __init__(self, config: DictConfig):
        """Initialize energy-based VAD.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.energy_threshold = config.vad.energy_threshold
        self.min_speech_duration = config.vad.min_speech_duration
        self.min_silence_duration = config.vad.min_silence_duration
        self.smoothing_window = config.vad.smoothing_window
        
    def detect(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
        """Detect voice activity using energy-based method.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (vad_array, metadata)
        """
        # Compute energy in frames
        frame_size = int(self.config.data.frame_size)
        hop_length = int(self.config.data.hop_length)
        
        energy = []
        for i in range(0, len(audio), hop_length):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Normalize energy
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        # Apply threshold
        vad_raw = (energy > self.energy_threshold).astype(int)
        
        # Apply smoothing
        vad_smoothed = self._smooth_vad(vad_raw)
        
        # Apply duration constraints
        vad_final = self._apply_duration_constraints(
            vad_smoothed, sample_rate, hop_length
        )
        
        metadata = {
            'energy': energy,
            'vad_raw': vad_raw,
            'vad_smoothed': vad_smoothed,
            'threshold': self.energy_threshold
        }
        
        return vad_final, metadata
    
    def _smooth_vad(self, vad: np.ndarray) -> np.ndarray:
        """Apply smoothing to VAD decisions.
        
        Args:
            vad: Raw VAD decisions
            
        Returns:
            np.ndarray: Smoothed VAD decisions
        """
        if self.smoothing_window <= 1:
            return vad
        
        # Apply median filtering
        from scipy.ndimage import median_filter
        smoothed = median_filter(vad, size=self.smoothing_window)
        
        return smoothed
    
    def _apply_duration_constraints(self, vad: np.ndarray, sample_rate: int, hop_length: int) -> np.ndarray:
        """Apply minimum duration constraints to VAD decisions.
        
        Args:
            vad: VAD decisions
            sample_rate: Sample rate
            hop_length: Hop length
            
        Returns:
            np.ndarray: VAD decisions with duration constraints applied
        """
        frame_duration = hop_length / sample_rate
        min_speech_frames = int(self.min_speech_duration / frame_duration)
        min_silence_frames = int(self.min_silence_duration / frame_duration)
        
        vad_constrained = vad.copy()
        
        # Remove short speech segments
        speech_segments = self._find_segments(vad_constrained, 1)
        for start, end in speech_segments:
            if end - start + 1 < min_speech_frames:
                vad_constrained[start:end+1] = 0
        
        # Remove short silence segments
        silence_segments = self._find_segments(vad_constrained, 0)
        for start, end in silence_segments:
            if end - start + 1 < min_silence_frames:
                vad_constrained[start:end+1] = 1
        
        return vad_constrained
    
    def _find_segments(self, vad: np.ndarray, value: int) -> List[Tuple[int, int]]:
        """Find continuous segments of a given value.
        
        Args:
            vad: VAD array
            value: Value to find segments for
            
        Returns:
            List of (start, end) tuples
        """
        segments = []
        start = None
        
        for i, v in enumerate(vad):
            if v == value and start is None:
                start = i
            elif v != value and start is not None:
                segments.append((start, i - 1))
                start = None
        
        if start is not None:
            segments.append((start, len(vad) - 1))
        
        return segments


class SpectralBasedVAD:
    """Spectral-based Voice Activity Detection using multiple features."""
    
    def __init__(self, config: DictConfig):
        """Initialize spectral-based VAD.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.spectral_threshold = config.vad.spectral_threshold
        self.zcr_threshold = config.vad.zero_crossing_threshold
        self.min_speech_duration = config.vad.min_speech_duration
        self.min_silence_duration = config.vad.min_silence_duration
        
    def detect(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
        """Detect voice activity using spectral features.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (vad_array, metadata)
        """
        import librosa
        
        # Extract spectral features
        frame_size = int(self.config.data.frame_size)
        hop_length = int(self.config.data.hop_length)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sample_rate, hop_length=hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=frame_size, hop_length=hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sample_rate, hop_length=hop_length
        )[0]
        
        # Normalize features
        spectral_centroid = spectral_centroid / np.max(spectral_centroid)
        spectral_rolloff = spectral_rolloff / np.max(spectral_rolloff)
        
        # Combine features for VAD decision
        # Speech typically has higher spectral centroid and rolloff, moderate ZCR
        speech_score = (
            spectral_centroid * 0.4 +
            spectral_rolloff * 0.3 +
            (1 - np.clip(zcr, 0, 1)) * 0.3
        )
        
        # Apply thresholds
        vad_raw = (
            (speech_score > self.spectral_threshold) &
            (zcr < self.zcr_threshold)
        ).astype(int)
        
        # Apply duration constraints
        vad_final = self._apply_duration_constraints(
            vad_raw, sample_rate, hop_length
        )
        
        metadata = {
            'spectral_centroid': spectral_centroid,
            'zcr': zcr,
            'spectral_rolloff': spectral_rolloff,
            'speech_score': speech_score,
            'vad_raw': vad_raw
        }
        
        return vad_final, metadata
    
    def _apply_duration_constraints(self, vad: np.ndarray, sample_rate: int, hop_length: int) -> np.ndarray:
        """Apply minimum duration constraints to VAD decisions."""
        frame_duration = hop_length / sample_rate
        min_speech_frames = int(self.min_speech_duration / frame_duration)
        min_silence_frames = int(self.min_silence_duration / frame_duration)
        
        vad_constrained = vad.copy()
        
        # Remove short speech segments
        speech_segments = self._find_segments(vad_constrained, 1)
        for start, end in speech_segments:
            if end - start + 1 < min_speech_frames:
                vad_constrained[start:end+1] = 0
        
        # Remove short silence segments
        silence_segments = self._find_segments(vad_constrained, 0)
        for start, end in silence_segments:
            if end - start + 1 < min_silence_frames:
                vad_constrained[start:end+1] = 1
        
        return vad_constrained
    
    def _find_segments(self, vad: np.ndarray, value: int) -> List[Tuple[int, int]]:
        """Find continuous segments of a given value."""
        segments = []
        start = None
        
        for i, v in enumerate(vad):
            if v == value and start is None:
                start = i
            elif v != value and start is not None:
                segments.append((start, i - 1))
                start = None
        
        if start is not None:
            segments.append((start, len(vad) - 1))
        
        return segments


class MLBasedVAD:
    """Machine Learning-based Voice Activity Detection."""
    
    def __init__(self, config: DictConfig):
        """Initialize ML-based VAD.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the ML model.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Binary labels (0 for non-speech, 1 for speech)
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, labels)
        self.is_trained = True
        
        logger.info("ML-based VAD model trained successfully")
    
    def detect(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
        """Detect voice activity using ML model.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (vad_array, metadata)
        """
        if not self.is_trained:
            logger.warning("ML model not trained. Falling back to energy-based VAD.")
            energy_vad = EnergyBasedVAD(self.config)
            return energy_vad.detect(audio, sample_rate)
        
        # Extract features
        features = self._extract_features(audio, sample_rate)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        vad_probs = self.model.predict_proba(features_scaled)[:, 1]
        vad_raw = (vad_probs > 0.5).astype(int)
        
        # Apply duration constraints
        hop_length = int(self.config.data.hop_length)
        vad_final = self._apply_duration_constraints(
            vad_raw, sample_rate, hop_length
        )
        
        metadata = {
            'vad_probs': vad_probs,
            'vad_raw': vad_raw,
            'features': features
        }
        
        return vad_final, metadata
    
    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract features for ML model."""
        import librosa
        
        frame_size = int(self.config.data.frame_size)
        hop_length = int(self.config.data.hop_length)
        
        features = []
        
        for i in range(0, len(audio), hop_length):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            
            # Energy
            energy = np.sum(frame ** 2)
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(frame.reshape(1, -1)))
            
            # Spectral features
            stft = librosa.stft(frame, n_fft=512)
            magnitude = np.abs(stft)
            
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=magnitude))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=magnitude))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=magnitude))
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Combine features
            frame_features = np.concatenate([
                [energy, zcr, spectral_centroid, spectral_rolloff, spectral_bandwidth],
                mfcc_mean
            ])
            
            features.append(frame_features)
        
        return np.array(features)
    
    def _apply_duration_constraints(self, vad: np.ndarray, sample_rate: int, hop_length: int) -> np.ndarray:
        """Apply minimum duration constraints to VAD decisions."""
        frame_duration = hop_length / sample_rate
        min_speech_frames = int(self.config.vad.min_speech_duration / frame_duration)
        min_silence_frames = int(self.config.vad.min_silence_duration / frame_duration)
        
        vad_constrained = vad.copy()
        
        # Remove short speech segments
        speech_segments = self._find_segments(vad_constrained, 1)
        for start, end in speech_segments:
            if end - start + 1 < min_speech_frames:
                vad_constrained[start:end+1] = 0
        
        # Remove short silence segments
        silence_segments = self._find_segments(vad_constrained, 0)
        for start, end in silence_segments:
            if end - start + 1 < min_silence_frames:
                vad_constrained[start:end+1] = 1
        
        return vad_constrained
    
    def _find_segments(self, vad: np.ndarray, value: int) -> List[Tuple[int, int]]:
        """Find continuous segments of a given value."""
        segments = []
        start = None
        
        for i, v in enumerate(vad):
            if v == value and start is None:
                start = i
            elif v != value and start is not None:
                segments.append((start, i - 1))
                start = None
        
        if start is not None:
            segments.append((start, len(vad) - 1))
        
        return segments


class VADFactory:
    """Factory class for creating VAD instances."""
    
    @staticmethod
    def create_vad(vad_type: str, config: DictConfig):
        """Create VAD instance based on type.
        
        Args:
            vad_type: Type of VAD ("energy", "spectral", "ml")
            config: Configuration object
            
        Returns:
            VAD instance
        """
        if vad_type == "energy":
            return EnergyBasedVAD(config)
        elif vad_type == "spectral":
            return SpectralBasedVAD(config)
        elif vad_type == "ml":
            return MLBasedVAD(config)
        else:
            raise ValueError(f"Unknown VAD type: {vad_type}")
