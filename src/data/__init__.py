"""Data loading and preprocessing utilities for speech segmentation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig

from .utils import anonymize_filename, remove_pii_from_text

logger = logging.getLogger(__name__)


class AudioLoader:
    """Audio loading and preprocessing class."""
    
    def __init__(self, config: DictConfig):
        """Initialize audio loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.sample_rate = config.data.sample_rate
        self.preemphasis = config.data.preemphasis
        
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Load audio with librosa for better format support
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate)
            
            # Apply pre-emphasis if specified
            if self.preemphasis > 0:
                audio = librosa.effects.preemphasis(audio, coef=self.preemphasis)
            
            logger.debug(f"Loaded audio: {file_path.name}, duration: {len(audio)/sr:.2f}s")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def load_audio_torch(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Load audio file as PyTorch tensor.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            torch.Tensor: Audio tensor of shape (1, samples)
        """
        audio, _ = self.load_audio(file_path)
        return torch.from_numpy(audio).unsqueeze(0).float()


class FeatureExtractor:
    """Feature extraction for speech segmentation."""
    
    def __init__(self, config: DictConfig):
        """Initialize feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.frame_size = config.data.frame_size
        self.hop_length = config.data.hop_length
        self.n_fft = config.data.n_fft
        self.n_mels = config.data.n_mels
        self.sample_rate = config.data.sample_rate
        
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract energy features from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            np.ndarray: Energy values per frame
        """
        # Compute energy in short frames
        energy = []
        for i in range(0, len(audio), self.hop_length):
            frame = audio[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Normalize energy
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        return energy
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dict containing spectral features
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.frame_size, hop_length=self.hop_length
        )[0]
        
        # Mel-frequency cepstral coefficients
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13, 
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zcr': zcr,
            'mfcc': mfcc,
            'magnitude': magnitude
        }
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            np.ndarray: Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec


class Dataset:
    """Dataset class for speech segmentation."""
    
    def __init__(self, config: DictConfig, split: str = "train"):
        """Initialize dataset.
        
        Args:
            config: Configuration object
            split: Dataset split ("train", "val", "test")
        """
        self.config = config
        self.split = split
        self.audio_loader = AudioLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by split
        if 'split' in self.metadata.columns:
            self.metadata = self.metadata[self.metadata['split'] == split]
        
        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata.
        
        Returns:
            pd.DataFrame: Metadata dataframe
        """
        meta_file = Path(self.config.paths.meta_file)
        
        if meta_file.exists():
            metadata = pd.read_csv(meta_file)
        else:
            # Create synthetic metadata for demo
            metadata = self._create_synthetic_metadata()
        
        return metadata
    
    def _create_synthetic_metadata(self) -> pd.DataFrame:
        """Create synthetic metadata for demonstration.
        
        Returns:
            pd.DataFrame: Synthetic metadata
        """
        logger.warning("No metadata file found. Creating synthetic dataset for demonstration.")
        
        # Create synthetic audio files
        wav_dir = Path(self.config.paths.wav_dir)
        wav_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        for i in range(10):  # Create 10 synthetic samples
            # Generate synthetic speech-like audio
            duration = np.random.uniform(2.0, 8.0)
            sample_rate = self.config.data.sample_rate
            
            # Create speech-like signal with pauses
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Generate multiple speech segments with pauses
            audio = np.zeros_like(t)
            segment_starts = np.random.uniform(0, duration - 1, 3)
            
            for start in segment_starts:
                end = min(start + np.random.uniform(0.5, 2.0), duration)
                segment_t = t[(t >= start) & (t < end)]
                segment_audio = np.sin(2 * np.pi * 200 * segment_t) * np.exp(-segment_t)
                audio[(t >= start) & (t < end)] = segment_audio
            
            # Add noise
            noise = np.random.normal(0, 0.1, len(audio))
            audio = audio + noise
            
            # Save synthetic audio
            filename = f"synthetic_speech_{i:03d}.wav"
            filepath = wav_dir / filename
            sf.write(filepath, audio, sample_rate)
            
            # Create metadata entry
            metadata.append({
                'id': f"synth_{i:03d}",
                'filename': filename,
                'path': str(filepath),
                'duration': duration,
                'sample_rate': sample_rate,
                'split': np.random.choice(['train', 'val', 'test'], p=[0.7, 0.2, 0.1]),
                'speaker_id': f"speaker_{i % 3}",
                'language': 'en',
                'quality': 'synthetic'
            })
        
        return pd.DataFrame(metadata)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.Tensor, str]]:
        """Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dict containing audio data and metadata
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio, sr = self.audio_loader.load_audio(row['path'])
        
        # Extract features
        energy = self.feature_extractor.extract_energy(audio)
        spectral_features = self.feature_extractor.extract_spectral_features(audio)
        mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
        
        return {
            'audio': audio,
            'sample_rate': sr,
            'energy': energy,
            'spectral_features': spectral_features,
            'mel_spectrogram': mel_spec,
            'id': row['id'],
            'filename': row['filename'],
            'duration': row['duration'],
            'speaker_id': row.get('speaker_id', 'unknown'),
            'language': row.get('language', 'unknown')
        }
    
    def get_audio_path(self, idx: int) -> str:
        """Get audio file path for given index.
        
        Args:
            idx: Item index
            
        Returns:
            str: Audio file path
        """
        return self.metadata.iloc[idx]['path']
