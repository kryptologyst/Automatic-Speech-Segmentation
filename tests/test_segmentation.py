"""Unit tests for automatic speech segmentation."""

import numpy as np
import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.data import AudioLoader, FeatureExtractor, Dataset
from src.features import EnergyBasedVAD, SpectralBasedVAD, MLBasedVAD
from src.models import TCNBasedSegmentation, CNNBasedSegmentation, TransformerBasedSegmentation
from src.metrics import SegmentationMetrics, MetricsAggregator
from src.utils import get_device, set_seed, format_duration, compute_rtf


@pytest.fixture
def config():
    """Create a test configuration."""
    config_dict = {
        'data': {
            'sample_rate': 16000,
            'frame_size': 1024,
            'hop_length': 512,
            'n_fft': 2048,
            'n_mels': 80,
            'preemphasis': 0.97
        },
        'vad': {
            'energy_threshold': 0.02,
            'spectral_threshold': 0.3,
            'zero_crossing_threshold': 0.1,
            'min_speech_duration': 0.1,
            'min_silence_duration': 0.1,
            'smoothing_window': 3
        },
        'segmentation': {
            'boundary_threshold': 0.5,
            'min_segment_duration': 0.2,
            'max_segment_duration': 10.0,
            'overlap_threshold': 0.1
        },
        'model': {
            'name': 'tcn',
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.1
        },
        'evaluation': {
            'metrics': ['precision', 'recall', 'f1', 'der', 'jer'],
            'collar_tolerance': 0.25
        },
        'device': 'cpu'
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def sample_audio():
    """Create sample audio data for testing."""
    # Generate synthetic speech-like audio
    duration = 2.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create speech segments with pauses
    audio = np.zeros_like(t)
    
    # First speech segment
    audio[8000:12000] = np.sin(2 * np.pi * 200 * t[8000:12000]) * np.exp(-t[8000:12000])
    
    # Second speech segment
    audio[20000:28000] = np.sin(2 * np.pi * 300 * t[20000:28000]) * np.exp(-t[20000:28000])
    
    # Add noise
    audio += np.random.normal(0, 0.05, len(audio))
    
    return audio, sample_rate


class TestAudioLoader:
    """Test AudioLoader class."""
    
    def test_audio_loader_init(self, config):
        """Test AudioLoader initialization."""
        loader = AudioLoader(config)
        assert loader.sample_rate == config.data.sample_rate
        assert loader.preemphasis == config.data.preemphasis
    
    def test_load_audio_synthetic(self, config, sample_audio):
        """Test loading synthetic audio."""
        loader = AudioLoader(config)
        
        # Create temporary audio file
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio, sr = sample_audio
            sf.write(tmp_file.name, audio, sr)
            
            # Load audio
            loaded_audio, loaded_sr = loader.load_audio(tmp_file.name)
            
            assert loaded_sr == config.data.sample_rate
            assert len(loaded_audio) > 0
            assert isinstance(loaded_audio, np.ndarray)
            
            # Clean up
            Path(tmp_file.name).unlink()


class TestFeatureExtractor:
    """Test FeatureExtractor class."""
    
    def test_feature_extractor_init(self, config):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor(config)
        assert extractor.frame_size == config.data.frame_size
        assert extractor.hop_length == config.data.hop_length
        assert extractor.n_fft == config.data.n_fft
        assert extractor.n_mels == config.data.n_mels
    
    def test_extract_energy(self, config, sample_audio):
        """Test energy extraction."""
        extractor = FeatureExtractor(config)
        audio, sr = sample_audio
        
        energy = extractor.extract_energy(audio)
        
        assert isinstance(energy, np.ndarray)
        assert len(energy) > 0
        assert np.all(energy >= 0)  # Energy should be non-negative
    
    def test_extract_spectral_features(self, config, sample_audio):
        """Test spectral feature extraction."""
        extractor = FeatureExtractor(config)
        audio, sr = sample_audio
        
        features = extractor.extract_spectral_features(audio)
        
        assert isinstance(features, dict)
        assert 'spectral_centroid' in features
        assert 'zcr' in features
        assert 'mfcc' in features
        assert 'magnitude' in features
    
    def test_extract_mel_spectrogram(self, config, sample_audio):
        """Test mel spectrogram extraction."""
        extractor = FeatureExtractor(config)
        audio, sr = sample_audio
        
        mel_spec = extractor.extract_mel_spectrogram(audio)
        
        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.shape[0] == config.data.n_mels
        assert len(mel_spec.shape) == 2


class TestVAD:
    """Test VAD implementations."""
    
    def test_energy_vad_init(self, config):
        """Test EnergyBasedVAD initialization."""
        vad = EnergyBasedVAD(config)
        assert vad.energy_threshold == config.vad.energy_threshold
        assert vad.min_speech_duration == config.vad.min_speech_duration
    
    def test_energy_vad_detect(self, config, sample_audio):
        """Test energy-based VAD detection."""
        vad = EnergyBasedVAD(config)
        audio, sr = sample_audio
        
        vad_result, metadata = vad.detect(audio, sr)
        
        assert isinstance(vad_result, np.ndarray)
        assert len(vad_result) > 0
        assert np.all(np.isin(vad_result, [0, 1]))  # Binary values
        assert 'energy' in metadata
        assert 'vad_raw' in metadata
    
    def test_spectral_vad_init(self, config):
        """Test SpectralBasedVAD initialization."""
        vad = SpectralBasedVAD(config)
        assert vad.spectral_threshold == config.vad.spectral_threshold
        assert vad.zcr_threshold == config.vad.zero_crossing_threshold
    
    def test_spectral_vad_detect(self, config, sample_audio):
        """Test spectral-based VAD detection."""
        vad = SpectralBasedVAD(config)
        audio, sr = sample_audio
        
        vad_result, metadata = vad.detect(audio, sr)
        
        assert isinstance(vad_result, np.ndarray)
        assert len(vad_result) > 0
        assert np.all(np.isin(vad_result, [0, 1]))  # Binary values
        assert 'spectral_centroid' in metadata
        assert 'zcr' in metadata
    
    def test_ml_vad_init(self, config):
        """Test MLBasedVAD initialization."""
        vad = MLBasedVAD(config)
        assert not vad.is_trained
        assert vad.model is not None
        assert vad.scaler is not None


class TestModels:
    """Test neural network models."""
    
    def test_tcn_model_init(self, config):
        """Test TCN model initialization."""
        model = TCNBasedSegmentation(config)
        assert model.config == config
        assert isinstance(model.input_proj, torch.nn.Conv1d)
        assert len(model.tcn_blocks) == config.model.num_layers
    
    def test_tcn_model_forward(self, config, sample_audio):
        """Test TCN model forward pass."""
        model = TCNBasedSegmentation(config)
        audio, sr = sample_audio
        
        # Create mel spectrogram
        extractor = FeatureExtractor(config)
        mel_spec = extractor.extract_mel_spectrogram(audio)
        
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
        
        # Forward pass
        output = model(mel_tensor)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1  # Batch size
        assert output.shape[1] == 1   # Output channels
        assert output.shape[2] == mel_spec.shape[1]  # Time dimension
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Probabilities
    
    def test_cnn_model_init(self, config):
        """Test CNN model initialization."""
        model = CNNBasedSegmentation(config)
        assert model.config == config
        assert len(model.conv_layers) > 0
        assert isinstance(model.fc_layers, torch.nn.Sequential)
    
    def test_transformer_model_init(self, config):
        """Test Transformer model initialization."""
        model = TransformerBasedSegmentation(config)
        assert model.config == config
        assert isinstance(model.input_proj, torch.nn.Linear)
        assert isinstance(model.transformer, torch.nn.TransformerEncoder)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_init(self, config):
        """Test SegmentationMetrics initialization."""
        metrics = SegmentationMetrics(config)
        assert metrics.config == config
        assert metrics.collar_tolerance == config.evaluation.collar_tolerance
    
    def test_compute_metrics(self, config, sample_audio):
        """Test metrics computation."""
        metrics = SegmentationMetrics(config)
        audio, sr = sample_audio
        
        # Create dummy predictions and ground truth
        predictions = np.random.randint(0, 2, 100)
        ground_truth = np.random.randint(0, 2, 100)
        
        result = metrics.compute_metrics(predictions, ground_truth, sr)
        
        assert isinstance(result, dict)
        assert 'frame_precision' in result
        assert 'frame_recall' in result
        assert 'frame_f1' in result
        assert 'der' in result
        assert 'jer' in result
    
    def test_metrics_aggregator(self):
        """Test MetricsAggregator."""
        aggregator = MetricsAggregator()
        
        # Add some metrics
        metrics1 = {'precision': 0.8, 'recall': 0.7, 'f1': 0.75}
        metrics2 = {'precision': 0.9, 'recall': 0.8, 'f1': 0.85}
        
        aggregator.add_metrics(metrics1)
        aggregator.add_metrics(metrics2)
        
        avg_metrics = aggregator.get_average_metrics()
        std_metrics = aggregator.get_std_metrics()
        
        assert 'precision' in avg_metrics
        assert 'recall' in avg_metrics
        assert 'f1' in avg_metrics
        assert avg_metrics['precision'] == 0.85  # (0.8 + 0.9) / 2
        assert avg_metrics['recall'] == 0.75    # (0.7 + 0.8) / 2
        assert avg_metrics['f1'] == 0.8          # (0.75 + 0.85) / 2


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device == torch.device("cpu")
        
        device = get_device("auto")
        assert device in [torch.device("cpu"), torch.device("cuda"), torch.device("mps")]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can verify it doesn't raise an error
        assert True
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(30.5) == "30.50s"
        assert format_duration(90.5) == "1m 30.50s"
        assert format_duration(3661.5) == "1h 1m 1.50s"
    
    def test_compute_rtf(self):
        """Test RTF computation."""
        rtf = compute_rtf(1.0, 2.0)
        assert rtf == 0.5
        
        rtf = compute_rtf(0.0, 1.0)
        assert rtf == float('inf')


if __name__ == "__main__":
    pytest.main([__file__])
