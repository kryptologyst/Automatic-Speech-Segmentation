"""Main speech segmentation pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

from .data import AudioLoader, FeatureExtractor
from .features import VADFactory
from .models import SegmentationModelFactory, BoundaryDetector
from .metrics import SegmentationMetrics, MetricsAggregator
from .utils import get_device, Timer

logger = logging.getLogger(__name__)


class SpeechSegmentationPipeline:
    """Main pipeline for automatic speech segmentation."""
    
    def __init__(self, config: DictConfig):
        """Initialize segmentation pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = get_device(config.device)
        
        # Initialize components
        self.audio_loader = AudioLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        
        # Initialize VAD
        vad_type = getattr(config.vad, 'type', 'energy')
        self.vad = VADFactory.create_vad(vad_type, config)
        
        # Initialize segmentation model
        model_type = config.model.name
        self.segmentation_model = SegmentationModelFactory.create_model(model_type, config)
        self.segmentation_model.to(self.device)
        
        # Initialize boundary detector
        self.boundary_detector = BoundaryDetector(config)
        
        # Initialize metrics
        self.metrics_calculator = SegmentationMetrics(config)
        self.metrics_aggregator = MetricsAggregator()
        
        logger.info(f"Initialized speech segmentation pipeline with {vad_type} VAD and {model_type} model")
    
    def process_audio(self, audio_path: Union[str, Path]) -> Dict:
        """Process audio file for speech segmentation.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict containing segmentation results
        """
        with Timer("Audio Processing"):
            # Load audio
            audio, sample_rate = self.audio_loader.load_audio(audio_path)
            
            # Extract features
            mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
            spectral_features = self.feature_extractor.extract_spectral_features(audio)
            
            # Apply VAD
            vad_result, vad_metadata = self.vad.detect(audio, sample_rate)
            
            # Get segmentation predictions
            if self.config.model.name != "energy_based":
                predictions = self._get_model_predictions(mel_spec)
            else:
                predictions = vad_result
            
            # Detect boundaries
            boundaries = self.boundary_detector.detect_boundaries(predictions, sample_rate)
            
            # Convert frame indices to time
            hop_length = self.config.data.hop_length
            frame_duration = hop_length / sample_rate
            
            # Create results
            results = {
                'audio_path': str(audio_path),
                'audio': audio,
                'sample_rate': sample_rate,
                'duration': len(audio) / sample_rate,
                'mel_spectrogram': mel_spec,
                'spectral_features': spectral_features,
                'vad_result': vad_result,
                'vad_metadata': vad_metadata,
                'predictions': predictions,
                'boundaries': boundaries,
                'segments': self._create_segments(audio, boundaries, sample_rate),
                'frame_duration': frame_duration
            }
            
            logger.info(f"Processed audio: {Path(audio_path).name}, "
                       f"duration: {results['duration']:.2f}s, "
                       f"segments: {len(boundaries)}")
            
            return results
    
    def _get_model_predictions(self, mel_spec: np.ndarray) -> np.ndarray:
        """Get predictions from segmentation model.
        
        Args:
            mel_spec: Mel spectrogram
            
        Returns:
            np.ndarray: Model predictions
        """
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.segmentation_model(mel_tensor)
            predictions = predictions.squeeze().cpu().numpy()
        
        return predictions
    
    def _create_segments(self, audio: np.ndarray, boundaries: List[Tuple[float, float]], 
                        sample_rate: int) -> List[np.ndarray]:
        """Create audio segments from boundaries.
        
        Args:
            audio: Original audio signal
            boundaries: List of (start_time, end_time) tuples
            sample_rate: Sample rate
            
        Returns:
            List of audio segments
        """
        segments = []
        
        for start_time, end_time in boundaries:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            segment = audio[start_sample:end_sample]
            segments.append(segment)
        
        return segments
    
    def evaluate(self, audio_path: Union[str, Path], ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate segmentation performance against ground truth.
        
        Args:
            audio_path: Path to audio file
            ground_truth: Ground truth segmentation (binary array)
            
        Returns:
            Dict containing evaluation metrics
        """
        # Process audio
        results = self.process_audio(audio_path)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_metrics(
            results['predictions'], 
            ground_truth, 
            results['sample_rate']
        )
        
        # Add to aggregator
        self.metrics_aggregator.add_metrics(metrics)
        
        return metrics
    
    def batch_evaluate(self, audio_paths: List[Union[str, Path]], 
                      ground_truths: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            ground_truths: List of ground truth segmentations
            
        Returns:
            Dict containing average metrics
        """
        logger.info(f"Evaluating {len(audio_paths)} audio files")
        
        # Reset aggregator
        self.metrics_aggregator.reset()
        
        # Evaluate each file
        for audio_path, ground_truth in zip(audio_paths, ground_truths):
            try:
                self.evaluate(audio_path, ground_truth)
            except Exception as e:
                logger.error(f"Error evaluating {audio_path}: {e}")
                continue
        
        # Get average metrics
        avg_metrics = self.metrics_aggregator.get_average_metrics()
        std_metrics = self.metrics_aggregator.get_std_metrics()
        
        # Combine average and std metrics
        combined_metrics = {}
        for key in avg_metrics:
            combined_metrics[f"{key}_mean"] = avg_metrics[key]
            combined_metrics[f"{key}_std"] = std_metrics.get(key, 0)
        
        logger.info(f"Batch evaluation completed. Average F1: {avg_metrics.get('frame_f1', 0):.3f}")
        
        return combined_metrics
    
    def train_model(self, train_dataset, val_dataset=None) -> None:
        """Train the segmentation model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
        """
        logger.info("Training segmentation model")
        
        # This is a placeholder for model training
        # In a real implementation, you would:
        # 1. Set up optimizer and loss function
        # 2. Create data loaders
        # 3. Implement training loop
        # 4. Add validation and checkpointing
        
        logger.warning("Model training not implemented in this demo version")
    
    def save_model(self, output_path: Union[str, Path]) -> None:
        """Save the trained model.
        
        Args:
            output_path: Path to save model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.segmentation_model.state_dict(),
            'config': self.config,
        }, output_path)
        
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a trained model.
        
        Args:
            model_path: Path to saved model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_segmentation_summary(self, results: Dict) -> str:
        """Get a summary of segmentation results.
        
        Args:
            results: Results from process_audio
            
        Returns:
            str: Summary string
        """
        duration = results['duration']
        num_segments = len(results['boundaries'])
        avg_segment_duration = np.mean([end - start for start, end in results['boundaries']]) if results['boundaries'] else 0
        
        summary = f"""
Segmentation Summary:
- Audio duration: {duration:.2f} seconds
- Number of segments: {num_segments}
- Average segment duration: {avg_segment_duration:.2f} seconds
- VAD method: {getattr(self.config.vad, 'type', 'energy')}
- Model: {self.config.model.name}
"""
        
        return summary
