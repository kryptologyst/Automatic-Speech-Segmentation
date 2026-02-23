#!/usr/bin/env python3
"""Example script demonstrating automatic speech segmentation."""

import logging
import numpy as np
import soundfile as sf
from pathlib import Path

from src import SpeechSegmentationPipeline
from src.utils import load_config, setup_logging, set_seed

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

def create_sample_audio(output_path: str, duration: float = 5.0, sample_rate: int = 16000):
    """Create a sample audio file with speech segments and pauses.
    
    Args:
        output_path: Path to save the audio file
        duration: Duration in seconds
        sample_rate: Sample rate
    """
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.zeros_like(t)
    
    # Create speech segments
    speech_segments = [
        (0.5, 1.5),   # First speech segment
        (2.0, 3.5),   # Second speech segment
        (4.0, 4.8),   # Third speech segment
    ]
    
    for start_time, end_time in speech_segments:
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        segment_t = t[start_idx:end_idx]
        
        # Create speech-like signal (sine wave with modulation)
        freq = 200 + 50 * np.sin(2 * np.pi * 0.5 * segment_t)
        speech_signal = np.sin(2 * np.pi * freq * segment_t) * np.exp(-segment_t * 0.5)
        
        audio[start_idx:end_idx] = speech_signal
    
    # Add some noise
    noise = np.random.normal(0, 0.05, len(audio))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save audio
    sf.write(output_path, audio, sample_rate)
    logger.info(f"Created sample audio: {output_path}")
    
    return audio, sample_rate

def main():
    """Main example function."""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config_path = "configs/config.yaml"
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    
    # Create sample audio
    sample_audio_path = "assets/sample_audio.wav"
    Path("assets").mkdir(exist_ok=True)
    
    audio, sample_rate = create_sample_audio(sample_audio_path)
    
    # Initialize pipeline
    logger.info("Initializing speech segmentation pipeline")
    pipeline = SpeechSegmentationPipeline(config)
    
    # Process audio
    logger.info("Processing audio...")
    results = pipeline.process_audio(sample_audio_path)
    
    # Display results
    print("\n" + "="*50)
    print("SPEECH SEGMENTATION RESULTS")
    print("="*50)
    
    print(f"Audio file: {results['audio_path']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Sample rate: {results['sample_rate']:,} Hz")
    print(f"Number of segments detected: {len(results['boundaries'])}")
    
    if results['boundaries']:
        print("\nDetected speech segments:")
        for i, (start_time, end_time) in enumerate(results['boundaries']):
            duration = end_time - start_time
            print(f"  Segment {i+1}: {start_time:.3f}s - {end_time:.3f}s (duration: {duration:.3f}s)")
    else:
        print("No speech segments detected.")
    
    # Display summary
    summary = pipeline.get_segmentation_summary(results)
    print(summary)
    
    # Save results
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    # Save segmentation results
    np.savez(
        output_dir / "segmentation_results.npz",
        audio=results['audio'],
        sample_rate=results['sample_rate'],
        vad_result=results['vad_result'],
        predictions=results['predictions'],
        boundaries=np.array(results['boundaries']),
        mel_spectrogram=results['mel_spectrogram']
    )
    
    # Save individual segments
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)
    
    for i, segment in enumerate(results['segments']):
        segment_path = segments_dir / f"segment_{i+1:02d}.wav"
        sf.write(segment_path, segment, results['sample_rate'])
        logger.info(f"Saved segment {i+1}: {segment_path}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Individual segments saved to: {segments_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
