#!/usr/bin/env python3
"""Main script for automatic speech segmentation."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from src import SpeechSegmentationPipeline
from src.utils import load_config, setup_logging, set_seed, create_directories

logger = logging.getLogger(__name__)


def main():
    """Main function for speech segmentation."""
    parser = argparse.ArgumentParser(description="Automatic Speech Segmentation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="assets/output",
        help="Path to output directory"
    )
    parser.add_argument(
        "--vad-method", 
        type=str, 
        choices=["energy", "spectral", "ml"],
        help="VAD method to use (overrides config)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["energy_based", "tcn", "cnn", "transformer"],
        help="Segmentation model to use (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.vad_method:
        config.vad.type = args.vad_method
    if args.model:
        config.model.name = args.model
    
    # Create output directory
    output_dir = Path(args.output)
    create_directories([output_dir])
    
    # Initialize pipeline
    logger.info("Initializing speech segmentation pipeline")
    pipeline = SpeechSegmentationPipeline(config)
    
    # Process audio
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    logger.info(f"Processing audio file: {input_path}")
    
    try:
        # Process audio
        results = pipeline.process_audio(input_path)
        
        # Save results
        output_file = output_dir / f"{input_path.stem}_segmentation.npz"
        import numpy as np
        np.savez(
            output_file,
            audio=results['audio'],
            sample_rate=results['sample_rate'],
            vad_result=results['vad_result'],
            predictions=results['predictions'],
            boundaries=np.array(results['boundaries']),
            mel_spectrogram=results['mel_spectrogram']
        )
        
        logger.info(f"Results saved to: {output_file}")
        
        # Print summary
        summary = pipeline.get_segmentation_summary(results)
        print(summary)
        
        # Save summary to file
        summary_file = output_dir / f"{input_path.stem}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Summary saved to: {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
