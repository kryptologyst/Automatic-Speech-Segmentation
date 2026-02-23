"""Interactive demo for speech segmentation using Streamlit."""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import librosa
import librosa.display
from omegaconf import DictConfig

from src import SpeechSegmentationPipeline
from src.utils import load_config, setup_logging, set_seed

# Configure logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Automatic Speech Segmentation",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Privacy disclaimer
st.markdown("""
<div style="background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
<h4 style="color: #c62828; margin: 0;">Privacy & Ethics Notice</h4>
<p style="margin: 0.5rem 0 0 0; color: #424242;">
This is a research demonstration tool for automatic speech segmentation. 
It is designed for educational and research purposes only. 
<strong>Do not use for biometric identification or voice cloning in production.</strong>
All uploaded audio files are processed locally and not stored permanently.
</p>
</div>
""", unsafe_allow_html=True)

# Title and description
st.title("🎤 Automatic Speech Segmentation")
st.markdown("""
This demo showcases automatic speech segmentation using Voice Activity Detection (VAD) 
and advanced neural network models to identify speech boundaries in audio recordings.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Load configuration
config_path = Path("configs/config.yaml")
if config_path.exists():
    config = load_config(config_path)
else:
    st.error("Configuration file not found. Please ensure configs/config.yaml exists.")
    st.stop()

# VAD method selection
vad_method = st.sidebar.selectbox(
    "VAD Method",
    ["energy", "spectral", "ml"],
    index=0,
    help="Choose the Voice Activity Detection method"
)

# Model selection
model_method = st.sidebar.selectbox(
    "Segmentation Model",
    ["energy_based", "tcn", "cnn", "transformer"],
    index=0,
    help="Choose the segmentation model"
)

# Update config
config.vad.type = vad_method
config.model.name = model_method

# Advanced parameters
st.sidebar.subheader("Advanced Parameters")

# Energy threshold
energy_threshold = st.sidebar.slider(
    "Energy Threshold",
    min_value=0.01,
    max_value=0.1,
    value=config.vad.energy_threshold,
    step=0.01,
    help="Threshold for energy-based VAD"
)

# Min speech duration
min_speech_duration = st.sidebar.slider(
    "Min Speech Duration (s)",
    min_value=0.05,
    max_value=1.0,
    value=config.vad.min_speech_duration,
    step=0.05,
    help="Minimum duration for speech segments"
)

# Min silence duration
min_silence_duration = st.sidebar.slider(
    "Min Silence Duration (s)",
    min_value=0.05,
    max_value=1.0,
    value=config.vad.min_silence_duration,
    step=0.05,
    help="Minimum duration for silence segments"
)

# Update config with user parameters
config.vad.energy_threshold = energy_threshold
config.vad.min_speech_duration = min_speech_duration
config.vad.min_silence_duration = min_silence_duration

# Initialize pipeline
@st.cache_resource
def initialize_pipeline(config):
    """Initialize the segmentation pipeline."""
    set_seed(42)  # For reproducibility
    return SpeechSegmentationPipeline(config)

pipeline = initialize_pipeline(config)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Audio Input")
    
    # Audio upload
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload an audio file for segmentation analysis"
    )
    
    # Audio recording
    st.subheader("Or Record Audio")
    audio_bytes = st.audio(
        "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
        format="audio/wav"
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = Path(tmp_file.name)
        
        try:
            # Process audio
            with st.spinner("Processing audio..."):
                results = pipeline.process_audio(tmp_path)
            
            # Display results
            st.success(f"✅ Audio processed successfully!")
            
            # Audio information
            st.subheader("Audio Information")
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Duration", f"{results['duration']:.2f}s")
            with col_info2:
                st.metric("Sample Rate", f"{results['sample_rate']:,} Hz")
            with col_info3:
                st.metric("Segments", len(results['boundaries']))
            
            # Play original audio
            st.subheader("Original Audio")
            st.audio(str(tmp_path))
            
            # Visualization
            st.subheader("Segmentation Visualization")
            
            # Create visualization
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            
            # Plot 1: Audio waveform
            time = np.arange(len(results['audio'])) / results['sample_rate']
            axes[0].plot(time, results['audio'], alpha=0.7)
            axes[0].set_title("Audio Waveform")
            axes[0].set_ylabel("Amplitude")
            axes[0].grid(True, alpha=0.3)
            
            # Add segment boundaries
            for start_time, end_time in results['boundaries']:
                axes[0].axvspan(start_time, end_time, alpha=0.3, color='green', label='Speech Segment')
            
            # Plot 2: Mel spectrogram
            librosa.display.specshow(
                results['mel_spectrogram'],
                sr=results['sample_rate'],
                hop_length=config.data.hop_length,
                x_axis='time',
                y_axis='mel',
                ax=axes[1]
            )
            axes[1].set_title("Mel Spectrogram")
            axes[1].set_ylabel("Mel Frequency")
            
            # Plot 3: VAD and predictions
            frame_times = np.arange(len(results['vad_result'])) * results['frame_duration']
            axes[2].plot(frame_times, results['vad_result'], label='VAD', alpha=0.7)
            axes[2].plot(frame_times, results['predictions'], label='Model Predictions', alpha=0.7)
            axes[2].set_title("Voice Activity Detection & Predictions")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Probability")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Segmentation results
            st.subheader("Segmentation Results")
            
            if results['boundaries']:
                # Create segments table
                segments_data = []
                for i, (start_time, end_time) in enumerate(results['boundaries']):
                    duration = end_time - start_time
                    segments_data.append({
                        "Segment": i + 1,
                        "Start Time": f"{start_time:.3f}s",
                        "End Time": f"{end_time:.3f}s",
                        "Duration": f"{duration:.3f}s"
                    })
                
                st.table(segments_data)
                
                # Play individual segments
                st.subheader("Individual Segments")
                for i, segment in enumerate(results['segments']):
                    st.write(f"**Segment {i+1}** ({results['boundaries'][i][0]:.2f}s - {results['boundaries'][i][1]:.2f}s)")
                    
                    # Save segment to temporary file for playback
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as seg_file:
                        import soundfile as sf
                        sf.write(seg_file.name, segment, results['sample_rate'])
                        st.audio(seg_file.name)
            else:
                st.warning("No speech segments detected.")
            
            # Summary
            st.subheader("Summary")
            summary = pipeline.get_segmentation_summary(results)
            st.text(summary)
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            logger.error(f"Error processing audio: {e}")
        
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()

with col2:
    st.header("Model Information")
    
    # VAD method info
    st.subheader("VAD Method")
    if vad_method == "energy":
        st.info("""
        **Energy-based VAD**
        - Uses signal energy to detect speech
        - Simple and fast
        - Good for clean recordings
        """)
    elif vad_method == "spectral":
        st.info("""
        **Spectral-based VAD**
        - Uses spectral features (centroid, rolloff, ZCR)
        - More robust to noise
        - Better for noisy environments
        """)
    elif vad_method == "ml":
        st.info("""
        **ML-based VAD**
        - Uses machine learning classifier
        - Requires training data
        - Most accurate but needs training
        """)
    
    # Model info
    st.subheader("Segmentation Model")
    if model_method == "energy_based":
        st.info("""
        **Energy-based Segmentation**
        - Uses VAD output directly
        - Fast and simple
        - Good baseline method
        """)
    elif model_method == "tcn":
        st.info("""
        **TCN Model**
        - Temporal Convolutional Network
        - Good for sequence modeling
        - Efficient for real-time processing
        """)
    elif model_method == "cnn":
        st.info("""
        **CNN Model**
        - Convolutional Neural Network
        - Good for local patterns
        - Efficient feature extraction
        """)
    elif model_method == "transformer":
        st.info("""
        **Transformer Model**
        - Attention-based architecture
        - Best for complex patterns
        - Requires more computation
        """)
    
    # Parameters
    st.subheader("Current Parameters")
    st.json({
        "Energy Threshold": energy_threshold,
        "Min Speech Duration": f"{min_speech_duration}s",
        "Min Silence Duration": f"{min_silence_duration}s",
        "Sample Rate": f"{config.data.sample_rate} Hz",
        "Frame Size": config.data.frame_size,
        "Hop Length": config.data.hop_length
    })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
<p>Automatic Speech Segmentation Demo | Research & Educational Use Only</p>
<p>For questions or issues, please refer to the project documentation.</p>
</div>
""", unsafe_allow_html=True)
