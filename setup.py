#!/usr/bin/env python3
"""Setup script for automatic speech segmentation project."""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Automatic Speech Segmentation Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return 1
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        return 1
    
    # Install development dependencies
    if not run_command("pip install -e \".[dev]\"", "Installing development dependencies"):
        print("⚠️  Development dependencies installation failed, but continuing...")
    
    # Create necessary directories
    directories = [
        "data/wav",
        "assets",
        "logs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create sample metadata file
    metadata_content = """id,filename,path,duration,sample_rate,split,speaker_id,language,quality
synth_001,synthetic_speech_001.wav,data/wav/synthetic_speech_001.wav,5.0,16000,train,speaker_1,en,synthetic
synth_002,synthetic_speech_002.wav,data/wav/synthetic_speech_002.wav,4.5,16000,train,speaker_2,en,synthetic
synth_003,synthetic_speech_003.wav,data/wav/synthetic_speech_003.wav,6.0,16000,val,speaker_1,en,synthetic
"""
    
    metadata_file = Path("data/meta.csv")
    if not metadata_file.exists():
        metadata_file.write_text(metadata_content)
        print("✅ Created sample metadata file: data/meta.csv")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but setup completed")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the example script: python scripts/example.py")
    print("2. Launch the demo: streamlit run demo/app.py")
    print("3. Process your own audio: python main.py --input your_audio.wav")
    print("\nFor more information, see README.md")
    
    return 0

if __name__ == "__main__":
    exit(main())
