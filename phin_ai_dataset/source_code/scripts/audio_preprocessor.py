#!/usr/bin/env python3
"""
Audio preprocessing pipeline for Phin AI Dataset
Handles audio cleaning, segmentation, and feature extraction
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, 
                 input_dir: str = "audio_sources/raw_audio",
                 output_dir: str = "audio_sources/preprocessed",
                 sample_rate: int = 44100):
        """Initialize audio preprocessor"""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio processing parameters
        self.segment_duration = 5.0  # seconds
        self.hop_length = 512
        self.n_fft = 2048
        
        logger.info(f"AudioPreprocessor initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Sample rate: {self.sample_rate}")
    
    def create_sample_phin_audio(self, output_name: str = None) -> str:
        """Create sample Phin-like audio for testing when YouTube downloads fail"""
        logger.info("🎵 Creating sample Phin audio for testing...")
        
        # Generate unique filename if not provided
        if output_name is None:
            timestamp = int(np.datetime64('now').astype('datetime64[s]').astype(int))
            output_name = f"sample_phin_{timestamp}.wav"
        
        # Phin tuning (approximate Thai 7-tone system)
        # Based on Thai solfa notation mapping to Western notes
        phin_scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C-D-E-F-G-A-B
        
        # Typical Phin pattern (simplified)
        duration = 10.0  # 10 seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Create melody
        audio = np.zeros_like(t)
        note_duration = 0.8  # seconds per note
        note_samples = int(self.sample_rate * note_duration)
        
        # Simple Phin-style melody
        melody_pattern = [0, 2, 4, 5, 4, 2, 0, 1, 3, 5]  # Scale degrees
        
        for i, scale_degree in enumerate(melody_pattern):
            start_idx = i * note_samples
            end_idx = min(start_idx + note_samples, len(t))
            
            if start_idx < len(t):
                freq = phin_scale[scale_degree % len(phin_scale)]
                
                # Create note with envelope (attack, sustain, release)
                note_t = t[start_idx:end_idx] - t[start_idx]
                envelope = np.exp(-note_t * 2)  # Exponential decay
                
                # Fundamental frequency
                note = 0.3 * envelope * np.sin(2 * np.pi * freq * note_t)
                
                # Add harmonics (typical for plucked string instruments)
                note += 0.15 * envelope * np.sin(2 * np.pi * 2 * freq * note_t)  # 2nd harmonic
                note += 0.08 * envelope * np.sin(2 * np.pi * 3 * freq * note_t)  # 3rd harmonic
                note += 0.04 * envelope * np.sin(2 * np.pi * 4 * freq * note_t)  # 4th harmonic
                
                audio[start_idx:end_idx] = note
        
        # Add some noise to simulate recording conditions
        noise = 0.01 * np.random.normal(0, 1, len(audio))
        audio += noise
        
        # Apply gentle low-pass filter
        audio = self.apply_lowpass_filter(audio, cutoff=8000)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save sample audio
        output_path = self.input_dir / output_name
        sf.write(output_path, audio, self.sample_rate)
        
        logger.info(f"✅ Sample Phin audio created: {output_path}")
        return str(output_path)
    
    def apply_lowpass_filter(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply simple low-pass filter"""
        # Simple Butterworth filter approximation
        from scipy import signal
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, audio)
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            logger.info(f"Loaded audio: {len(audio)} samples at {sr} Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {str(e)}")
            raise
    
    def clean_audio(self, audio: np.ndarray) -> np.ndarray:
        """Clean audio by removing noise and normalizing"""
        logger.info("🔧 Cleaning audio...")
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # Simple noise gate
        threshold = 0.01
        audio[np.abs(audio) < threshold] = 0
        
        return audio
    
    def segment_audio(self, audio: np.ndarray, segment_duration: Optional[float] = None) -> List[np.ndarray]:
        """Segment audio into smaller chunks"""
        if segment_duration is None:
            segment_duration = self.segment_duration
        
        segment_samples = int(self.sample_rate * segment_duration)
        segments = []
        
        logger.info(f"🎯 Segmenting audio into {segment_duration}s chunks...")
        
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            if len(segment) >= segment_samples * 0.5:  # At least 50% of segment duration
                segments.append(segment)
        
        logger.info(f"Created {len(segments)} segments")
        return segments
    
    def extract_features(self, audio: np.ndarray) -> Dict:
        """Extract audio features for analysis"""
        logger.info("📊 Extracting audio features...")
        
        features = {}
        
        # Basic features
        features["duration"] = len(audio) / self.sample_rate
        features["rms_energy"] = float(np.sqrt(np.mean(audio**2)))
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))
        
        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features["mfcc_means"] = [float(np.mean(mfcc)) for mfcc in mfccs]
        features["mfcc_stds"] = [float(np.std(mfcc)) for mfcc in mfccs]
        
        # Chroma features (useful for pitch analysis)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features["chroma_energy"] = [float(np.mean(chroma[i])) for i in range(chroma.shape[0])]
        
        # Tempo estimation
        tempo = librosa.beat.beat_track(y=audio, sr=self.sample_rate)[0]
        features["estimated_tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        
        logger.info(f"✅ Extracted {len(features)} features")
        return features
    
    def process_single_file(self, audio_path: str) -> Dict:
        """Process a single audio file"""
        logger.info(f"🔄 Processing: {audio_path}")
        
        try:
            # Load audio
            audio, sr = self.load_audio(audio_path)
            
            # Clean audio
            audio = self.clean_audio(audio)
            
            # Extract features from full audio
            full_features = self.extract_features(audio)
            
            # Segment audio
            segments = self.segment_audio(audio)
            
            # Process each segment
            segment_results = []
            for i, segment in enumerate(segments):
                segment_features = self.extract_features(segment)
                
                # Save segment
                segment_filename = f"{Path(audio_path).stem}_segment_{i:03d}.wav"
                segment_path = self.output_dir / segment_filename
                sf.write(segment_path, segment, self.sample_rate)
                
                segment_results.append({
                    "filename": segment_filename,
                    "features": segment_features
                })
            
            result = {
                "original_file": str(audio_path),
                "full_audio_features": full_features,
                "segments": segment_results,
                "segment_count": len(segments),
                "status": "success"
            }
            
            logger.info(f"✅ Processed: {Path(audio_path).name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {str(e)}")
            return {
                "original_file": str(audio_path),
                "error": str(e),
                "status": "failed"
            }
    
    def process_dataset(self) -> Dict:
        """Process all audio files in the input directory"""
        logger.info("🔄 Processing audio dataset...")
        
        # Ensure input directory exists
        if not self.input_dir.exists():
            logger.warning(f"Input directory does not exist: {self.input_dir}")
            self.input_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created input directory: {self.input_dir}")
        
        # Find all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(self.input_dir.glob(ext))
            audio_files.extend(self.input_dir.glob(f"**/{ext}"))
        
        # Remove duplicates and sort
        audio_files = sorted(list(set(audio_files)))
        
        if not audio_files:
            logger.warning("No audio files found. Creating sample audio...")
            # Check if sample file already exists to avoid duplicates
            sample_files = list(self.input_dir.glob("sample_phin*.wav"))
            if not sample_files:
                sample_path = self.create_sample_phin_audio()
                audio_files = [Path(sample_path)]
            else:
                # Use existing sample file
                audio_files = sample_files
                logger.info(f"Using existing sample file(s): {[f.name for f in sample_files]}")
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process each file
        results = []
        for audio_file in audio_files:
            result = self.process_single_file(str(audio_file))
            results.append(result)
        
        # Save processing results
        results_path = self.output_dir / "processing_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Summary
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = len(results) - successful
        
        logger.info(f"🎯 Processing complete!")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved: {results_path}")
        
        return {
            "total_files": len(results),
            "successful": successful,
            "failed": failed,
            "results": results
        }

def main():
    """Main function for audio preprocessing"""
    logger.info("🎵 Phin AI Dataset - Audio Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # Create preprocessor
    preprocessor = AudioPreprocessor()
    
    # Process dataset
    results = preprocessor.process_dataset()
    
    # Print summary
    logger.info("\n🎯 Preprocessing Summary:")
    logger.info(f"Total files processed: {results['total_files']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    
    logger.info("\n🎯 Next steps:")
    logger.info("1. Check preprocessed audio files")
    logger.info("2. Review feature extraction results")
    logger.info("3. Prepare for transcription training")
    logger.info("4. Evaluate audio quality")

if __name__ == "__main__":
    main()