#!/usr/bin/env python3
"""
Refactored Audio Preprocessing Pipeline for Phin AI Dataset
Improved modularity, error handling, and performance
"""

import os
import json
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    sample_rate: int = 44100
    segment_duration: float = 5.0
    noise_threshold: float = 0.01
    dc_offset_threshold: float = 0.1
    clipping_threshold: float = 0.95
    min_segment_duration: float = 0.5
    max_file_duration: float = 300.0
    n_mfcc: int = 13
    n_chroma: int = 12

@dataclass
class ProcessingResult:
    """Result of audio processing"""
    file_path: str
    status: str  # 'success', 'failed', 'skipped'
    message: str = ""
    features: Optional[Dict] = None
    segments: List[Dict] = None
    error: Optional[str] = None

class AudioValidator:
    """Audio file validation utilities"""
    
    @staticmethod
    def validate_audio_file(file_path: Path) -> Tuple[bool, str]:
        """Validate audio file format and content"""
        try:
            # Check file exists
            if not file_path.exists():
                return False, f"File does not exist: {file_path}"
            
            # Check file size
            if file_path.stat().st_size == 0:
                return False, "File is empty"
            
            if file_path.stat().st_size < 1024:  # Less than 1KB
                return False, "File too small"
            
            # Try to load audio
            y, sr = librosa.load(file_path, sr=None, duration=1.0)  # Load first second
            
            # Check for silence
            if np.max(np.abs(y)) < 0.001:
                return False, "Audio appears to be silent"
            
            # Check duration
            duration = len(y) / sr
            if duration < 0.1:  # Less than 100ms
                return False, "Audio too short"
            
            return True, "Valid audio file"
            
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    @staticmethod
    def check_audio_quality(y: np.ndarray, sr: int, config: AudioConfig) -> Dict[str, Any]:
        """Check audio quality and identify issues"""
        issues = []
        warnings_list = []
        
        # Check for DC offset
        dc_offset = np.abs(np.mean(y))
        if dc_offset > config.dc_offset_threshold:
            issues.append("dc_offset")
        
        # Check for clipping
        max_amplitude = np.max(np.abs(y))
        if max_amplitude > config.clipping_threshold:
            issues.append("possible_clipping")
        
        # Check for very low signal
        rms_energy = np.sqrt(np.mean(y**2))
        if rms_energy < config.noise_threshold:
            issues.append("low_signal")
        
        # Check duration
        duration = len(y) / sr
        if duration < 1.0:
            warnings_list.append("very_short")
        elif duration > config.max_file_duration:
            warnings_list.append("very_long")
        
        return {
            "issues": issues,
            "warnings": warnings_list,
            "duration": duration,
            "sample_rate": sr,
            "dc_offset": float(dc_offset),
            "max_amplitude": float(max_amplitude),
            "rms_energy": float(rms_energy)
        }

class AudioCleaner:
    """Audio cleaning and enhancement utilities"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def clean_audio(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Clean audio with multiple techniques"""
        logger.info("🧹 Cleaning audio...")
        
        original_y = y.copy()
        cleaning_steps = []
        
        # Step 1: Remove DC offset
        if np.abs(np.mean(y)) > self.config.dc_offset_threshold:
            y = y - np.mean(y)
            cleaning_steps.append("dc_offset_removed")
        
        # Step 2: Normalize
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val * 0.8  # Leave some headroom
            cleaning_steps.append("normalized")
        
        # Step 3: Gentle noise gate (very conservative)
        rms_energy = np.sqrt(np.mean(y**2))
        if rms_energy < self.config.noise_threshold * 10:  # Very quiet
            # Apply gentle noise gate
            mask = np.abs(y) > (self.config.noise_threshold * 5)
            y = y * mask
            cleaning_steps.append("noise_gate_applied")
        
        # Step 4: Anti-clipping (soft limiting)
        if np.max(np.abs(y)) > 0.9:
            # Apply soft limiting instead of hard clipping
            y = np.tanh(y * 2) / 2
            cleaning_steps.append("soft_limiting_applied")
        
        logger.info(f"✅ Audio cleaning completed: {cleaning_steps}")
        
        return y, {
            "steps_applied": cleaning_steps,
            "original_max": float(np.max(np.abs(original_y))),
            "cleaned_max": float(np.max(np.abs(y))),
            "original_mean": float(np.mean(original_y)),
            "cleaned_mean": float(np.mean(y))
        }

class AudioSegmenter:
    """Audio segmentation utilities"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def segment_audio(self, y: np.ndarray, sr: int) -> List[Tuple[np.ndarray, Dict]]:
        """Segment audio into chunks with overlap"""
        logger.info(f"🎯 Segmenting audio into {self.config.segment_duration}s chunks...")
        
        segment_samples = int(sr * self.config.segment_duration)
        min_segment_samples = int(segment_samples * self.config.min_segment_duration)
        
        segments = []
        overlap = int(segment_samples * 0.1)  # 10% overlap
        
        # Handle very short audio
        if len(y) < min_segment_samples:
            logger.warning(f"Audio too short for segmentation ({len(y)/sr:.2f}s < {self.config.segment_duration * self.config.min_segment_duration:.2f}s)")
            # Return the whole audio as one segment
            segment_info = {
                "index": 0,
                "start_time": 0.0,
                "end_time": len(y) / sr,
                "duration": len(y) / sr,
                "overlap": 0.0
            }
            segments.append((y, segment_info))
            logger.info(f"Created 1 segment from short audio")
            return segments
        
        start = 0
        segment_idx = 0
        
        while start < len(y):
            end = min(start + segment_samples, len(y))
            segment = y[start:end]
            
            # Check if segment is long enough
            if len(segment) >= min_segment_samples:
                segment_info = {
                    "index": segment_idx,
                    "start_time": start / sr,
                    "end_time": end / sr,
                    "duration": len(segment) / sr,
                    "overlap": overlap / sr if segment_idx > 0 else 0
                }
                segments.append((segment, segment_info))
                segment_idx += 1
            
            # Move to next segment with overlap
            start = end - overlap if end < len(y) else end
            
            # Break if remaining audio is too short
            if len(y) - start < min_segment_samples:
                break
        
        logger.info(f"Created {len(segments)} segments")
        return segments

class FeatureExtractor:
    """Audio feature extraction utilities"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def extract_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive audio features"""
        logger.info("📊 Extracting audio features...")
        
        features = {}
        
        # Basic features
        features["duration"] = len(y) / sr
        features["sample_rate"] = sr
        features["samples"] = len(y)
        
        # Time-domain features
        features["rms_energy"] = float(np.sqrt(np.mean(y**2)))
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.n_mfcc)
        features["mfcc_means"] = [float(np.mean(mfcc)) for mfcc in mfccs]
        features["mfcc_stds"] = [float(np.std(mfcc)) for mfcc in mfccs]
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.config.n_chroma)
        features["chroma_energy"] = [float(np.mean(chroma[i])) for i in range(chroma.shape[0])]
        features["chroma_std"] = [float(np.std(chroma[i])) for i in range(chroma.shape[0])]
        
        # Tempo and rhythm
        try:
            tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
            features["estimated_tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        except Exception as e:
            logger.warning(f"Could not estimate tempo: {e}")
            features["estimated_tempo"] = 0.0
        
        # Additional spectral features
        features["spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        features["spectral_contrast"] = [float(np.mean(sc)) for sc in librosa.feature.spectral_contrast(y=y, sr=sr)]
        
        # Zero crossing rate statistics
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        
        logger.info(f"✅ Extracted {len(features)} features")
        return features

class RefactoredAudioPreprocessor:
    """Refactored audio preprocessor with improved modularity"""
    
    def __init__(self, input_dir: str = "audio_sources/raw_audio", 
                 output_dir: str = "audio_sources/preprocessed",
                 config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.validator = AudioValidator()
        self.cleaner = AudioCleaner(self.config)
        self.segmenter = AudioSegmenter(self.config)
        self.extractor = FeatureExtractor(self.config)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized RefactoredAudioPreprocessor")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
    
    def process_single_file(self, file_path: Path) -> ProcessingResult:
        """Process a single audio file with comprehensive error handling"""
        logger.info(f"🔄 Processing: {file_path.name}")
        
        try:
            # Validate file
            is_valid, validation_msg = self.validator.validate_audio_file(file_path)
            if not is_valid:
                return ProcessingResult(
                    file_path=str(file_path),
                    status="failed",
                    message=f"Validation failed: {validation_msg}",
                    error=validation_msg
                )
            
            # Load audio
            y, sr = librosa.load(file_path, sr=self.config.sample_rate)
            
            # Check quality
            quality_info = self.validator.check_audio_quality(y, sr, self.config)
            if quality_info["issues"]:
                logger.warning(f"Audio quality issues in {file_path.name}: {quality_info['issues']}")
            
            # Clean audio
            y_cleaned, cleaning_info = self.cleaner.clean_audio(y, sr)
            
            # Extract features from full audio
            full_features = self.extractor.extract_features(y_cleaned, sr)
            
            # Segment audio
            segments = self.segmenter.segment_audio(y_cleaned, sr)
            
            # Process each segment
            segment_results = []
            for i, (segment, segment_info) in enumerate(segments):
                segment_features = self.extractor.extract_features(segment, sr)
                
                # Save segment
                segment_filename = f"{file_path.stem}_segment_{i:03d}.wav"
                segment_path = self.output_dir / segment_filename
                sf.write(segment_path, segment, sr)
                
                segment_results.append({
                    "filename": segment_filename,
                    "features": segment_features,
                    "info": segment_info
                })
            
            return ProcessingResult(
                file_path=str(file_path),
                status="success",
                message=f"Processed {len(segments)} segments",
                features=full_features,
                segments=segment_results
            )
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return ProcessingResult(
                file_path=str(file_path),
                status="failed",
                message=error_msg,
                error=str(e)
            )
    
    def process_dataset(self, max_workers: int = 4, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Process all audio files in the input directory with parallel processing"""
        logger.info("🔄 Processing audio dataset...")
        
        # Ensure input directory exists
        if not self.input_dir.exists():
            logger.warning(f"Input directory does not exist: {self.input_dir}")
            self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(self.input_dir.glob(ext))
            audio_files.extend(self.input_dir.glob(f"**/{ext}"))
        
        # Remove duplicates and sort
        audio_files = sorted(list(set(audio_files)))
        
        if not audio_files:
            logger.warning("No audio files found")
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "results": []
            }
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process files in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path 
                for file_path in audio_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(len(results), len(audio_files), result)
                    
                    logger.info(f"Completed {file_path.name}: {result.status}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
                    results.append(ProcessingResult(
                        file_path=str(file_path),
                        status="failed",
                        message=f"Processing error: {str(e)}",
                        error=str(e)
                    ))
        
        # Calculate statistics
        successful = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")
        
        # Save results
        results_data = []
        for result in results:
            result_dict = {
                "file_path": result.file_path,
                "status": result.status,
                "message": result.message,
                "features": result.features,
                "segments": result.segments
            }
            if result.error:
                result_dict["error"] = result.error
            results_data.append(result_dict)
        
        results_path = self.output_dir / "processing_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # Summary
        logger.info(f"🎯 Processing complete!")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved: {results_path}")
        
        return {
            "total_files": len(results),
            "successful": successful,
            "failed": failed,
            "results": results_data
        }

def main():
    """Main function for refactored audio preprocessing"""
    logger.info("🎵 Phin AI Dataset - Refactored Audio Preprocessing Pipeline")
    logger.info("=" * 70)
    
    # Create preprocessor with custom config
    config = AudioConfig(
        sample_rate=44100,
        segment_duration=5.0,
        noise_threshold=0.01,
        n_mfcc=13
    )
    
    preprocessor = RefactoredAudioPreprocessor(config=config)
    
    # Progress callback
    def progress_callback(completed, total, result):
        percentage = (completed / total) * 100
        logger.info(f"Progress: {completed}/{total} ({percentage:.1f}%) - {result.file_path}")
    
    # Process dataset
    results = preprocessor.process_dataset(max_workers=2, progress_callback=progress_callback)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files processed: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {(results['successful']/results['total_files']*100):.1f}%" if results['total_files'] > 0 else "N/A")

if __name__ == "__main__":
    main()