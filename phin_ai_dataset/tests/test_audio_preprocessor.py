#!/usr/bin/env python3
"""
Unit tests for Audio Preprocessor module
Tests audio processing pipeline functionality
"""

import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
import shutil
import os

# Add source code to path for imports
import sys
sys.path.append('../source_code/scripts')

from audio_preprocessor import AudioPreprocessor

class TestAudioPreprocessor(unittest.TestCase):
    """Test cases for AudioPreprocessor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "input"
        self.output_dir = Path(self.test_dir) / "output"
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create test preprocessor
        self.preprocessor = AudioPreprocessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            sample_rate=22050  # Lower sample rate for faster tests
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test AudioPreprocessor initialization"""
        self.assertEqual(self.preprocessor.input_dir, self.input_dir)
        self.assertEqual(self.preprocessor.output_dir, self.output_dir)
        self.assertEqual(self.preprocessor.sample_rate, 22050)
        self.assertEqual(self.preprocessor.segment_duration, 5.0)
    
    def test_create_sample_phin_audio(self):
        """Test sample audio creation"""
        sample_file = self.preprocessor.create_sample_phin_audio("test_sample.wav")
        
        self.assertTrue(Path(sample_file).exists())
        self.assertEqual(Path(sample_file).name, "test_sample.wav")
        
        # Test that file is not empty
        self.assertGreater(Path(sample_file).stat().st_size, 0)
    
    def test_apply_lowpass_filter(self):
        """Test low-pass filter application"""
        # Create test signal with high frequency component
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create signal with both low and high frequency components
        signal = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)
        
        # Apply low-pass filter
        filtered = self.preprocessor.apply_lowpass_filter(signal, cutoff=1000)
        
        # Check that filtered signal has same length
        self.assertEqual(len(signal), len(filtered))
        
        # Check that signal is modified (filter applied)
        self.assertFalse(np.array_equal(signal, filtered))
    
    def test_clean_audio(self):
        """Test audio cleaning functionality"""
        # Create test audio with DC offset and noise
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create signal with DC offset
        audio = np.sin(2 * np.pi * 440 * t) + 0.1
        
        # Add some low-level noise
        noise = 0.005 * np.random.normal(0, 1, len(audio))
        audio += noise
        
        # Clean the audio
        cleaned = self.preprocessor.clean_audio(audio)
        
        # Check that DC offset is removed (mean should be close to 0)
        self.assertAlmostEqual(np.mean(cleaned), 0.0, places=2)
        
        # Check that audio is normalized
        max_val = np.max(np.abs(cleaned))
        self.assertLessEqual(max_val, 0.95)
        self.assertGreater(max_val, 0.1)
    
    def test_segment_audio(self):
        """Test audio segmentation"""
        # Create test audio (5 seconds)
        sample_rate = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        # Segment with 1-second duration
        segments = self.preprocessor.segment_audio(audio, segment_duration=1.0)
        
        # Should create approximately 5 segments
        self.assertGreaterEqual(len(segments), 4)
        self.assertLessEqual(len(segments), 6)
        
        # Check segment length
        for segment in segments:
            self.assertEqual(len(segment), sample_rate)  # 1 second at 22050 Hz
    
    def test_extract_features(self):
        """Test feature extraction"""
        # Create test audio
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        # Extract features
        features = self.preprocessor.extract_features(audio)
        
        # Check required features are present
        required_features = [
            "duration", "rms_energy", "zero_crossing_rate",
            "spectral_centroid_mean", "spectral_centroid_std",
            "mfcc_means", "mfcc_stds", "chroma_energy", "estimated_tempo"
        ]
        
        for feature in required_features:
            self.assertIn(feature, features)
        
        # Check feature values are reasonable
        self.assertEqual(features["duration"], duration)
        self.assertGreater(features["rms_energy"], 0)
        self.assertGreaterEqual(features["zero_crossing_rate"], 0)
        self.assertEqual(len(features["mfcc_means"]), 13)
        self.assertEqual(len(features["mfcc_stds"]), 13)
        self.assertEqual(len(features["chroma_energy"]), 12)
    
    def test_process_single_file_success(self):
        """Test successful processing of a single file"""
        # Create a sample audio file
        sample_file = self.preprocessor.create_sample_phin_audio("test_audio.wav")
        
        # Process the file
        result = self.preprocessor.process_single_file(sample_file)
        
        # Check result structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["original_file"], sample_file)
        self.assertIn("full_audio_features", result)
        self.assertIn("segments", result)
        self.assertIn("segment_count", result)
        
        # Check that segments were created
        self.assertGreater(result["segment_count"], 0)
        self.assertEqual(len(result["segments"]), result["segment_count"])
    
    def test_process_single_file_failure(self):
        """Test handling of invalid file"""
        # Try to process non-existent file
        result = self.preprocessor.process_single_file("non_existent_file.wav")
        
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)
        self.assertEqual(result["original_file"], "non_existent_file.wav")
    
    def test_process_dataset_empty(self):
        """Test processing empty dataset"""
        # Process empty directory
        results = self.preprocessor.process_dataset()
        
        # Should create sample audio and process it
        self.assertGreater(results["total_files"], 0)
        self.assertGreater(results["successful"], 0)
        self.assertEqual(results["failed"], 0)
    
    def test_process_dataset_with_files(self):
        """Test processing dataset with existing files"""
        # Create multiple sample files
        for i in range(3):
            self.preprocessor.create_sample_phin_audio(f"sample_{i}.wav")
        
        # Process dataset
        results = self.preprocessor.process_dataset()
        
        # Should process at least 3 files (might be more if process_dataset creates extras)
        self.assertGreaterEqual(results["total_files"], 3)
        self.assertGreaterEqual(results["successful"], 3)
        self.assertEqual(results["failed"], 0)
        
        # Check that results file was created
        results_file = self.output_dir / "processing_results.json"
        self.assertTrue(results_file.exists())
        
        # Check results file content
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
        
        self.assertGreaterEqual(len(saved_results), 3)


class TestAudioPreprocessorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.preprocessor = AudioPreprocessor(
            input_dir=f"{self.test_dir}/input",
            output_dir=f"{self.test_dir}/output",
            sample_rate=16000
        )
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_silent_audio(self):
        """Test handling of silent audio"""
        silent_audio = np.zeros(16000)  # 1 second of silence
        
        # Should not crash
        features = self.preprocessor.extract_features(silent_audio)
        
        # Check that features are extracted (even if values are 0)
        self.assertIn("duration", features)
        self.assertEqual(features["duration"], 1.0)
    
    def test_very_short_audio(self):
        """Test handling of very short audio"""
        short_audio = np.random.normal(0, 0.1, 100)  # Very short audio
        
        # Should handle gracefully
        features = self.preprocessor.extract_features(short_audio)
        
        self.assertIn("duration", features)
        self.assertEqual(features["duration"], 100/16000)
    
    def test_invalid_audio_path(self):
        """Test handling of invalid audio path"""
        with self.assertRaises(Exception):
            self.preprocessor.load_audio("invalid_path.wav")


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAudioPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioPreprocessorEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)