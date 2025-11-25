#!/usr/bin/env python3
"""
Enhanced unit tests for refactored audio preprocessing pipeline
"""

import unittest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import soundfile as sf
import librosa

# Add parent directory to path
import sys
sys.path.append('..')
sys.path.append('../source_code/scripts')

from refactored_audio_preprocessor import (
    AudioConfig, ProcessingResult, AudioValidator, AudioCleaner, 
    AudioSegmenter, FeatureExtractor, RefactoredAudioPreprocessor
)

class TestAudioConfig(unittest.TestCase):
    """Test AudioConfig dataclass"""
    
    def test_default_config(self):
        config = AudioConfig()
        self.assertEqual(config.sample_rate, 44100)
        self.assertEqual(config.segment_duration, 5.0)
        self.assertEqual(config.n_mfcc, 13)
    
    def test_custom_config(self):
        config = AudioConfig(
            sample_rate=22050,
            segment_duration=3.0,
            noise_threshold=0.05
        )
        self.assertEqual(config.sample_rate, 22050)
        self.assertEqual(config.segment_duration, 3.0)
        self.assertEqual(config.noise_threshold, 0.05)

class TestAudioValidator(unittest.TestCase):
    """Test AudioValidator class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.validator = AudioValidator()
        self.config = AudioConfig()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_validate_valid_audio_file(self):
        # Create a valid audio file
        test_file = Path(self.temp_dir) / "test.wav"
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        sf.write(test_file, audio, sample_rate)
        
        is_valid, message = self.validator.validate_audio_file(test_file)
        self.assertTrue(is_valid)
        self.assertIn("Valid", message)
    
    def test_validate_nonexistent_file(self):
        test_file = Path(self.temp_dir) / "nonexistent.wav"
        is_valid, message = self.validator.validate_audio_file(test_file)
        self.assertFalse(is_valid)
        self.assertIn("does not exist", message)
    
    def test_validate_empty_file(self):
        test_file = Path(self.temp_dir) / "empty.wav"
        test_file.touch()  # Create empty file
        
        is_valid, message = self.validator.validate_audio_file(test_file)
        self.assertFalse(is_valid)
        self.assertIn("empty", message.lower())
    
    def test_validate_silent_audio(self):
        # Create silent audio file
        test_file = Path(self.temp_dir) / "silent.wav"
        duration = 2.0
        sample_rate = 44100
        audio = np.zeros(int(sample_rate * duration))  # Silent audio
        sf.write(test_file, audio, sample_rate)
        
        is_valid, message = self.validator.validate_audio_file(test_file)
        self.assertFalse(is_valid)
        self.assertIn("silent", message.lower())
    
    def test_check_audio_quality(self):
        # Create audio with DC offset
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2  # Add DC offset
        
        quality_info = self.validator.check_audio_quality(audio, sample_rate, self.config)
        
        self.assertIn("dc_offset", quality_info["issues"])
        self.assertGreater(quality_info["dc_offset"], self.config.dc_offset_threshold)
    
    def test_check_clipping_audio(self):
        # Create clipped audio
        duration = 1.0
        sample_rate = 44100
        audio = np.ones(int(sample_rate * duration))  # Clipped at 1.0
        
        quality_info = self.validator.check_audio_quality(audio, sample_rate, self.config)
        
        self.assertIn("possible_clipping", quality_info["issues"])
        self.assertGreater(quality_info["max_amplitude"], self.config.clipping_threshold)

class TestAudioCleaner(unittest.TestCase):
    """Test AudioCleaner class"""
    
    def setUp(self):
        self.config = AudioConfig()
        self.cleaner = AudioCleaner(self.config)
    
    def test_clean_normal_audio(self):
        # Create normal audio
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        cleaned_audio, cleaning_info = self.cleaner.clean_audio(audio, sample_rate)
        
        self.assertEqual(len(cleaned_audio), len(audio))
        self.assertIn("steps_applied", cleaning_info)
        self.assertLessEqual(np.max(np.abs(cleaned_audio)), 0.8)  # Should be normalized
    
    def test_clean_audio_with_dc_offset(self):
        # Create audio with DC offset
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3  # Add DC offset
        
        cleaned_audio, cleaning_info = self.cleaner.clean_audio(audio, sample_rate)
        
        # DC offset should be removed
        self.assertLess(np.abs(np.mean(cleaned_audio)), 0.1)
        self.assertIn("dc_offset_removed", cleaning_info["steps_applied"])
    
    def test_clean_clipped_audio(self):
        # Create clipped audio (more realistic - sine wave with clipping)
        duration = 1.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a sine wave that will be clipped
        audio = 1.2 * np.sin(2 * np.pi * 440 * t)  # Amplitude > 1.0 to simulate clipping
        
        cleaned_audio, cleaning_info = self.cleaner.clean_audio(audio, sample_rate)
        
        # Should apply soft limiting
        self.assertLessEqual(np.max(np.abs(cleaned_audio)), 1.0)
        # Should have either normalized or soft limiting applied
        self.assertTrue(
            "normalized" in cleaning_info["steps_applied"] or 
            "soft_limiting_applied" in cleaning_info["steps_applied"]
        )

class TestAudioSegmenter(unittest.TestCase):
    """Test AudioSegmenter class"""
    
    def setUp(self):
        self.config = AudioConfig(segment_duration=1.0)  # Short segments for testing
        self.segmenter = AudioSegmenter(self.config)
    
    def test_segment_normal_audio(self):
        # Create 5-second audio
        duration = 5.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        segments = self.segmenter.segment_audio(audio, sample_rate)
        
        # Should create approximately 5 segments (allowing for overlap)
        self.assertGreaterEqual(len(segments), 4)
        self.assertLessEqual(len(segments), 6)
        
        # Check segment info
        for i, (segment, info) in enumerate(segments):
            self.assertEqual(info["index"], i)
            self.assertGreater(info["duration"], 0.4)  # At least 80% of segment duration
            self.assertLessEqual(info["end_time"], duration)
    
    def test_segment_short_audio(self):
        # Create audio shorter than segment duration
        duration = 0.8  # Less than segment_duration (1.0)
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        segments = self.segmenter.segment_audio(audio, sample_rate)
        
        # Should still create at least one segment
        self.assertGreaterEqual(len(segments), 1)
        
        # The segment should be the full audio
        if segments:
            segment, info = segments[0]
            self.assertAlmostEqual(info["duration"], duration, places=2)
    
    def test_segment_very_short_audio(self):
        # Create very short audio (less than minimum segment duration)
        duration = 0.1  # Much less than min_segment_duration (0.5 * segment_duration)
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        segments = self.segmenter.segment_audio(audio, sample_rate)
        
        # Should create one segment even if very short
        self.assertGreaterEqual(len(segments), 1)

class TestFeatureExtractor(unittest.TestCase):
    """Test FeatureExtractor class"""
    
    def setUp(self):
        self.config = AudioConfig()
        self.extractor = FeatureExtractor(self.config)
    
    def test_extract_features_normal_audio(self):
        # Create normal audio
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        features = self.extractor.extract_features(audio, sample_rate)
        
        # Check that all expected features are present
        expected_features = [
            "duration", "sample_rate", "samples", "rms_energy", 
            "zero_crossing_rate", "spectral_rolloff", "spectral_bandwidth",
            "spectral_centroid_mean", "spectral_centroid_std", 
            "mfcc_means", "mfcc_stds", "chroma_energy", "chroma_std",
            "estimated_tempo", "spectral_flatness", "spectral_contrast",
            "zcr_mean", "zcr_std"
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
        
        # Check feature values
        self.assertEqual(features["duration"], duration)
        self.assertEqual(features["sample_rate"], sample_rate)
        self.assertEqual(len(features["mfcc_means"]), self.config.n_mfcc)
        self.assertEqual(len(features["chroma_energy"]), self.config.n_chroma)
    
    def test_extract_features_silent_audio(self):
        # Create silent audio
        duration = 2.0
        sample_rate = 44100
        audio = np.zeros(int(sample_rate * duration))
        
        features = self.extractor.extract_features(audio, sample_rate)
        
        # Silent audio should have zero RMS energy
        self.assertEqual(features["rms_energy"], 0.0)
        self.assertEqual(features["zero_crossing_rate"], 0.0)
    
    def test_extract_features_very_short_audio(self):
        # Create very short audio
        duration = 0.1
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Should handle very short audio gracefully
        features = self.extractor.extract_features(audio, sample_rate)
        
        self.assertEqual(features["duration"], duration)
        self.assertGreater(len(features), 10)  # Should still extract features

class TestRefactoredAudioPreprocessor(unittest.TestCase):
    """Test RefactoredAudioPreprocessor class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        self.config = AudioConfig(sample_rate=22050, segment_duration=1.0)
        self.preprocessor = RefactoredAudioPreprocessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            config=self.config
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_process_single_valid_file(self):
        # Create a valid audio file
        test_file = self.input_dir / "test.wav"
        duration = 3.0
        sample_rate = self.config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_file, audio, sample_rate)
        
        result = self.preprocessor.process_single_file(test_file)
        
        self.assertEqual(result.status, "success")
        self.assertEqual(result.file_path, str(test_file))
        self.assertIsNotNone(result.features)
        self.assertGreater(len(result.segments), 0)
        self.assertIn("message", result.__dict__)
    
    def test_process_single_invalid_file(self):
        # Create an invalid file
        test_file = self.input_dir / "invalid.wav"
        test_file.write_text("This is not audio data")
        
        result = self.preprocessor.process_single_file(test_file)
        
        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.__dict__)
        self.assertIsNotNone(result.error)
    
    def test_process_single_nonexistent_file(self):
        test_file = self.input_dir / "nonexistent.wav"
        
        result = self.preprocessor.process_single_file(test_file)
        
        self.assertEqual(result.status, "failed")
        self.assertIn("error", result.__dict__)
    
    def test_process_empty_dataset(self):
        # Test with empty input directory
        results = self.preprocessor.process_dataset()
        
        self.assertEqual(results["total_files"], 0)
        self.assertEqual(results["successful"], 0)
        self.assertEqual(results["failed"], 0)
    
    def test_process_dataset_with_files(self):
        # Create multiple test files
        for i in range(3):
            test_file = self.input_dir / f"test_{i}.wav"
            duration = 2.0
            sample_rate = self.config.sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * (440 + i * 50) * t)  # Different frequencies
            sf.write(test_file, audio, sample_rate)
        
        results = self.preprocessor.process_dataset()
        
        self.assertEqual(results["total_files"], 3)
        self.assertEqual(results["successful"], 3)
        self.assertEqual(results["failed"], 0)
        
        # Check that segments were created
        segment_files = list(self.output_dir.glob("*_segment_*.wav"))
        self.assertGreater(len(segment_files), 0)
        
        # Check that results file was created
        results_file = self.output_dir / "processing_results.json"
        self.assertTrue(results_file.exists())
    
    def test_process_dataset_with_mixed_files(self):
        # Create mix of valid and invalid files
        
        # Valid file
        valid_file = self.input_dir / "valid.wav"
        duration = 2.0
        sample_rate = self.config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(valid_file, audio, sample_rate)
        
        # Invalid file
        invalid_file = self.input_dir / "invalid.wav"
        invalid_file.write_text("Not audio data")
        
        results = self.preprocessor.process_dataset()
        
        self.assertEqual(results["total_files"], 2)
        self.assertEqual(results["successful"], 1)
        self.assertEqual(results["failed"], 1)
    
    def test_progress_callback(self):
        # Create a test file
        test_file = self.input_dir / "test.wav"
        duration = 2.0
        sample_rate = self.config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_file, audio, sample_rate)
        
        # Track progress callback calls
        callback_calls = []
        
        def test_callback(completed, total, result):
            callback_calls.append((completed, total, result.status))
        
        results = self.preprocessor.process_dataset(progress_callback=test_callback)
        
        # Should have called callback at least once
        self.assertGreater(len(callback_calls), 0)
        
        # Check callback parameters
        for completed, total, status in callback_calls:
            self.assertGreaterEqual(completed, 1)
            self.assertEqual(total, 1)
            self.assertIn(status, ["success", "failed"])

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        self.config = AudioConfig()
        self.preprocessor = RefactoredAudioPreprocessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            config=self.config
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_very_long_audio_file(self):
        # Create a very long audio file (but short for testing)
        test_file = self.input_dir / "long.wav"
        duration = 10.0  # Simulate long file
        sample_rate = self.config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_file, audio, sample_rate)
        
        result = self.preprocessor.process_single_file(test_file)
        
        # Should handle long files gracefully
        self.assertEqual(result.status, "success")
        self.assertGreaterEqual(len(result.segments), 2)  # Should create multiple segments
    
    def test_audio_with_extreme_values(self):
        # Create audio with extreme values
        test_file = self.input_dir / "extreme.wav"
        duration = 2.0
        sample_rate = self.config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.ones(int(sample_rate * duration)) * 1000  # Very large values
        sf.write(test_file, audio, sample_rate)
        
        result = self.preprocessor.process_single_file(test_file)
        
        # Should handle extreme values gracefully
        self.assertEqual(result.status, "success")
    
    def test_corrupted_audio_file(self):
        # Create a corrupted WAV file
        test_file = self.input_dir / "corrupted.wav"
        with open(test_file, 'wb') as f:
            f.write(b'This is not a valid WAV file header')
            f.write(b'corrupted data' * 1000)
        
        result = self.preprocessor.process_single_file(test_file)
        
        # Should fail gracefully
        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error)
    
    def test_unicode_filename(self):
        # Create file with unicode name
        test_file = self.input_dir / "ทดสอบ_ phin_音频.wav"
        duration = 2.0
        sample_rate = self.config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_file, audio, sample_rate)
        
        result = self.preprocessor.process_single_file(test_file)
        
        # Should handle unicode filenames
        self.assertEqual(result.status, "success")
    
    def test_concurrent_processing(self):
        # Create multiple files for concurrent processing
        for i in range(5):
            test_file = self.input_dir / f"concurrent_{i}.wav"
            duration = 1.5
            sample_rate = self.config.sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * (440 + i * 30) * t)
            sf.write(test_file, audio, sample_rate)
        
        # Process with multiple workers
        results = self.preprocessor.process_dataset(max_workers=3)
        
        self.assertEqual(results["total_files"], 5)
        self.assertEqual(results["successful"], 5)

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAudioConfig,
        TestAudioValidator,
        TestAudioCleaner,
        TestAudioSegmenter,
        TestFeatureExtractor,
        TestRefactoredAudioPreprocessor,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "N/A")