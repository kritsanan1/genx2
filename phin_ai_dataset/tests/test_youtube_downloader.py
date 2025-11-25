#!/usr/bin/env python3
"""
Unit tests for YouTube Downloader module
Tests video downloading and dataset management functionality
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
import os
from unittest.mock import patch, MagicMock

# Add source code to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../data_collection'))

from youtube_downloader import YouTubeDownloader

class TestYouTubeDownloader(unittest.TestCase):
    """Test cases for YouTubeDownloader class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Create downloader instance
        self.downloader = YouTubeDownloader(output_dir=str(self.output_dir))
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test YouTubeDownloader initialization"""
        self.assertEqual(self.downloader.output_dir, self.output_dir)
        self.assertIsInstance(self.downloader.phin_channels, dict)
        self.assertIsInstance(self.downloader.priority_patterns, list)
        self.assertGreater(len(self.downloader.phin_channels), 0)
        self.assertGreater(len(self.downloader.priority_patterns), 0)
    
    def test_phin_channels_content(self):
        """Test that Phin channels are properly configured"""
        expected_keys = [
            "ดุลย์เพลงพิณ", "M MUSIC GROUP", "สตีฟ ฐิติวัสส์", 
            "พิณอีสาน บ้านเฮา", "เพลงพิณดนตรีอีสาน"
        ]
        
        for key in expected_keys:
            self.assertIn(key, self.downloader.phin_channels)
            self.assertTrue(self.downloader.phin_channels[key].startswith("https://"))
    
    def test_priority_patterns_content(self):
        """Test that priority patterns are properly configured"""
        expected_patterns = [
            "ลายนกไส่บินข้ามทุ่ง", "ลายมโหรีอีสาน", "ลายแมลงภู่ตอมดอกไม้",
            "ลายเต้ยโขง", "ลายเซิ้งบั้งไฟ", "ลายลำเพลิน",
            "ลายสะบ้องทุ่ง", "ลายโคมประทีป", "ลายไหว้ครู"
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, self.downloader.priority_patterns)
    
    def test_get_sample_videos(self):
        """Test sample videos retrieval"""
        sample_videos = self.downloader.get_sample_videos()
        
        self.assertIsInstance(sample_videos, list)
        self.assertEqual(len(sample_videos), 3)
        
        for video in sample_videos:
            self.assertIn("url", video)
            self.assertIn("title", video)
            self.assertIn("pattern", video)
            self.assertTrue(video["url"].startswith("https://www.youtube.com/"))
            self.assertTrue(len(video["title"]) > 0)
            self.assertTrue(len(video["pattern"]) > 0)
    
    @patch('subprocess.run')
    def test_download_video_success(self, mock_run):
        """Test successful video download"""
        # Mock successful subprocess call
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        
        url = "https://www.youtube.com/watch?v=test123"
        result = self.downloader.download_video(url, "test_video")
        
        self.assertTrue(result)
        mock_run.assert_called_once()
        
        # Check that command contains expected arguments
        call_args = mock_run.call_args[0][0]
        self.assertIn('yt-dlp', call_args)
        self.assertIn('--format', call_args)
        self.assertIn('--extract-audio', call_args)
        self.assertIn('--audio-format', call_args)
        self.assertIn('wav', call_args)
    
    @patch('subprocess.run')
    def test_download_video_failure(self, mock_run):
        """Test failed video download"""
        # Mock failed subprocess call
        mock_run.return_value = MagicMock(returncode=1, stderr="Download failed")
        
        url = "https://www.youtube.com/watch?v=test123"
        result = self.downloader.download_video(url, "test_video")
        
        self.assertFalse(result)
        mock_run.assert_called_once()
    
    def test_download_video_exception(self):
        """Test video download with exception"""
        # Test with invalid URL
        result = self.downloader.download_video("invalid_url")
        self.assertFalse(result)
    
    @patch.object(YouTubeDownloader, 'download_video')
    def test_download_batch(self, mock_download):
        """Test batch download functionality"""
        # Mock individual downloads
        mock_download.return_value = True
        
        urls = ["url1", "url2", "url3"]
        pattern_name = "test_pattern"
        
        results = self.downloader.download_batch(urls, pattern_name)
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results.values()))  # All should be True
        self.assertEqual(mock_download.call_count, 3)
    
    def test_create_dataset_manifest_empty(self):
        """Test manifest creation with empty directory"""
        manifest = self.downloader.create_dataset_manifest()
        
        self.assertEqual(manifest["total_files"], 0)
        self.assertEqual(len(manifest["files"]), 0)
        self.assertEqual(len(manifest["patterns"]), 0)
        
        # Check that manifest file was created
        manifest_path = self.output_dir.parent / "dataset_manifest.json"
        self.assertTrue(manifest_path.exists())
        
        # Load and verify saved manifest
        with open(manifest_path, 'r') as f:
            saved_manifest = json.load(f)
        
        self.assertEqual(saved_manifest["total_files"], 0)
    
    def test_create_dataset_manifest_with_files(self):
        """Test manifest creation with audio files"""
        # Create mock audio files in directory structure
        pattern_dir = self.output_dir / "ลายนกไส่บินข้ามทุ่ง"
        pattern_dir.mkdir(exist_ok=True)
        
        # Create mock WAV files
        (pattern_dir / "audio1.wav").write_text("fake audio data 1")
        (pattern_dir / "audio2.wav").write_text("fake audio data 2")
        
        # Create file in root directory
        (self.output_dir / "audio3.wav").write_text("fake audio data 3")
        
        manifest = self.downloader.create_dataset_manifest()
        
        self.assertEqual(manifest["total_files"], 3)
        self.assertEqual(len(manifest["files"]), 3)
        self.assertIn("ลายนกไส่บินข้ามทุ่ง", manifest["patterns"])
        
        # Check file details
        for file_info in manifest["files"]:
            self.assertIn("filename", file_info)
            self.assertIn("path", file_info)
            self.assertIn("pattern", file_info)
            self.assertIn("size_bytes", file_info)
            self.assertIn("duration_seconds", file_info)
            
            self.assertTrue(file_info["size_bytes"] > 0)
    
    def test_yt_dlp_config(self):
        """Test yt-dlp configuration"""
        config = self.downloader.yt_dlp_config
        
        self.assertIn("format", config)
        self.assertIn("postprocessors", config)
        self.assertIn("outtmpl", config)
        self.assertIn("quiet", config)
        self.assertIn("no_warnings", config)
        
        # Check postprocessor configuration
        postprocessor = config["postprocessors"][0]
        self.assertEqual(postprocessor["key"], "FFmpegExtractAudio")
        self.assertEqual(postprocessor["preferredcodec"], "wav")
        self.assertEqual(postprocessor["preferredquality"], "192")


class TestYouTubeDownloaderEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.downloader = YouTubeDownloader(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_invalid_output_directory(self):
        """Test with invalid output directory"""
        # Use a path within test directory instead of system root
        invalid_path = os.path.join(self.test_dir, "invalid", "path", "that", "does", "not", "exist")
        downloader = YouTubeDownloader(output_dir=invalid_path)
        
        # Should create the directory
        self.assertTrue(Path(invalid_path).exists())
    
    def test_empty_url_list_batch_download(self):
        """Test batch download with empty URL list"""
        results = self.downloader.download_batch([], "test_pattern")
        
        self.assertEqual(len(results), 0)
    
    def test_special_characters_in_pattern_name(self):
        """Test pattern names with special characters"""
        # Create directory with Thai characters and spaces
        pattern_name = "ลาย นก ไส่ บิน ข้าม ทุ่ง"
        
        # Create mock file
        pattern_dir = Path(self.test_dir) / pattern_name
        pattern_dir.mkdir(exist_ok=True)
        (pattern_dir / "test.wav").write_text("test data")
        
        # Should handle special characters properly
        manifest = self.downloader.create_dataset_manifest()
        
        self.assertEqual(manifest["total_files"], 1)
        self.assertIn(pattern_name, manifest["patterns"])


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    @patch.object(YouTubeDownloader, 'download_video')
    def test_complete_download_workflow(self, mock_download):
        """Test complete download workflow"""
        mock_download.return_value = True
        
        downloader = YouTubeDownloader(output_dir=self.test_dir)
        
        # Get sample videos
        sample_videos = downloader.get_sample_videos()
        
        # Download videos
        for video in sample_videos:
            downloader.download_video(video['url'], video['pattern'])
        
        # Create manifest
        manifest = downloader.create_dataset_manifest()
        
        # Verify workflow completed
        self.assertEqual(mock_download.call_count, 3)
        self.assertTrue(mock_download.called)


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestYouTubeDownloader))
    suite.addTests(loader.loadTestsFromTestCase(TestYouTubeDownloaderEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)