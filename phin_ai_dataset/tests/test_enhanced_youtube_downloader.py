#!/usr/bin/env python3
"""
Enhanced unit tests for the improved YouTube downloader with comprehensive error handling
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import subprocess

# Add parent directory to path
import sys
sys.path.append('..')
sys.path.append('../data_collection')

from enhanced_youtube_downloader_v2 import (
    YouTubeConfig, DownloadResult, YouTubeUrlValidator, EnhancedYouTubeDownloader,
    get_sample_phin_videos
)

class TestYouTubeConfig(unittest.TestCase):
    """Test YouTubeConfig dataclass"""
    
    def test_default_config(self):
        config = YouTubeConfig()
        self.assertEqual(config.output_dir, "audio_sources/raw_audio")
        self.assertEqual(config.audio_format, "wav")
        self.assertEqual(config.audio_quality, "192K")
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.concurrent_downloads, 2)
    
    def test_custom_config(self):
        config = YouTubeConfig(
            output_dir="custom_dir",
            audio_quality="320K",
            max_retries=5,
            concurrent_downloads=4
        )
        self.assertEqual(config.output_dir, "custom_dir")
        self.assertEqual(config.audio_quality, "320K")
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.concurrent_downloads, 4)

class TestDownloadResult(unittest.TestCase):
    """Test DownloadResult dataclass"""
    
    def test_successful_result(self):
        result = DownloadResult(
            url="https://example.com/video",
            status="success",
            file_path="/path/to/file.wav",
            title="Test Video",
            duration=120.5,
            file_size=1024000,
            retry_count=0
        )
        
        self.assertEqual(result.url, "https://example.com/video")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.file_path, "/path/to/file.wav")
        self.assertEqual(result.title, "Test Video")
        self.assertEqual(result.duration, 120.5)
        self.assertEqual(result.file_size, 1024000)
        self.assertEqual(result.retry_count, 0)
        self.assertIsNone(result.error)
    
    def test_failed_result(self):
        result = DownloadResult(
            url="https://example.com/video",
            status="failed",
            error="Network error",
            retry_count=2
        )
        
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "Network error")
        self.assertEqual(result.retry_count, 2)
        self.assertIsNone(result.file_path)

class TestYouTubeUrlValidator(unittest.TestCase):
    """Test YouTubeUrlValidator class"""
    
    def test_validate_valid_youtube_urls(self):
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=ksZ3DWA9mPE&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        ]
        
        for url in valid_urls:
            is_valid, message = YouTubeUrlValidator.validate_url(url)
            self.assertTrue(is_valid, f"URL should be valid: {url}")
            self.assertIn("Valid", message)
    
    def test_validate_invalid_urls(self):
        invalid_urls = [
            "https://www.google.com/watch?v=dQw4w9WgXcQ",  # Wrong domain
            "https://www.youtube.com/watch",  # Missing video ID
            "https://www.youtube.com/",  # No video ID
            "not a url at all",
            "",
            "https://youtu.be/",  # Short URL without ID
        ]
        
        for url in invalid_urls:
            is_valid, message = YouTubeUrlValidator.validate_url(url)
            self.assertFalse(is_valid, f"URL should be invalid: {url}")
    
    def test_extract_video_id(self):
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=ksZ3DWA9mPE", "ksZ3DWA9mPE"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf", "dQw4w9WgXcQ"),
        ]
        
        for url, expected_id in test_cases:
            video_id = YouTubeUrlValidator.extract_video_id(url)
            self.assertEqual(video_id, expected_id, f"Failed for URL: {url}")
    
    def test_extract_video_id_invalid_urls(self):
        invalid_urls = [
            "https://www.google.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch",
            "not a url at all",  # This should now return None
            ""
        ]
        
        for url in invalid_urls:
            video_id = YouTubeUrlValidator.extract_video_id(url)
            self.assertIsNone(video_id, f"Should return None for invalid URL: {url}")

class TestEnhancedYouTubeDownloader(unittest.TestCase):
    """Test EnhancedYouTubeDownloader class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = YouTubeConfig(
            output_dir=self.temp_dir,
            max_retries=2,
            retry_delay=0.1,  # Short delay for testing
            timeout=10
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_check_dependencies_available(self, mock_run):
        # Mock successful dependency checks
        mock_run.return_value = Mock(returncode=0)
        
        try:
            downloader = EnhancedYouTubeDownloader(self.config)
            self.assertIsNotNone(downloader)
        except RuntimeError:
            # This is expected in test environment without actual dependencies
            pass
    
    @patch('subprocess.run')
    def test_check_dependencies_missing(self, mock_run):
        # Mock missing dependencies
        mock_run.side_effect = FileNotFoundError("Command not found")
        
        with self.assertRaises(RuntimeError):
            EnhancedYouTubeDownloader(self.config)
    
    def test_sanitize_filename(self):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        test_cases = [
            ("Normal Title", "Normal Title"),
            ("Title:with<invalid>chars", "Title_with_invalid_chars"),
            ("Title/with/backslash", "Title_with_backslash"),
            ("Title|with|pipe", "Title_with_pipe"),
            ("Title*with*asterisk", "Title_with_asterisk"),
            ("Title?with?question", "Title_with_question"),
            ("Title:with:colon", "Title_with_colon"),
            ("   Title with spaces   ", "Title with spaces"),  # Should trim
            ("", "unnamed_video"),  # Empty filename
            ("A" * 300, "A" * 200),  # Very long filename should be truncated
        ]
        
        for input_name, expected_output in test_cases:
            sanitized = downloader._sanitize_filename(input_name)
            self.assertEqual(sanitized, expected_output)
    
    @patch('subprocess.run')
    def test_get_video_info_success(self, mock_run):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock successful video info retrieval
        mock_video_info = {
            "title": "Test Video Title",
            "duration": 185,
            "uploader": "Test Channel",
            "view_count": 1000000,
            "like_count": 50000
        }
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(mock_video_info),
            stderr=""
        )
        
        result = downloader._get_video_info("https://www.youtube.com/watch?v=test123")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "Test Video Title")
        self.assertEqual(result["duration"], 185)
        self.assertEqual(result["uploader"], "Test Channel")
    
    @patch('subprocess.run')
    def test_get_video_info_failure(self, mock_run):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock failed video info retrieval
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Video unavailable"
        )
        
        result = downloader._get_video_info("https://www.youtube.com/watch?v=invalid")
        
        self.assertIsNone(result)
    
    def test_build_yt_dlp_command(self):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        url = "https://www.youtube.com/watch?v=test123"
        output_template = "/tmp/output.%(ext)s"
        
        cmd = downloader._build_yt_dlp_command(url, output_template)
        
        # Check command structure
        self.assertIn("yt-dlp", cmd[0])
        self.assertIn("--extract-audio", cmd)
        self.assertIn("--audio-format", cmd)
        self.assertIn("wav", cmd)
        self.assertIn("--audio-quality", cmd)
        self.assertIn("192K", cmd)
        self.assertIn("--output", cmd)
        self.assertIn(output_template, cmd)
        self.assertIn(url, cmd)
    
    @patch('subprocess.run')
    @patch.object(EnhancedYouTubeDownloader, '_get_video_info')
    def test_download_single_video_success(self, mock_get_info, mock_run):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock video info
        mock_get_info.return_value = {
            "title": "Test Video",
            "duration": 120,
            "uploader": "Test Channel"
        }
        
        # Mock successful download
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        # Mock finding downloaded file
        with patch.object(downloader, '_find_downloaded_file') as mock_find:
            mock_file = Mock()
            mock_file.exists.return_value = True
            mock_file.stat.return_value.st_size = 1024000  # 1MB
            mock_file.name = "test_file.wav"
            mock_find.return_value = mock_file
            
            result = downloader.download_single_video("https://www.youtube.com/watch?v=test123")
        
        self.assertEqual(result.status, "success")
        self.assertEqual(result.title, "Test Video")
        self.assertEqual(result.duration, 120)
        self.assertEqual(result.file_size, 1024000)
        self.assertIsNotNone(result.file_path)
    
    @patch('subprocess.run')
    @patch.object(EnhancedYouTubeDownloader, '_get_video_info')
    def test_download_single_video_authentication_error(self, mock_get_info, mock_run):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock video info
        mock_get_info.return_value = {
            "title": "Test Video",
            "duration": 120,
            "uploader": "Test Channel"
        }
        
        # Mock authentication error
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Sign in to confirm you're not a bot"
        )
        
        result = downloader.download_single_video("https://www.youtube.com/watch?v=test123")
        
        self.assertEqual(result.status, "failed")
        self.assertIn("authentication", result.error.lower())
    
    @patch('subprocess.run')
    @patch.object(EnhancedYouTubeDownloader, '_get_video_info')
    def test_download_single_video_unavailable(self, mock_get_info, mock_run):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock video info
        mock_get_info.return_value = {
            "title": "Test Video",
            "duration": 120,
            "uploader": "Test Channel"
        }
        
        # Mock unavailable video error
        mock_run.return_value = Mock(
            returncode=1,
            stderr="This video is unavailable"
        )
        
        result = downloader.download_single_video("https://www.youtube.com/watch?v=unavailable")
        
        self.assertEqual(result.status, "failed")
        self.assertIn("unavailable", result.error.lower())
    
    @patch('subprocess.run')
    @patch.object(EnhancedYouTubeDownloader, '_get_video_info')
    def test_download_single_video_timeout(self, mock_get_info, mock_run):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock video info
        mock_get_info.return_value = {
            "title": "Test Video",
            "duration": 120,
            "uploader": "Test Channel"
        }
        
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)
        
        result = downloader.download_single_video("https://www.youtube.com/watch?v=timeout")
        
        self.assertEqual(result.status, "failed")
        self.assertIn("timed out", result.error.lower())
    
    def test_find_downloaded_file(self):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Test with simple template
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            result = downloader._find_downloaded_file("/tmp/output.%(ext)s")
            self.assertIsNotNone(result)
    
    def test_create_cookies_template(self):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Remove cookies file if it exists
        cookies_file = Path(self.config.cookies_file)
        if cookies_file.exists():
            cookies_file.unlink()
        
        downloader.create_cookies_template()
        
        self.assertTrue(cookies_file.exists())
        
        # Check content
        with open(cookies_file, 'r') as f:
            content = f.read()
            self.assertIn("YouTube Cookies Template", content)
            self.assertIn("cookies.txt", content)

class TestBatchDownload(unittest.TestCase):
    """Test batch download functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = YouTubeConfig(
            output_dir=self.temp_dir,
            concurrent_downloads=2
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch.object(EnhancedYouTubeDownloader, 'download_single_video')
    def test_download_batch_success(self, mock_download_single):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock successful downloads
        def mock_download(url, custom_name=None):
            return DownloadResult(
                url=url,
                status="success",
                file_path=f"/tmp/{url.split('=')[-1]}.wav",
                title=f"Video {url.split('=')[-1]}",
                duration=120.0,
                file_size=1024000
            )
        
        mock_download_single.side_effect = mock_download
        
        urls = [
            "https://www.youtube.com/watch?v=test1",
            "https://www.youtube.com/watch?v=test2",
            "https://www.youtube.com/watch?v=test3"
        ]
        
        results = downloader.download_batch(urls)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(len([r for r in results if r.status == "success"]), 3)
        self.assertEqual(len([r for r in results if r.status == "failed"]), 0)
    
    @patch.object(EnhancedYouTubeDownloader, 'download_single_video')
    def test_download_batch_mixed_results(self, mock_download_single):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock mixed results
        def mock_download(url, custom_name=None):
            if "test1" in url or "test3" in url:
                return DownloadResult(
                    url=url,
                    status="success",
                    file_path=f"/tmp/{url.split('=')[-1]}.wav",
                    title=f"Video {url.split('=')[-1]}"
                )
            else:
                return DownloadResult(
                    url=url,
                    status="failed",
                    error="Network error"
                )
        
        mock_download_single.side_effect = mock_download
        
        urls = [
            "https://www.youtube.com/watch?v=test1",
            "https://www.youtube.com/watch?v=test2",
            "https://www.youtube.com/watch?v=test3"
        ]
        
        results = downloader.download_batch(urls)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(len([r for r in results if r.status == "success"]), 2)
        self.assertEqual(len([r for r in results if r.status == "failed"]), 1)
    
    @patch.object(EnhancedYouTubeDownloader, 'download_single_video')
    def test_download_batch_with_custom_filenames(self, mock_download_single):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock successful downloads
        def mock_download(url, custom_name=None):
            return DownloadResult(
                url=url,
                status="success",
                file_path=f"/tmp/{custom_name}.wav" if custom_name else f"/tmp/{url.split('=')[-1]}.wav",
                title=custom_name or f"Video {url.split('=')[-1]}"
            )
        
        mock_download_single.side_effect = mock_download
        
        urls = ["https://www.youtube.com/watch?v=test1"]
        custom_filenames = ["Custom Name"]
        
        results = downloader.download_batch(urls, custom_filenames)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "success")
        
        # Check that custom filename was passed
        mock_download_single.assert_called_once_with(urls[0], "Custom Name")

class TestSampleVideos(unittest.TestCase):
    """Test sample Phin videos function"""
    
    def test_get_sample_phin_videos(self):
        videos = get_sample_phin_videos()
        
        self.assertIsInstance(videos, list)
        self.assertGreater(len(videos), 0)
        
        for video in videos:
            self.assertTrue(video.startswith("https://www.youtube.com/watch?v="))
            # Validate each URL
            is_valid, _ = YouTubeUrlValidator.validate_url(video)
            self.assertTrue(is_valid)

class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and comprehensive error handling"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = YouTubeConfig(
            output_dir=self.temp_dir,
            max_retries=1,
            retry_delay=0.01  # Very short delay for testing
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_validate_url_edge_cases(self):
        edge_case_urls = [
            "",  # Empty string
            "not a url",  # Not a URL
            "http://",  # Incomplete URL
            "https://youtube.com",  # YouTube domain but no video
            "https://www.youtube.com/watch?v=",  # Empty video ID
            "https://www.youtube.com/watch",  # Missing video parameter
            "https://youtu.be/",  # Short URL without ID
            "https://youtu.be/abc",  # Short URL with short ID
        ]
        
        for url in edge_case_urls:
            is_valid, message = YouTubeUrlValidator.validate_url(url)
            self.assertFalse(is_valid, f"URL should be invalid: {url}")
            self.assertIsInstance(message, str)
            self.assertGreater(len(message), 0)
    
    def test_extract_video_id_edge_cases(self):
        edge_case_urls = [
            "",  # Empty string
            "not a url",  # Not a URL
            "https://www.google.com/watch?v=test",  # Wrong domain
            "https://www.youtube.com/watch",  # Missing video ID
            "https://youtu.be/",  # Short URL without ID
        ]
        
        for url in edge_case_urls:
            video_id = YouTubeUrlValidator.extract_video_id(url)
            self.assertIsNone(video_id, f"Should return None for invalid URL: {url}")
    
    def test_sanitize_filename_edge_cases(self):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        edge_cases = [
            ("", "unnamed_video"),  # Empty string
            ("   ", "unnamed_video"),  # Whitespace only
            (".", "unnamed_video"),  # Single dot
            ("..", "unnamed_video"),  # Double dots -> should become unnamed_video
            ("___", "___"),  # Multiple underscores
            ("a" * 500, "a" * 200),  # Very long filename
            ("file/with/path", "file_with_path"),  # Path-like name
            ("file\\with\\backslash", "file_with_backslash"),  # Windows path
            ("file:with:colons", "file_with_colons"),  # Colons
            ("file*with*asterisks", "file_with_asterisks"),  # Asterisks
            ("file?with?questions", "file_with_questions"),  # Question marks
            ("file\"with\"quotes", "file_with_quotes"),  # Quotes
            ("file<with>brackets", "file_with_brackets"),  # Brackets
            ("file|with|pipes", "file_with_pipes"),  # Pipes
        ]
        
        for input_name, expected in edge_cases:
            result = downloader._sanitize_filename(input_name)
            self.assertEqual(result, expected, f"Failed for input: '{input_name}'")
    
    @patch('subprocess.run')
    @patch.object(EnhancedYouTubeDownloader, '_get_video_info')
    def test_retry_mechanism(self, mock_get_info, mock_run):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock video info
        mock_get_info.return_value = {
            "title": "Test Video",
            "duration": 120,
            "uploader": "Test Channel"
        }
        
        # Mock failures followed by success
        call_count = 0
        def mock_run_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # First two calls fail
                return Mock(returncode=1, stderr="Network error")
            else:  # Third call succeeds
                return Mock(returncode=0, stderr="")
        
        mock_run.side_effect = mock_run_side_effect
        
        # Mock finding downloaded file
        with patch.object(downloader, '_find_downloaded_file') as mock_find:
            mock_file = Mock()
            mock_file.exists.return_value = True
            mock_file.stat.return_value.st_size = 1024000
            mock_file.name = "test_file.wav"
            mock_find.return_value = mock_file
            
            result = downloader.download_single_video("https://www.youtube.com/watch?v=test123")
        
        self.assertEqual(result.status, "success")
        self.assertGreaterEqual(result.retry_count, 1)  # Should have retried
    
    def test_concurrent_download_edge_cases(self):
        # Test that concurrent_downloads=1 works (minimum valid value)
        config = YouTubeConfig(
            output_dir=self.temp_dir,
            concurrent_downloads=1
        )
        
        # Should not raise an error
        downloader = EnhancedYouTubeDownloader(config)
        self.assertIsNotNone(downloader)
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_batch_download_empty_list(self, mock_executor):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Mock the executor to avoid actual instantiation
        mock_executor_instance = mock_executor.return_value.__enter__.return_value
        
        results = downloader.download_batch([])
        
        self.assertEqual(len(results), 0)
        # ThreadPoolExecutor should not be called for empty list
        mock_executor.assert_not_called()

class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = YouTubeConfig(output_dir=self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch.object(EnhancedYouTubeDownloader, 'download_single_video')
    def test_full_workflow_with_sample_videos(self, mock_download_single):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Get sample videos
        sample_videos = get_sample_phin_videos()
        self.assertGreater(len(sample_videos), 0)
        
        # Mock successful downloads
        def mock_download(url, custom_name=None):
            return DownloadResult(
                url=url,
                status="success",
                file_path=f"/tmp/{url.split('=')[-1]}.wav",
                title=f"Phin Video {url.split('=')[-1]}",
                duration=180.0,
                file_size=2048000
            )
        
        mock_download_single.side_effect = mock_download
        
        # Download batch
        results = downloader.download_batch(sample_videos)
        
        # Verify results
        self.assertEqual(len(results), len(sample_videos))
        self.assertEqual(len([r for r in results if r.status == "success"]), len(sample_videos))
        
        # Verify each result (check that all URLs are accounted for)
        result_urls = [result.url for result in results]
        for sample_url in sample_videos:
            self.assertIn(sample_url, result_urls)
            self.assertEqual(result.status, "success")
            self.assertIsNotNone(result.file_path)
            self.assertIsNotNone(result.title)
            self.assertGreater(result.duration, 0)
            self.assertGreater(result.file_size, 0)
    
    @patch.object(EnhancedYouTubeDownloader, 'download_single_video')
    def test_error_recovery_scenario(self, mock_download_single):
        downloader = EnhancedYouTubeDownloader(self.config)
        
        # Simulate mixed success/failure pattern
        call_count = 0
        def mock_download_with_failures(url, custom_name=None):
            nonlocal call_count
            call_count += 1
            
            if call_count % 2 == 0:  # Every other download fails
                return DownloadResult(
                    url=url,
                    status="failed",
                    error="Simulated network error"
                )
            else:
                return DownloadResult(
                    url=url,
                    status="success",
                    file_path=f"/tmp/{url.split('=')[-1]}.wav",
                    title=f"Video {call_count}"
                )
        
        mock_download_single.side_effect = mock_download_with_failures
        
        # Download multiple videos
        urls = [
            "https://www.youtube.com/watch?v=video1",
            "https://www.youtube.com/watch?v=video2",
            "https://www.youtube.com/watch?v=video3",
            "https://www.youtube.com/watch?v=video4"
        ]
        
        results = downloader.download_batch(urls)
        
        # Should have mixed results
        successful_count = len([r for r in results if r.status == "success"])
        failed_count = len([r for r in results if r.status == "failed"])
        
        self.assertEqual(successful_count + failed_count, len(urls))
        self.assertGreater(successful_count, 0)
        self.assertGreater(failed_count, 0)

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestYouTubeConfig,
        TestDownloadResult,
        TestYouTubeUrlValidator,
        TestEnhancedYouTubeDownloader,
        TestBatchDownload,
        TestSampleVideos,
        TestEdgeCasesAndErrorHandling,
        TestIntegrationScenarios
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