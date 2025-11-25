#!/usr/bin/env python3
"""
Enhanced YouTube Video Downloader for Phin AI Dataset
Fixes authentication issues and adds better error handling
"""

import os
import json
import subprocess
import logging
import time
import random
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedYouTubeDownloader:
    """Enhanced YouTube downloader with better error handling and authentication support"""
    
    def __init__(self, output_dir: str = "audio_sources/raw_audio", use_cookies: bool = True):
        """Initialize enhanced YouTube downloader"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cookies = use_cookies
        
        # YouTube channels and video sources
        self.phin_channels = {
            "ดุลย์เพลงพิณ": "https://www.youtube.com/@dulyplengphin",
            "M MUSIC GROUP": "https://www.youtube.com/@mmusicgroup5547",
            "สตีฟ ฐิติวัสส์": "https://www.youtube.com/@steve_thitiwat",
            "พิณอีสาน บ้านเฮา": "https://www.youtube.com/@phinisanbanhao",
            "เพลงพิณดนตรีอีสาน": "https://www.youtube.com/@phinmusicisan"
        }
        
        # Priority Phin patterns to collect
        self.priority_patterns = [
            "ลายนกไส่บินข้ามทุ่ง",
            "ลายมโหรีอีสาน", 
            "ลายแมลงภู่ตอมดอกไม้",
            "ลายเต้ยโขง",
            "ลายเซิ้งบั้งไฟ",
            "ลายลำเพลิน",
            "ลายสะบ้องทุ่ง",
            "ลายโคมประทีป",
            "ลายไหว้ครู"
        ]
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # User agent rotation to avoid bot detection
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
    
    def get_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        try:
            if "youtube.com/watch" in url:
                parsed_url = urlparse(url)
                video_id = parse_qs(parsed_url.query).get('v', [None])[0]
                return video_id
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            return None
        except Exception as e:
            logger.error(f"Failed to extract video ID from {url}: {str(e)}")
            return None
    
    def is_valid_youtube_url(self, url: str) -> bool:
        """Validate YouTube URL format"""
        valid_patterns = [
            r'youtube\.com/watch\?v=[\w-]+',
            r'youtu\.be/[\w-]+',
            r'youtube\.com/embed/[\w-]+'
        ]
        
        import re
        for pattern in valid_patterns:
            if re.search(pattern, url):
                return True
        return False
    
    def build_download_command(self, url: str, output_filename: Optional[str] = None) -> List[str]:
        """Build yt-dlp command with authentication bypass"""
        cmd = ['yt-dlp']
        
        # Basic download options
        cmd.extend(['--format', 'bestaudio/best'])
        cmd.extend(['--extract-audio'])
        cmd.extend(['--audio-format', 'wav'])
        cmd.extend(['--audio-quality', '192K'])
        
        # Authentication bypass options
        cmd.extend(['--no-check-certificate'])
        cmd.extend(['--ignore-errors'])
        cmd.extend(['--no-playlist'])
        cmd.extend(['--no-warnings'])
        
        # Random user agent to avoid bot detection
        user_agent = random.choice(self.user_agents)
        cmd.extend(['--user-agent', user_agent])
        
        # Add cookies if available (helps with age-restricted content)
        if self.use_cookies:
            # Try common cookie locations
            cookie_paths = [
                os.path.expanduser("~/.cookies.txt"),
                os.path.expanduser("~/.config/yt-dlp/cookies.txt"),
                "cookies.txt"
            ]
            
            for cookie_path in cookie_paths:
                if os.path.exists(cookie_path):
                    cmd.extend(['--cookies', cookie_path])
                    logger.info(f"Using cookies from: {cookie_path}")
                    break
        
        # Add referer to appear more like a browser
        cmd.extend(['--referer', 'https://www.youtube.com/'])
        
        # Output template
        if output_filename:
            output_path = self.output_dir / f"{output_filename}.wav"
        else:
            output_path = self.output_dir / '%(title)s.%(ext)s'
        
        cmd.extend(['--output', str(output_path)])
        
        # Add URL
        cmd.append(url)
        
        return cmd
    
    def download_video_with_retry(self, url: str, output_filename: Optional[str] = None) -> bool:
        """Download video with retry logic and multiple strategies"""
        if not self.is_valid_youtube_url(url):
            logger.error(f"Invalid YouTube URL: {url}")
            return False
        
        video_id = self.get_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from: {url}")
            return False
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading video (attempt {attempt + 1}/{self.max_retries}): {url}")
                
                # Build and execute command
                cmd = self.build_download_command(url, output_filename)
                
                logger.info(f"Executing: {' '.join(cmd[:10])}...")  # Log first 10 args
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"Successfully downloaded: {url}")
                    return True
                else:
                    logger.warning(f"Download failed (attempt {attempt + 1}): {result.stderr}")
                    
                    # Try alternative approach for age-restricted content
                    if "age-restricted" in result.stderr.lower():
                        logger.info("Trying alternative approach for age-restricted content...")
                        return self.download_age_restricted_video(url, output_filename)
                    
                    # Wait before retry
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (attempt + 1)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        
            except subprocess.TimeoutExpired:
                logger.error(f"Download timeout for: {url}")
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
        
        logger.error(f"Failed to download after {self.max_retries} attempts: {url}")
        return False
    
    def download_age_restricted_video(self, url: str, output_filename: Optional[str] = None) -> bool:
        """Download age-restricted videos using alternative methods"""
        try:
            logger.info(f"Attempting age-restricted video download: {url}")
            
            # Use alternative downloader options
            cmd = ['yt-dlp']
            cmd.extend(['--format', 'worstaudio'])  # Lower quality often works for restricted content
            cmd.extend(['--extract-audio'])
            cmd.extend(['--audio-format', 'wav'])
            cmd.extend(['--audio-quality', '128K'])  # Lower quality for restricted content
            cmd.extend(['--no-check-certificate'])
            cmd.extend(['--ignore-errors'])
            cmd.extend(['--no-playlist'])
            
            # Add output template
            if output_filename:
                output_path = self.output_dir / f"{output_filename}.wav"
            else:
                output_path = self.output_dir / '%(title)s.%(ext)s'
            
            cmd.extend(['--output', str(output_path)])
            cmd.append(url)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded age-restricted video: {url}")
                return True
            else:
                logger.error(f"Failed to download age-restricted video: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading age-restricted video {url}: {str(e)}")
            return False
    
    def download_video(self, url: str, output_filename: Optional[str] = None) -> bool:
        """Download a single YouTube video (main method)"""
        return self.download_video_with_retry(url, output_filename)
    
    def download_batch(self, urls: List[str], pattern_name: str = "unknown") -> Dict[str, bool]:
        """Download multiple videos with better error handling"""
        results = {}
        
        # Sanitize pattern name for filesystem
        safe_pattern_name = "".join(c for c in pattern_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        # Create pattern-specific directory
        pattern_dir = self.output_dir / safe_pattern_name
        pattern_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting batch download for pattern: {pattern_name}")
        logger.info(f"Output directory: {pattern_dir}")
        logger.info(f"URLs to download: {len(urls)}")
        
        for i, url in enumerate(urls):
            filename = f"{safe_pattern_name}_{i+1:03d}"
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
            
            success = self.download_video(url, filename)
            results[url] = success
            
            # Add delay between downloads to avoid rate limiting
            if i < len(urls) - 1:
                delay = random.uniform(2, 5)
                logger.info(f"Waiting {delay:.1f} seconds before next download...")
                time.sleep(delay)
        
        logger.info(f"Batch download complete. Success: {sum(results.values())}/{len(urls)}")
        return results
    
    def get_sample_videos(self) -> List[Dict[str, str]]:
        """Return sample videos for initial testing with working URLs"""
        return [
            {
                "url": "https://www.youtube.com/watch?v=ksZ3DWA9mPE",
                "title": "ลายนกไส่บินข้ามทุ่ง - ดุลย์เพลงพิณ",
                "pattern": "ลายนกไส่บินข้ามทุ่ง"
            },
            {
                "url": "https://www.youtube.com/watch?v=ZRK75tNHqKc", 
                "title": "ลายมโหรีอีสาน - M MUSIC GROUP",
                "pattern": "ลายมโหรีอีสาน"
            },
            {
                "url": "https://www.youtube.com/watch?v=aQZEN3y8zWo",
                "title": "ลายแมลงภู่ตอมดอกไม้ - สตีฟ ฐิติวัสส์",
                "pattern": "ลายแมลงภู่ตอมดอกไม้"
            }
        ]
    
    def create_dataset_manifest(self) -> Dict:
        """Create manifest of downloaded files with enhanced metadata"""
        manifest = {
            "total_files": 0,
            "patterns": {},
            "files": [],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "downloader_version": "1.1.0"
        }
        
        # Scan output directory
        for audio_file in self.output_dir.glob("**/*.wav"):
            relative_path = audio_file.relative_to(self.output_dir)
            
            # Extract pattern from directory structure
            if relative_path.parent != Path("."):
                pattern = relative_path.parent.name
            else:
                pattern = "unknown"
            
            file_info = {
                "filename": audio_file.name,
                "path": str(relative_path),
                "pattern": pattern,
                "size_bytes": audio_file.stat().st_size,
                "duration_seconds": None,  # Will be filled by audio analysis
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            manifest["files"].append(file_info)
            
            if pattern not in manifest["patterns"]:
                manifest["patterns"][pattern] = []
            manifest["patterns"][pattern].append(str(relative_path))
            
            manifest["total_files"] += 1
        
        # Save manifest
        manifest_path = self.output_dir.parent / "dataset_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset manifest created: {manifest_path}")
        logger.info(f"Total files: {manifest['total_files']}")
        logger.info(f"Patterns found: {len(manifest['patterns'])}")
        
        return manifest
    
    def get_download_statistics(self) -> Dict:
        """Get statistics about downloaded files"""
        manifest = self.create_dataset_manifest()
        
        total_size = sum(file_info["size_bytes"] for file_info in manifest["files"])
        
        stats = {
            "total_files": manifest["total_files"],
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "patterns_count": len(manifest["patterns"]),
            "patterns": list(manifest["patterns"].keys()),
            "average_file_size_bytes": total_size / manifest["total_files"] if manifest["total_files"] > 0 else 0
        }
        
        return stats


def main():
    """Main function for testing the enhanced downloader"""
    print("🎵 Enhanced Phin AI Dataset - YouTube Downloader")
    print("=" * 60)
    
    # Create enhanced downloader
    downloader = EnhancedYouTubeDownloader(use_cookies=True)
    
    print(f"Output directory: {downloader.output_dir}")
    print(f"Using cookies: {downloader.use_cookies}")
    print(f"Max retries: {downloader.max_retries}")
    print()
    
    # Get sample videos
    sample_videos = downloader.get_sample_videos()
    print(f"Sample videos to download: {len(sample_videos)}")
    print()
    
    # Download sample videos with retry logic
    successful_downloads = 0
    for i, video in enumerate(sample_videos, 1):
        print(f"[{i}/{len(sample_videos)}] Downloading: {video['title']}")
        
        success = downloader.download_video(video['url'], video['pattern'])
        status = "✅ Success" if success else "❌ Failed"
        print(f"Status: {status}")
        
        if success:
            successful_downloads += 1
        
        # Add delay between downloads
        if i < len(sample_videos):
            delay = 3
            print(f"Waiting {delay} seconds before next download...")
            time.sleep(delay)
        print()
    
    # Create dataset manifest
    manifest = downloader.create_dataset_manifest()
    print(f"Dataset manifest created with {manifest['total_files']} files")
    
    # Get statistics
    stats = downloader.get_download_statistics()
    print(f"\n📊 Download Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print(f"Patterns: {stats['patterns_count']}")
    print(f"Success rate: {successful_downloads}/{len(sample_videos)}")
    
    print(f"\n🎯 Next steps:")
    print("1. Check downloaded audio files in audio_sources/raw_audio/")
    print("2. Run audio preprocessing pipeline")
    print("3. Test with Basic Pitch transcription")
    print("4. If downloads failed, check:")
    print("   - yt-dlp is installed: pip install yt-dlp")
    print("   - FFmpeg is installed: ffmpeg -version")
    print("   - Try creating cookies.txt file for authentication")


if __name__ == "__main__":
    main()