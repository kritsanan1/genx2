#!/usr/bin/env python3
"""
YouTube Video Downloader for Phin AI Dataset
Downloads high-quality audio from YouTube videos for training data
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeDownloader:
    def __init__(self, output_dir: str = "audio_sources/raw_audio"):
        """Initialize YouTube downloader with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # YouTube download configuration
        self.yt_dlp_config = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
        }
    
    def download_video(self, url: str, output_filename: Optional[str] = None) -> bool:
        """Download a single YouTube video as high-quality WAV audio"""
        try:
            logger.info(f"Downloading video: {url}")
            
            # Build command
            cmd = [
                'yt-dlp',
                '--format', 'bestaudio/best',
                '--extract-audio',
                '--audio-format', 'wav',
                '--audio-quality', '192K',
                '--output', str(self.output_dir / '%(title)s.%(ext)s') if not output_filename else str(self.output_dir / f"{output_filename}.wav"),
                url
            ]
            
            # Execute download
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded: {url}")
                return True
            else:
                logger.error(f"Failed to download {url}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False
    
    def download_batch(self, urls: List[str], pattern_name: str = "unknown") -> Dict[str, bool]:
        """Download multiple videos and organize by pattern"""
        results = {}
        
        # Create pattern-specific directory
        pattern_dir = self.output_dir / pattern_name
        pattern_dir.mkdir(exist_ok=True)
        
        for i, url in enumerate(urls):
            filename = f"{pattern_name}_{i+1:03d}"
            success = self.download_video(url, filename)
            results[url] = success
            
        return results
    
    def get_sample_videos(self) -> List[Dict[str, str]]:
        """Return sample videos for initial testing"""
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
        """Create manifest of downloaded files"""
        manifest = {
            "total_files": 0,
            "patterns": {},
            "files": []
        }
        
        # Scan output directory
        for audio_file in self.output_dir.glob("**/*.wav"):
            relative_path = audio_file.relative_to(self.output_dir)
            pattern = relative_path.parent.name if relative_path.parent != Path(".") else "unknown"
            
            file_info = {
                "filename": audio_file.name,
                "path": str(relative_path),
                "pattern": pattern,
                "size_bytes": audio_file.stat().st_size,
                "duration_seconds": None  # Will be filled by audio analysis
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
        return manifest

def main():
    """Main function for testing the downloader"""
    downloader = YouTubeDownloader()
    
    # Get sample videos for testing
    sample_videos = downloader.get_sample_videos()
    
    print("🎵 Phin AI Dataset - YouTube Downloader")
    print("=" * 50)
    print(f"Output directory: {downloader.output_dir}")
    print(f"Sample videos to download: {len(sample_videos)}")
    print()
    
    # Download sample videos
    for video in sample_videos:
        print(f"Downloading: {video['title']}")
        success = downloader.download_video(video['url'], video['pattern'])
        status = "✅ Success" if success else "❌ Failed"
        print(f"Status: {status}")
        print()
    
    # Create dataset manifest
    manifest = downloader.create_dataset_manifest()
    print(f"Dataset manifest created with {manifest['total_files']} files")
    
    print("\n🎯 Next steps:")
    print("1. Check downloaded audio files in audio_sources/raw_audio/")
    print("2. Run audio preprocessing pipeline")
    print("3. Test with Basic Pitch transcription")

if __name__ == "__main__":
    main()