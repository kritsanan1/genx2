#!/usr/bin/env python3
"""
Mock YouTube Downloader for Phin AI Dataset
Creates synthetic Phin audio data for testing when YouTube downloads fail
"""

import os
import json
import logging
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockYouTubeDownloader:
    """
    Mock YouTube downloader that creates synthetic Phin audio data
    Useful for testing when real YouTube downloads fail
    """
    
    def __init__(self, output_dir: str = "audio_sources/raw_audio"):
        """Initialize mock downloader"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phin channels (for compatibility)
        self.phin_channels = {
            "ดุลย์เพลงพิณ": "https://www.youtube.com/@dulyplengphin",
            "M MUSIC GROUP": "https://www.youtube.com/@mmusicgroup5547",
            "สตีฟ ฐิติวัสส์": "https://www.youtube.com/@steve_thitiwat",
            "พิณอีสาน บ้านเฮา": "https://www.youtube.com/@phinisanbanhao",
            "เพลงพิณดนตรีอีสาน": "https://www.youtube.com/@phinmusicisan"
        }
        
        # Priority patterns
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
        
        # Sample rate
        self.sample_rate = 44100
        
        # Musical parameters for realistic Phin simulation
        self.phin_tuning = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C-D-E-F-G-A-B
        self.phin_scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "pentatonic": [0, 2, 4, 7, 9]
        }
        
        # Traditional Phin patterns (simplified)
        self.traditional_patterns = {
            "ลายนกไส่บินข้ามทุ่ง": {
                "rhythm": "moderate",
                "mood": "peaceful",
                "tempo_range": (80, 100),
                "scale": "pentatonic"
            },
            "ลายมโหรีอีสาน": {
                "rhythm": "processional",
                "mood": "ceremonial",
                "tempo_range": (60, 80),
                "scale": "major"
            },
            "ลายแมลงภู่ตอมดอกไม้": {
                "rhythm": "flowing",
                "mood": "gentle",
                "tempo_range": (90, 110),
                "scale": "pentatonic"
            },
            "ลายเต้ยโขง": {
                "rhythm": "energetic",
                "mood": "playful",
                "tempo_range": (100, 120),
                "scale": "major"
            },
            "ลายเซิ้งบั้งไฟ": {
                "rhythm": "festive",
                "mood": "celebratory",
                "tempo_range": (110, 130),
                "scale": "pentatonic"
            }
        }
    
    def generate_phin_melody(self, pattern_name: str, duration: float = 30.0) -> np.ndarray:
        """Generate a synthetic Phin melody based on pattern"""
        logger.info(f"Generating synthetic Phin melody: {pattern_name}")
        
        # Get pattern characteristics
        pattern_info = self.traditional_patterns.get(pattern_name, {
            "rhythm": "moderate",
            "mood": "peaceful", 
            "tempo_range": (80, 100),
            "scale": "pentatonic"
        })
        
        # Generate base parameters
        tempo = random.uniform(*pattern_info["tempo_range"])
        scale_type = pattern_info["scale"]
        scale_notes = self.phin_scales[scale_type]
        
        # Create time array
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Initialize audio array
        audio = np.zeros_like(t)
        
        # Generate melody based on traditional patterns
        beat_duration = 60.0 / tempo  # seconds per beat
        beat_samples = int(beat_duration * self.sample_rate)
        
        # Create melody with traditional phrasing
        note_duration_range = (1, 4)  # beats per note
        current_beat = 0
        
        while current_beat * beat_samples < len(t):
            # Select note from scale
            scale_degree = random.choice(scale_notes)
            base_freq = self.phin_tuning[scale_degree % len(self.phin_tuning)]
            
            # Add some ornamentation (typical for Phin)
            ornamentation = random.choice([0, 0, 2, -2])  # Sometimes add 2 semitones
            freq = base_freq * (2 ** (ornamentation / 12))
            
            # Note duration
            note_beats = random.uniform(*note_duration_range)
            note_samples = int(note_beats * beat_samples)
            
            # Time range for this note
            start_idx = int(current_beat * beat_samples)
            end_idx = min(start_idx + note_samples, len(t))
            
            if start_idx < len(t):
                # Generate note with envelope
                note_t = t[start_idx:end_idx] - t[start_idx]
                
                # Exponential decay envelope (typical for plucked strings)
                envelope = np.exp(-note_t * random.uniform(1, 3))
                
                # Fundamental frequency
                note = 0.4 * envelope * np.sin(2 * np.pi * freq * note_t)
                
                # Add harmonics (rich harmonic content for plucked string)
                note += 0.2 * envelope * np.sin(2 * np.pi * 2 * freq * note_t)
                note += 0.1 * envelope * np.sin(2 * np.pi * 3 * freq * note_t)
                note += 0.05 * envelope * np.sin(2 * np.pi * 4 * freq * note_t)
                
                # Add slight detuning for realism
                detune = random.uniform(-0.02, 0.02)
                note += 0.1 * envelope * np.sin(2 * np.pi * freq * (1 + detune) * note_t)
                
                audio[start_idx:end_idx] += note
                
                # Add subtle pitch bend typical of Phin
                if random.random() < 0.3:
                    bend_duration = min(0.5, note_beats * beat_duration * 0.5)
                    bend_samples = int(bend_duration * self.sample_rate)
                    bend_end = min(end_idx, start_idx + bend_samples)
                    if bend_end > start_idx:
                        bend_t = t[start_idx:bend_end] - t[start_idx]
                        bend_factor = 1 + 0.02 * np.sin(bend_t * 10)
                        audio[start_idx:bend_end] *= bend_factor
            
            current_beat += note_beats
        
        # Add some traditional rhythmic patterns
        self.add_rhythmic_patterns(audio, t, tempo, pattern_info)
        
        # Apply Phin-specific effects
        audio = self.apply_phin_effects(audio, pattern_info)
        
        return audio
    
    def add_rhythmic_patterns(self, audio: np.ndarray, t: np.ndarray, tempo: float, pattern_info: dict):
        """Add traditional rhythmic patterns"""
        # Add subtle rhythmic emphasis
        beat_duration = 60.0 / tempo
        beat_times = np.arange(0, t[-1], beat_duration)
        
        for i, beat_time in enumerate(beat_times):
            if beat_time < t[-1]:
                # Find closest sample index
                beat_idx = np.argmin(np.abs(t - beat_time))
                
                # Add subtle emphasis on strong beats
                if i % 4 == 0:  # Strong beat
                    emphasis = 1.0 + 0.1 * np.exp(-(t[beat_idx] - beat_time) * 5)
                    if beat_idx < len(audio):
                        audio[beat_idx:beat_idx+100] *= emphasis
    
    def apply_phin_effects(self, audio: np.ndarray, pattern_info: dict) -> np.ndarray:
        """Apply Phin-specific audio effects"""
        # Add subtle noise to simulate recording conditions
        noise_level = 0.005
        noise = noise_level * np.random.normal(0, 1, len(audio))
        audio += noise
        
        # Apply gentle low-pass filter (Phin has mellow tone)
        audio = self.apply_lowpass_filter(audio, cutoff=6000)
        
        # Add slight saturation (typical for traditional instruments)
        audio = np.tanh(audio * 0.8) * 1.2
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85
        
        return audio
    
    def apply_lowpass_filter(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply simple low-pass filter"""
        try:
            from scipy import signal
            nyquist = self.sample_rate / 2
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
            return signal.filtfilt(b, a, audio)
        except ImportError:
            logger.warning("scipy not available, returning original audio")
            return audio
    
    def download_video(self, url: str, output_filename: Optional[str] = None) -> bool:
        """Mock download - generates synthetic audio instead"""
        try:
            logger.info(f"Mock downloading from: {url}")
            
            # Extract pattern from URL or use random pattern
            pattern_name = None
            for pattern in self.priority_patterns:
                if pattern in url:
                    pattern_name = pattern
                    break
            
            if not pattern_name:
                pattern_name = random.choice(list(self.traditional_patterns.keys()))
            
            # Generate synthetic audio
            duration = random.uniform(25, 35)  # 25-35 seconds
            audio = self.generate_phin_melody(pattern_name, duration)
            
            # Save audio file
            if output_filename:
                output_path = self.output_dir / f"{output_filename}.wav"
            else:
                safe_title = f"synthetic_{pattern_name.replace(' ', '_')}"
                output_path = self.output_dir / f"{safe_title}.wav"
            
            sf.write(output_path, audio, self.sample_rate)
            
            logger.info(f"✅ Generated synthetic audio: {output_path}")
            logger.info(f"   Pattern: {pattern_name}")
            logger.info(f"   Duration: {duration:.1f}s")
            logger.info(f"   Sample rate: {self.sample_rate}Hz")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic audio: {str(e)}")
            return False
    
    def download_batch(self, urls: List[str], pattern_name: str = "unknown") -> Dict[str, bool]:
        """Download multiple videos and organize by pattern"""
        results = {}
        
        logger.info(f"Mock batch download for pattern: {pattern_name}")
        logger.info(f"URLs to process: {len(urls)}")
        
        for i, url in enumerate(urls):
            filename = f"{pattern_name}_{i+1:03d}"
            success = self.download_video(url, filename)
            results[url] = success
            
            # Add small delay for realism
            time.sleep(0.1)
        
        logger.info(f"Batch processing complete. Success: {sum(results.values())}/{len(urls)}")
        return results
    
    def get_sample_videos(self) -> List[Dict[str, str]]:
        """Return sample videos for testing"""
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
        """Create manifest of generated files"""
        manifest = {
            "total_files": 0,
            "patterns": {},
            "files": [],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "downloader_type": "mock",
            "synthetic_data": True
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
                "duration_seconds": None,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "synthetic": True
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
        
        logger.info(f"Mock dataset manifest created: {manifest_path}")
        return manifest


def main():
    """Main function for testing the mock downloader"""
    print("🎵 Mock Phin AI Dataset - YouTube Downloader")
    print("=" * 50)
    print("Note: This is a mock downloader that creates synthetic Phin audio data")
    print("Useful for testing when real YouTube downloads fail due to authentication")
    print()
    
    # Create mock downloader
    downloader = MockYouTubeDownloader()
    
    print(f"Output directory: {downloader.output_dir}")
    print(f"Available patterns: {len(downloader.traditional_patterns)}")
    print()
    
    # Get sample videos
    sample_videos = downloader.get_sample_videos()
    print(f"Sample videos to process: {len(sample_videos)}")
    print()
    
    # Process sample videos
    for video in sample_videos:
        print(f"Processing: {video['title']}")
        success = downloader.download_video(video['url'], video['pattern'])
        status = "✅ Success" if success else "❌ Failed"
        print(f"Status: {status}")
        print()
    
    # Create dataset manifest
    manifest = downloader.create_dataset_manifest()
    print(f"Mock dataset manifest created with {manifest['total_files']} files")
    
    print(f"\n🎯 Next steps:")
    print("1. Check generated audio files in audio_sources/raw_audio/")
    print("2. Run audio preprocessing pipeline on synthetic data")
    print("3. Use synthetic data for model training and testing")
    print("4. When ready, try real YouTube downloads with cookies/authentication")


if __name__ == "__main__":
    import time
    main()