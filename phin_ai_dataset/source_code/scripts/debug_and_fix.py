#!/usr/bin/env python3
"""
Comprehensive debugging and error handling script for Phin AI Dataset project.
This script provides utilities for debugging, error handling, and fixing common issues.
"""

import os
import sys
import json
import logging
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DebugAndFix:
    """Comprehensive debugging and error handling utilities."""
    
    def __init__(self, project_root: str = "/home/user/webapp/phin_ai_dataset"):
        self.project_root = Path(project_root)
        self.audio_dir = self.project_root / "audio_sources"
        self.raw_audio_dir = self.audio_dir / "raw_audio"
        self.preprocessed_dir = self.audio_dir / "preprocessed"
        self.test_dir = self.project_root / "tests"
        
    def check_project_structure(self) -> Dict[str, Any]:
        """Check if project structure is correct."""
        logger.info("Checking project structure...")
        
        structure_check = {
            "project_root_exists": self.project_root.exists(),
            "audio_dir_exists": self.audio_dir.exists(),
            "raw_audio_dir_exists": self.raw_audio_dir.exists(),
            "preprocessed_dir_exists": self.preprocessed_dir.exists(),
            "test_dir_exists": self.test_dir.exists(),
            "missing_directories": []
        }
        
        # Check each directory
        for dir_name, dir_path in {
            "project_root": self.project_root,
            "audio_dir": self.audio_dir,
            "raw_audio_dir": self.raw_audio_dir,
            "preprocessed_dir": self.preprocessed_dir,
            "test_dir": self.test_dir
        }.items():
            if not dir_path.exists():
                structure_check["missing_directories"].append(dir_name)
                logger.warning(f"Missing directory: {dir_path}")
        
        return structure_check
    
    def analyze_audio_files(self, directory: str = "raw_audio") -> Dict[str, Any]:
        """Analyze audio files for common issues."""
        target_dir = self.audio_dir / directory
        if not target_dir.exists():
            logger.error(f"Directory {target_dir} does not exist")
            return {"error": f"Directory {target_dir} does not exist"}
        
        logger.info(f"Analyzing audio files in {target_dir}")
        
        audio_files = list(target_dir.glob("*.wav"))
        analysis = {
            "total_files": len(audio_files),
            "files_analyzed": 0,
            "problematic_files": [],
            "duration_stats": [],
            "sample_rate_stats": []
        }
        
        for audio_file in audio_files:
            try:
                logger.info(f"Analyzing {audio_file}")
                
                # Load audio file
                y, sr = librosa.load(audio_file, sr=None)
                duration = len(y) / sr
                
                analysis["files_analyzed"] += 1
                analysis["duration_stats"].append(duration)
                analysis["sample_rate_stats"].append(sr)
                
                # Check for common issues
                issues = []
                
                # Check for very short files
                if duration < 1.0:
                    issues.append("very_short")
                
                # Check for very long files
                if duration > 300.0:  # 5 minutes
                    issues.append("very_long")
                
                # Check for silence
                if np.max(np.abs(y)) < 0.01:
                    issues.append("near_silence")
                
                # Check for DC offset
                if np.abs(np.mean(y)) > 0.1:
                    issues.append("dc_offset")
                
                # Check for clipping
                if np.max(np.abs(y)) > 0.95:
                    issues.append("possible_clipping")
                
                if issues:
                    analysis["problematic_files"].append({
                        "file": str(audio_file.name),
                        "duration": duration,
                        "sample_rate": sr,
                        "issues": issues
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {audio_file}: {str(e)}")
                analysis["problematic_files"].append({
                    "file": str(audio_file.name),
                    "error": str(e)
                })
        
        # Calculate statistics
        if analysis["duration_stats"]:
            analysis["duration_summary"] = {
                "min": min(analysis["duration_stats"]),
                "max": max(analysis["duration_stats"]),
                "mean": np.mean(analysis["duration_stats"]),
                "median": np.median(analysis["duration_stats"])
            }
        
        if analysis["sample_rate_stats"]:
            analysis["sample_rate_summary"] = {
                "unique_rates": list(set(analysis["sample_rate_stats"])),
                "most_common": max(set(analysis["sample_rate_stats"]), key=analysis["sample_rate_stats"].count)
            }
        
        return analysis
    
    def fix_dc_offset(self, input_file: str, output_file: str) -> bool:
        """Fix DC offset in audio file."""
        try:
            logger.info(f"Fixing DC offset for {input_file}")
            
            y, sr = librosa.load(input_file, sr=None)
            
            # Remove DC offset
            y_fixed = y - np.mean(y)
            
            # Save fixed audio
            sf.write(output_file, y_fixed, sr)
            logger.info(f"Fixed DC offset saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error fixing DC offset: {str(e)}")
            return False
    
    def fix_clipping(self, input_file: str, output_file: str) -> bool:
        """Fix clipping in audio file using soft limiting."""
        try:
            logger.info(f"Fixing clipping for {input_file}")
            
            y, sr = librosa.load(input_file, sr=None)
            
            # Apply soft limiting to reduce clipping
            # This is a simple approach - more sophisticated methods exist
            y_fixed = np.tanh(y * 0.95) / 0.95
            
            # Save fixed audio
            sf.write(output_file, y_fixed, sr)
            logger.info(f"Fixed clipping saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error fixing clipping: {str(e)}")
            return False
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        dependencies = {
            "librosa": {"import": "librosa", "version": "0.10.0"},
            "soundfile": {"import": "soundfile", "version": "0.12.1"},
            "numpy": {"import": "numpy", "version": "1.24.0"},
            "matplotlib": {"import": "matplotlib", "version": "3.7.0"},
            "scipy": {"import": "scipy", "version": "1.10.0"},
            "youtube_dl": {"import": "youtube_dl", "version": "2021.12.17"},
            "yt_dlp": {"import": "yt_dlp", "version": "2023.1.6"}
        }
        
        results = {}
        
        for name, info in dependencies.items():
            try:
                module = __import__(info["import"])
                version = getattr(module, "__version__", "unknown")
                results[name] = {
                    "installed": True,
                    "version": version,
                    "required_version": info["version"]
                }
                logger.info(f"{name}: {version} (required: {info['version']})")
            except ImportError:
                results[name] = {
                    "installed": False,
                    "version": None,
                    "required_version": info["version"]
                }
                logger.warning(f"{name}: not installed (required: {info['version']})")
        
        return results
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive checks and fixes."""
        logger.info("Running comprehensive checks and fixes...")
        
        results = {
            "timestamp": str(np.datetime64('now')),
            "structure_check": self.check_project_structure(),
            "dependencies": self.check_dependencies(),
            "audio_analysis": {}
        }
        
        # Analyze audio files if they exist
        if self.raw_audio_dir.exists():
            results["audio_analysis"]["raw_audio"] = self.analyze_audio_files("raw_audio")
        
        if self.preprocessed_dir.exists():
            results["audio_analysis"]["preprocessed"] = self.analyze_audio_files("preprocessed")
        
        # Save results
        results_file = self.project_root / "debug_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive check completed. Results saved to {results_file}")
        return results
    
    def fix_sample_file_issue(self) -> bool:
        """Fix the sample file creation issue in process_dataset."""
        logger.info("Fixing sample file creation issue...")
        
        try:
            # This is a placeholder for the actual fix
            # The issue is that process_dataset creates extra sample files when files are missing
            
            # Find and remove duplicate sample files
            if self.raw_audio_dir.exists():
                sample_files = list(self.raw_audio_dir.glob("*sample*.wav"))
                
                if len(sample_files) > 1:
                    # Keep only the first sample file
                    for sample_file in sample_files[1:]:
                        logger.info(f"Removing duplicate sample file: {sample_file}")
                        sample_file.unlink()
                
                logger.info("Sample file issue fixed")
                return True
            else:
                logger.warning("Raw audio directory does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Error fixing sample file issue: {str(e)}")
            return False

def main():
    """Main function."""
    debugger = DebugAndFix()
    
    print("Phin AI Dataset - Debug and Fix Tool")
    print("=" * 50)
    
    # Run comprehensive check
    results = debugger.run_comprehensive_check()
    
    # Print summary
    print("\nSummary:")
    print(f"- Project structure: {'OK' if not results['structure_check']['missing_directories'] else 'Issues found'}")
    print(f"- Audio files analyzed: {results['audio_analysis'].get('raw_audio', {}).get('total_files', 0)}")
    print(f"- Problematic files: {len(results['audio_analysis'].get('raw_audio', {}).get('problematic_files', []))}")
    
    # Check dependencies
    missing_deps = [name for name, info in results['dependencies'].items() if not info['installed']]
    if missing_deps:
        print(f"- Missing dependencies: {', '.join(missing_deps)}")
    else:
        print("- All dependencies installed")
    
    print(f"\nDetailed results saved to: debug_results.json")
    print("Check debug.log for detailed logs")

if __name__ == "__main__":
    main()