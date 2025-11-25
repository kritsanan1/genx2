#!/bin/bash
# Batch download script for Phin YouTube videos
# Usage: ./download_phin_videos.sh [pattern_name] [video_urls_file]

set -e

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/audio_sources/raw_audio"
PATTERN_NAME="${1:-unknown}"
URLS_FILE="${2:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v yt-dlp &> /dev/null; then
        error "yt-dlp not found. Please install: pip install yt-dlp"
        exit 1
    fi
    
    if ! command -v ffmpeg &> /dev/null; then
        error "ffmpeg not found. Please install: apt-get install ffmpeg"
        exit 1
    fi
    
    success "All dependencies satisfied"
}

# Create output directory
create_output_dir() {
    log "Creating output directory: $OUTPUT_DIR/$PATTERN_NAME"
    mkdir -p "$OUTPUT_DIR/$PATTERN_NAME"
}

# Download single video
download_video() {
    local url=$1
    local filename=$2
    local output_path="$OUTPUT_DIR/$PATTERN_NAME/$filename.wav"
    
    log "Downloading: $url"
    log "Output: $output_path"
    
    yt-dlp \
        --format "bestaudio/best" \
        --extract-audio \
        --audio-format wav \
        --audio-quality 192K \
        --output "$output_path" \
        --quiet \
        "$url"
    
    if [ $? -eq 0 ]; then
        success "Downloaded: $filename.wav"
    else
        error "Failed to download: $url"
        return 1
    fi
}

# Download from file
batch_download_from_file() {
    local urls_file=$1
    local count=0
    local success_count=0
    
    log "Batch downloading from file: $urls_file"
    
    while IFS= read -r line; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        count=$((count + 1))
        filename="${PATTERN_NAME}_$(printf "%03d" $count)"
        
        if download_video "$line" "$filename"; then
            success_count=$((success_count + 1))
        fi
    done < "$urls_file"
    
    log "Download summary: $success_count/$count videos downloaded successfully"
}

# Download priority patterns
priority_download() {
    log "Downloading priority Phin patterns..."
    
    # Define priority videos for each pattern
    declare -A priority_videos=(
        ["ลายนกไส่บินข้ามทุ่ง"]="https://www.youtube.com/watch?v=ksZ3DWA9mPE"
        ["ลายมโหรีอีสาน"]="https://www.youtube.com/watch?v=ZRK75tNHqKc"
        ["ลายแมลงภู่ตอมดอกไม้"]="https://www.youtube.com/watch?v=aQZEN3y8zWo"
        ["ลายเต้ยโขง"]="https://www.youtube.com/watch?v=example1"
        ["ลายเซิ้งบั้งไฟ"]="https://www.youtube.com/watch?v=example2"
    )
    
    for pattern in "${!priority_videos[@]}"; do
        log "Downloading pattern: $pattern"
        mkdir -p "$OUTPUT_DIR/$pattern"
        
        url="${priority_videos[$pattern]}"
        filename="${pattern}_001"
        
        download_video "$url" "$filename"
    done
}

# Main function
main() {
    log "🎵 Phin AI Dataset - YouTube Video Downloader"
    log "=============================================="
    
    check_dependencies
    create_output_dir
    
    if [[ -n "$URLS_FILE" && -f "$URLS_FILE" ]]; then
        batch_download_from_file "$URLS_FILE"
    elif [[ "$PATTERN_NAME" == "priority" ]]; then
        priority_download
    else
        log "Pattern: $PATTERN_NAME"
        log "Usage examples:"
        log "  ./download_phin_videos.sh ลายนกไส่บินข้ามทุ่ง"
        log "  ./download_phin_videos.sh priority"
        log "  ./download_phin_videos.sh custom urls.txt"
    fi
    
    success "Download process completed!"
    log "Check output directory: $OUTPUT_DIR/$PATTERN_NAME"
}

# Run main function
main "$@"