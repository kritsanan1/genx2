#!/bin/bash
# Script สำหรับดาวน์โหลดวิดีโอพิณอีสาน
# วันที่สร้าง: 25 พฤศจิกายน 2025

echo "🎵 เริ่มดาวน์โหลดวิดีโอพิณอีสาน..."

# สร้างโฟลเดอร์
mkdir -p phin_audio_dataset/{basics,lai_lam_perlin,lai_hae,lai_mahoree,techniques,covers}

# Array ของ URLs พร้อมหมวดหมู่
declare -A videos=(
  ["basics"]="ksZ3DWA9mPE 1mBXmd5nD4s"
  ["lai_lam_perlin"]="pKCaf-f19rQ ZRK75tNHqKc 9dERGSNL5Ak fNWa5EbppDc 7EZJ6YEWeMI xouLuPjn90A"
  ["lai_hae"]="lWp9Y66qzeE HJZxuD57joI RpSV75Thj4E Aavl7vllMP4 gyDbsN6jbzc 0KHy-5lQYoo"
  ["lai_mahoree"]="ZT7q9pcWLDc"
  ["techniques"]="bnjzxgUC6jI HGyFU1gm2Zc"
  ["covers"]="pX5_9tpeG9k biVIbpcOEgQ -l1Pj7N_eI8"
)

# ดาวน์โหลดแต่ละหมวดหมู่
for category in "${!videos[@]}"; do
  echo "📁 กำลังดาวน์โหลดหมวด: $category"
  
  for video_id in ${videos[$category]}; do
    echo "  ⬇️  Downloading: $video_id"
    yt-dlp -f bestaudio --extract-audio --audio-format wav \
      --audio-quality 0 \
      --postprocessor-args "-ar 22050" \
      -o "phin_audio_dataset/$category/%(title)s.%(ext)s" \
      "https://www.youtube.com/watch?v=$video_id"
    
    # หน่วงเวลา 2 วินาที
    sleep 2
  done
done

echo "✅ ดาวน์โหลดเสร็จสิ้น!"
echo "📊 สรุป:"
find phin_audio_dataset -name "*.wav" | wc -l
echo "ไฟล์ WAV ทั้งหมด"
du -sh phin_audio_dataset
