# 🎬 รายการวิดีโอ YouTube สำหรับดาวน์โหลด

**วันที่รวบรวม:** 25 พฤศจิกายน 2025

---

## 📥 **คำสั่งดาวน์โหลด (Copy & Paste)**

### **ติดตั้ง yt-dlp:**
```bash
pip install yt-dlp
```

### **ดาวน์โหลดทีละวิดีโอ:**
```bash
# Template
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  --audio-quality 0 \
  -o "%(upload_date)s_%(title)s.%(ext)s" \
  "VIDEO_URL"
```

---

## 🎵 **รายการวิดีโอแนะนำ (20 วิดีโอแรก)**

### **1. พื้นฐานพิณ**
```bash
# [82K+ views] สอนพิณพื้นฐาน - ดุลย์เพลงพิณ
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "01_พื้นฐาน_ดุลย์เพลงพิณ.wav" \
  "https://www.youtube.com/watch?v=ksZ3DWA9mPE"

# [340K+ views] สอนดีดพิณเบื้องต้นสำหรับมือใหม่ - นายนาจาน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "02_เบื้องต้นมือใหม่_นายนาจาน.wav" \
  "https://www.youtube.com/watch?v=1mBXmd5nD4s"
```

### **2. ลายลำเพลิน**
```bash
# [200K+ views] ลายลำเพลินต่อเนื่อง - M MUSIC GROUP
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "03_ลายลำเพลิน_M_MUSIC.wav" \
  "https://www.youtube.com/watch?v=pKCaf-f19rQ"

# [353K+ views] เทคนิคการไหลพิณ - M MUSIC GROUP
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "04_เทคนิคไหลพิณ_M_MUSIC.wav" \
  "https://www.youtube.com/watch?v=ZRK75tNHqKc"

# [64K+ views] ลายลำเพลินง่ายๆ EP1 - มูนมังอีสาน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "05_ลำเพลินง่าย_EP1_มูนมัง.wav" \
  "https://www.youtube.com/watch?v=9dERGSNL5Ak"

# [20K+ views] ลายลำเพลินง่ายๆ EP2 - มูนมังอีสาน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "06_ลำเพลินง่าย_EP2_มูนมัง.wav" \
  "https://www.youtube.com/watch?v=fNWa5EbppDc"

# [18K+ views] ลายลำเพลิน EP12 - ดุลย์เพลงพิณ
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "07_ลำเพลิน_EP12_ดุลย์.wav" \
  "https://www.youtube.com/watch?v=7EZJ6YEWeMI"

# [3K+ views] ลายลำเพลินสั้นๆ - มูนมังอีสาน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "08_ลำเพลินสั้น_มูนมัง.wav" \
  "https://www.youtube.com/watch?v=xouLuPjn90A"
```

### **3. ลายแห่**
```bash
# [116K+ views] ลายเลาะบ้าน - สตีฟ ฐิติวัสส์
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "09_ลายเลาะบ้าน_สตีฟ.wav" \
  "https://www.youtube.com/watch?v=lWp9Y66qzeE"

# [31K+ views] ลายแห่ ลูกห่าว - ลูกอีสาน มักม่วน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "10_ลายแห่_ลูกห่าว_ลูกอีสาน.wav" \
  "https://www.youtube.com/watch?v=HJZxuD57joI"

# [44K+ views] ลายแห่สงกรานต์ - มูนมังอีสาน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "11_ลายแห่_สงกรานต์_มูนมัง.wav" \
  "https://www.youtube.com/watch?v=RpSV75Thj4E"

# [18K+ views] ลายแห่ แบบที่ 1 - สตีฟ ฐิติวัสส์
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "12_ลายแห่_แบบที่1_สตีฟ.wav" \
  "https://www.youtube.com/watch?v=Aavl7vllMP4"

# [9K+ views] ลายแห่ในตำนาน - มูนมังอีสาน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "13_ลายแห่_ตำนาน_มูนมัง.wav" \
  "https://www.youtube.com/watch?v=gyDbsN6jbzc"

# [3K+ views] ลายแห่ สั้นๆ - มูนมังอีสาน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "14_ลายแห่_สั้น_มูนมัง.wav" \
  "https://www.youtube.com/watch?v=0KHy-5lQYoo"
```

### **4. ลายมโหรีอีสาน**
```bash
# [7K+ views] ลายมโหรีอีสาน - ต้อม โปงลาง
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "15_มโหรีอีสาน_ต้อมโปงลาง.wav" \
  "https://www.youtube.com/watch?v=ZT7q9pcWLDc"
```

### **5. เทคนิคและลูกเล่น**
```bash
# [32K+ views] เทคนิคลูกเล่น - เดี่ยว วรวีร์
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "16_เทคนิคลูกเล่น_เดี่ยววรวีร์.wav" \
  "https://www.youtube.com/watch?v=bnjzxgUC6jI"

# [45K+ views] สอนเกริ่นพิณพื้นฐาน - ร้านพิณน๊อตชลบุรี
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "17_เกริ่นพิณ_พิณน๊อต.wav" \
  "https://www.youtube.com/watch?v=HGyFU1gm2Zc"
```

### **6. เพลงCover**
```bash
# [14K+ views] กุหลาบแดง - ต้อม โปงลาง
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "18_กุหลาบแดง_ต้อมโปงลาง.wav" \
  "https://www.youtube.com/watch?v=pX5_9tpeG9k"

# [10K+ views] พิณอิมโพรไวส์ - สตีฟ ฐิติวัสส์
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "19_พิณอิมโพรไวส์_สตีฟ.wav" \
  "https://www.youtube.com/watch?v=biVIbpcOEgQ"

# [12K+ views] อีสานบ้านเฮา - มูนมังอีสาน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "20_อีสานบ้านเฮา_มูนมัง.wav" \
  "https://www.youtube.com/watch?v=-l1Pj7N_eI8"
```

---

## 📦 **ดาวน์โหลดทั้งหมด (20 วิดีโอ) ในคำสั่งเดียว**

```bash
#!/bin/bash
# สร้างโฟลเดอร์
mkdir -p phin_audio_dataset

# Array ของ URLs
declare -a urls=(
  "https://www.youtube.com/watch?v=ksZ3DWA9mPE"
  "https://www.youtube.com/watch?v=1mBXmd5nD4s"
  "https://www.youtube.com/watch?v=pKCaf-f19rQ"
  "https://www.youtube.com/watch?v=ZRK75tNHqKc"
  "https://www.youtube.com/watch?v=9dERGSNL5Ak"
  "https://www.youtube.com/watch?v=fNWa5EbppDc"
  "https://www.youtube.com/watch?v=7EZJ6YEWeMI"
  "https://www.youtube.com/watch?v=xouLuPjn90A"
  "https://www.youtube.com/watch?v=lWp9Y66qzeE"
  "https://www.youtube.com/watch?v=HJZxuD57joI"
  "https://www.youtube.com/watch?v=RpSV75Thj4E"
  "https://www.youtube.com/watch?v=Aavl7vllMP4"
  "https://www.youtube.com/watch?v=gyDbsN6jbzc"
  "https://www.youtube.com/watch?v=0KHy-5lQYoo"
  "https://www.youtube.com/watch?v=ZT7q9pcWLDc"
  "https://www.youtube.com/watch?v=bnjzxgUC6jI"
  "https://www.youtube.com/watch?v=HGyFU1gm2Zc"
  "https://www.youtube.com/watch?v=pX5_9tpeG9k"
  "https://www.youtube.com/watch?v=biVIbpcOEgQ"
  "https://www.youtube.com/watch?v=-l1Pj7N_eI8"
)

# ดาวน์โหลดทีละไฟล์
for url in "${urls[@]}"; do
  echo "Downloading: $url"
  yt-dlp -f bestaudio --extract-audio --audio-format wav \
    --audio-quality 0 \
    -o "phin_audio_dataset/%(playlist_index)s_%(title)s.%(ext)s" \
    "$url"
  sleep 2  # Delay เพื่อไม่ให้โหลดเร็วเกินไป
done

echo "✅ Downloaded 20 videos successfully!"
```

---

## 📊 **สรุปข้อมูลที่จะได้**

| หมวดหมู่ | จำนวน | ประมาณเวลา |
|----------|--------|------------|
| พื้นฐานพิณ | 2 วิดีโอ | ~6 นาที |
| ลายลำเพลิน | 6 วิดีโอ | ~90 นาที |
| ลายแห่ | 6 วิดีโอ | ~125 นาที |
| ลายมโหรีอีสาน | 1 วิดีโอ | ~8 นาที |
| เทคนิค/ลูกเล่น | 2 วิดีโอ | ~36 นาที |
| เพลง Cover | 3 วิดีโอ | ~34 นาที |

**รวม:** 20 วิดีโอ, ~5 ชั่วโมง, ขนาดประมาณ **2-3 GB** (WAV format)

---

## ⚙️ **การตั้งค่าที่แนะนำ**

```bash
# ตัวเลือกเต็ม
yt-dlp \
  --format bestaudio \
  --extract-audio \
  --audio-format wav \
  --audio-quality 0 \
  --postprocessor-args "-ar 22050" \
  --write-info-json \
  --write-thumbnail \
  --output "%(upload_date)s_%(title)s.%(ext)s" \
  VIDEO_URL
```

**พารามิเตอร์สำคัญ:**
- `--audio-quality 0`: คุณภาพสูงสุด
- `-ar 22050`: Sample rate 22.05 kHz (เพียงพอสำหรับพิณ)
- `--write-info-json`: บันทึก metadata
- `--write-thumbnail`: ดาวน์โหลด thumbnail

---

## 🔄 **การแปลงไฟล์ (หลังดาวน์โหลด)**

### **แปลงเป็น MP3 (เพื่อประหยัดพื้นที่):**
```bash
for file in *.wav; do
  ffmpeg -i "$file" -ab 320k "${file%.wav}.mp3"
done
```

### **Normalize volume:**
```bash
for file in *.wav; do
  ffmpeg -i "$file" -af "loudnorm" "${file%.wav}_normalized.wav"
done
```

### **Extract 5-second clips:**
```bash
for file in *.wav; do
  ffmpeg -i "$file" -t 5 -c copy "${file%.wav}_5sec.wav"
done
```

---

**หมายเหตุ:** โปรดเคารพลิขสิทธิ์ของผู้สร้างวิดีโอ ข้อมูลนี้ควรใช้เพื่อการศึกษาและวิจัยเท่านั้น
