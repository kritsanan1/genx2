# 🚀 Quick Start Guide: การเริ่มต้นใช้งาน Dataset พิณอีสาน

**เอกสารนี้เหมาะสำหรับผู้เริ่มต้น**

---

## ✅ **Checklist ก่อนเริ่มต้น**

- [ ] Python 3.8+ installed
- [ ] 10+ GB พื้นที่ว่างบน hard disk
- [ ] Internet connection (สำหรับดาวน์โหลด)
- [ ] (Optional) GPU สำหรับเทรนโมเดล

---

## 📦 **ขั้นตอนที่ 1: ติดตั้ง Dependencies**

### **สร้าง Virtual Environment:**
```bash
# สร้าง environment
python -m venv phin_env

# เปิดใช้งาน (Linux/Mac)
source phin_env/bin/activate

# เปิดใช้งาน (Windows)
phin_env\Scripts\activate
```

### **ติดตั้ง Libraries:**
```bash
# Core libraries
pip install numpy scipy matplotlib

# Audio processing
pip install librosa soundfile pydub

# MIDI handling
pip install pretty_midi mido music21

# Machine learning
pip install tensorflow torch  # เลือกอันใดอันหนึ่ง
pip install basic-pitch omnizart

# Evaluation
pip install mir_eval jams

# YouTube download
pip install yt-dlp

# Utilities
pip install tqdm pandas jupyter
```

### **ตรวจสอบการติดตั้ง:**
```python
import librosa
import tensorflow as tf
import pretty_midi
print("✓ All libraries installed successfully!")
print(f"Librosa version: {librosa.__version__}")
print(f"TensorFlow version: {tf.__version__}")
```

---

## 📥 **ขั้นตอนที่ 2: ดาวน์โหลดข้อมูล**

### **2.1 ดาวน์โหลดวิดีโอจาก YouTube**

```bash
# ดาวน์โหลดวิดีโอเดี่ยว
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "audio_sources/%(title)s.%(ext)s" \
  https://www.youtube.com/watch?v=aQZEN3y8zWo

# ดาวน์โหลดหลายๆ วิดีโอพร้อมกัน
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  -o "audio_sources/%(title)s.%(ext)s" \
  -a video_urls.txt
```

**ไฟล์ `video_urls.txt`:**
```
https://www.youtube.com/watch?v=ksZ3DWA9mPE
https://www.youtube.com/watch?v=ZRK75tNHqKc
https://www.youtube.com/watch?v=aQZEN3y8zWo
```

### **2.2 ใช้ Python Script**

```python
# download_dataset.py
import subprocess
from pathlib import Path

def download_video(url, output_dir="audio_sources"):
    """ดาวน์โหลดวิดีโอแปลงเป็น WAV"""
    Path(output_dir).mkdir(exist_ok=True)
    
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        url
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✓ Downloaded: {url}")

# รายการวิดีโอ
videos = [
    "https://www.youtube.com/watch?v=ksZ3DWA9mPE",
    "https://www.youtube.com/watch?v=ZRK75tNHqKc",
    "https://www.youtube.com/watch?v=aQZEN3y8zWo"
]

for url in videos:
    try:
        download_video(url)
    except Exception as e:
        print(f"✗ Failed: {url} - {e}")
```

---

## 🎵 **ขั้นตอนที่ 3: Audio Preprocessing**

### **3.1 Basic Processing**

```python
# process_audio.py
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def process_single_file(input_path, output_path):
    """
    ประมวลผลไฟล์เดี่ยว:
    - Load
    - Normalize
    - Denoise
    - Save
    """
    print(f"Processing: {input_path}")
    
    # Load
    y, sr = librosa.load(input_path, sr=22050)
    
    # Normalize
    y = librosa.util.normalize(y)
    
    # Simple noise reduction (trim silence)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    # Save
    sf.write(output_path, y_trimmed, sr)
    print(f"✓ Saved: {output_path}")

# Process all files in directory
input_dir = Path("audio_sources")
output_dir = Path("processed_audio")
output_dir.mkdir(exist_ok=True)

for audio_file in input_dir.glob("*.wav"):
    output_path = output_dir / audio_file.name
    process_single_file(audio_file, output_path)
```

### **3.2 Extract Features**

```python
# extract_features.py
import librosa
import numpy as np
import matplotlib.pyplot as plt

def visualize_audio(audio_path):
    """แสดง waveform และ spectrogram"""
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time (s)')
    
    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel',
        sr=sr, ax=axes[1]
    )
    axes[1].set_title('Mel Spectrogram')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    img = librosa.display.specshow(
        chroma, x_axis='time', y_axis='chroma',
        sr=sr, ax=axes[2]
    )
    axes[2].set_title('Chromagram')
    fig.colorbar(img, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('audio_visualization.png', dpi=150)
    plt.show()

# Test
visualize_audio("processed_audio/your_audio.wav")
```

---

## 🤖 **ขั้นตอนที่ 4: ทดลอง Transcription**

### **4.1 ใช้ Spotify Basic Pitch**

```python
# test_basic_pitch.py
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

def transcribe_with_basic_pitch(audio_path, output_dir="output_midi"):
    """Transcribe ด้วย Basic Pitch"""
    predict_and_save(
        [audio_path],
        output_dir,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False
    )
    print(f"✓ MIDI saved to: {output_dir}")

# Test
transcribe_with_basic_pitch("processed_audio/lai_mahoree.wav")
```

### **4.2 ใช้ EWMA Method (จากงานวิจัย KMUTT)**

```python
# test_ewma.py
import librosa
import numpy as np
import pretty_midi

def simple_transcription(audio_path, output_midi="output.mid"):
    """
    Simple transcription ด้วย onset + pitch detection
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, wait=10)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Detect pitches at onsets
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Create MIDI
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    for i, onset_time in enumerate(onset_times):
        # Get pitch at onset
        onset_frame = onset_frames[i]
        pitch_values = pitches[:, onset_frame]
        
        if pitch_values.max() > 0:
            pitch_idx = pitch_values.argmax()
            pitch_hz = librosa.midi_to_hz(pitch_idx)
            midi_note = librosa.hz_to_midi(pitch_hz)
            
            # Estimate duration (until next onset or 0.5s)
            if i < len(onset_times) - 1:
                duration = onset_times[i+1] - onset_time
            else:
                duration = 0.5
            
            # Create note
            note = pretty_midi.Note(
                velocity=100,
                pitch=int(midi_note),
                start=onset_time,
                end=onset_time + duration
            )
            instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    midi.write(output_midi)
    print(f"✓ MIDI saved: {output_midi}")

# Test
simple_transcription("processed_audio/lai_mahoree.wav")
```

---

## 📊 **ขั้นตอนที่ 5: Evaluation**

### **5.1 Compare MIDI Files**

```python
# compare_midi.py
import pretty_midi
import mir_eval
import numpy as np

def compare_transcriptions(reference_midi, estimated_midi):
    """เปรียบเทียบ MIDI 2 ไฟล์"""
    
    # Load MIDI files
    ref = pretty_midi.PrettyMIDI(reference_midi)
    est = pretty_midi.PrettyMIDI(estimated_midi)
    
    # Extract notes
    ref_intervals = []
    ref_pitches = []
    for instrument in ref.instruments:
        for note in instrument.notes:
            ref_intervals.append([note.start, note.end])
            ref_pitches.append(note.pitch)
    
    est_intervals = []
    est_pitches = []
    for instrument in est.instruments:
        for note in instrument.notes:
            est_intervals.append([note.start, note.end])
            est_pitches.append(note.pitch)
    
    ref_intervals = np.array(ref_intervals)
    est_intervals = np.array(est_intervals)
    ref_pitches = np.array(ref_pitches)
    est_pitches = np.array(est_pitches)
    
    # Evaluate
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches
    )
    
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")
    
    return precision, recall, f1

# Test
compare_transcriptions(
    "ground_truth/lai_mahoree.mid",
    "output_midi/lai_mahoree.mid"
)
```

---

## 📚 **ขั้นตอนที่ 6: Build Dataset**

### **6.1 Organize Files**

```python
# organize_dataset.py
from pathlib import Path
import shutil
import json

def organize_dataset(audio_dir, midi_dir, output_dir="organized_dataset"):
    """
    จัดเรียง dataset เป็นโครงสร้างที่เหมาะสม
    """
    output_path = Path(output_dir)
    
    # Create directories
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "test").mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = list(Path(audio_dir).glob("*.wav"))
    
    # Split: 80% train, 10% val, 10% test
    n_total = len(audio_files)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_files = audio_files[:n_train]
    val_files = audio_files[n_train:n_train+n_val]
    test_files = audio_files[n_train+n_val:]
    
    # Copy files
    for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        for audio_file in files:
            # Copy audio
            shutil.copy(audio_file, output_path / split / audio_file.name)
            
            # Copy corresponding MIDI if exists
            midi_file = Path(midi_dir) / audio_file.with_suffix(".mid").name
            if midi_file.exists():
                shutil.copy(midi_file, output_path / split / midi_file.name)
        
        print(f"✓ {split}: {len(files)} files")
    
    # Create metadata
    metadata = {
        "total_files": n_total,
        "train": n_train,
        "val": n_val,
        "test": len(test_files),
        "audio_format": "wav",
        "sample_rate": 22050
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset organized in: {output_dir}")

# Usage
organize_dataset("processed_audio", "ground_truth_midi")
```

---

## 🎯 **Next Steps**

### **หลังจากทำตามขั้นตอนข้างต้นแล้ว:**

1. ✅ **รวบรวมข้อมูลเพิ่ม:** 
   - ดาวน์โหลดวิดีโอเพิ่มเติม (20-30 วิดีโอต่อลายพิณ)
   - หาโน๊ตดนตรีหรือ MIDI files เพื่อเป็น ground truth

2. ✅ **Data Augmentation:**
   - Time stretching
   - Pitch shifting
   - Add noise
   
3. ✅ **Train Custom Model:**
   - ใช้ TensorFlow/PyTorch
   - Fine-tune pre-trained models
   - Experiment with architectures

4. ✅ **Evaluate & Iterate:**
   - ทดสอบกับลายพิณต่างๆ
   - ปรับ hyperparameters
   - วัดผล F1-score

5. ✅ **Deploy:**
   - สร้าง web interface
   - Export model เป็น ONNX
   - Share บน GitHub

---

## 🆘 **Troubleshooting**

### **ปัญหา: yt-dlp ดาวน์โหลดไม่ได้**
```bash
# อัปเดต yt-dlp
pip install -U yt-dlp

# ใช้ proxy (ถ้าจำเป็น)
yt-dlp --proxy http://proxy:port <URL>
```

### **ปัญหา: librosa ใช้ไม่ได้**
```bash
# ติดตั้ง dependencies เพิ่ม
pip install numba audioread

# หรือใช้ conda
conda install -c conda-forge librosa
```

### **ปัญหา: MIDI file เปิดไม่ได้**
```python
# ตรวจสอบ MIDI file
import pretty_midi

try:
    midi = pretty_midi.PrettyMIDI("file.mid")
    print(f"✓ Valid MIDI: {len(midi.instruments)} instruments")
except Exception as e:
    print(f"✗ Invalid MIDI: {e}")
```

---

## 💡 **Tips**

1. **เริ่มจากง่าย:** ทดลองกับ 1-2 ลายพิณก่อน
2. **ใช้ GPU:** เร็วกว่า CPU มาก (10-100x)
3. **Checkpoint บ่อยๆ:** บันทึกผลงานเป็นระยะ
4. **Visualize:** ใช้ matplotlib ดูข้อมูลเสมอ
5. **Documentation:** เขียนบันทึกสิ่งที่ทำ

---

## 📖 **เอกสารเพิ่มเติม**

- **คู่มือหลัก:** `phin_dataset_master_guide.md`
- **แหล่งวิดีโอ:** `youtube_sources.md`
- **Training Pipeline:** `training_pipeline.md`

---

**Happy Coding! 🎵🤖**
