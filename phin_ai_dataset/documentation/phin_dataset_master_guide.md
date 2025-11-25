# 🎵 พิณอีสาน AI Dataset - คู่มือหลัก
**สร้างเมื่อ: 25 พฤศจิกายน 2025**

---

## 📚 **ภาพรวมโปรเจค**

โปรเจคนี้รวบรวมข้อมูลครบวงจรสำหรับการพัฒนา **AI Model สำหรับการเรียนรู้และ Transcription ลายพิณอีสาน** ซึ่งเป็นเครื่องดนตรีไทยโบราณที่มีเอกลักษณ์เฉพาะ

### **วัตถุประสงค์:**
1. ✅ รวบรวมข้อมูลเสียงพิณคุณภาพสูงจากแหล่งต่างๆ
2. ✅ เก็บโน๊ตดนตรีลายพิณต่างๆ สำหรับเป็น Ground Truth
3. ✅ รวม Source Code และงานวิจัยที่เกี่ยวข้อง
4. ✅ พัฒนาโมเดล AI ที่เข้าใจระบบดนตรีไทย (7-tone system)

---

## 📁 **โครงสร้างโฟลเดอร์**

```
/phin_ai_dataset/
├── audio_sources/        # ไฟล์เสียงพิณจาก YouTube และแหล่งอื่นๆ
├── sheet_music/          # โน๊ตดนตรีลายพิณต่างๆ
├── research_papers/      # งานวิจัยและเอกสารวิชาการ
├── source_code/          # โค้ดสำหรับ training และ preprocessing
└── documentation/        # เอกสารคำแนะนำและบันทึก
```

---

## 🎼 **ลายพิณที่ควรรวบรวม (Priority List)**

### **ลายพิณหลัก (High Priority):**
1. **ลายนกไส่บินข้ามทุ่ง** - ลายพื้นฐานที่นิยมเรียนกันมาก
2. **ลายมโหรีอีสาน** - ลายที่มีความซับซ้อนและสวยงาม
3. **ลายแมลงภู่ตอมดอกไม้** - ลายที่มีจังหวะเร็วและท้าทาย
4. **ลายเต้ยโขง** - ลายคลาสสิกที่มีในงานวิจัยหลายชิ้น
5. **ลายเซิ้งบั้งไฟ** - ลายที่มีลักษณะเด่นชัด

### **ลายพิณเสริม (Medium Priority):**
- ลายเต้ยพม่า
- ลายโปงลาง
- ลายลำเพลิน
- ลายแห่

---

## 🌐 **แหล่งข้อมูลหลัก**

### **1. YouTube Channels (สำหรับเสียงและวิดีโอ)**

#### **ช่องสอนพิณที่แนะนำ:**
- **ดุลย์เพลงพิณ**
  - วิดีโอ: [สอนพิณพื้นฐาน](https://www.youtube.com/watch?v=ksZ3DWA9mPE)
  - มีการสอนแบบเป็นขั้นตอน เหมาะสำหรับ beginner
  
- **M MUSIC GROUP**
  - วิดีโอ: [เทคนิคการไหลพิณ](https://www.youtube.com/watch?v=ZRK75tNHqKc)
  - โฟกัสเทคนิคการเล่นที่ละเอียด
  
- **สตีฟ ฐิติวัสส์**
  - วิดีโอ: [ลายมโหรีอีสาน](https://www.youtube.com/watch?v=aQZEN3y8zWo)
  - มีคุณภาพเสียงดีมาก เหมาะสำหรับเทรนโมเดล

#### **วิธีดาวน์โหลด:**
```bash
# ใช้ yt-dlp (tool ที่แนะนำ)
yt-dlp -f bestaudio[ext=m4a] --extract-audio --audio-format wav <URL>

# หรือใช้ youtube-dl
youtube-dl -x --audio-format wav <URL>
```

---

### **2. โน๊ตดนตรี (Sheet Music)**

#### **แหล่งโน๊ตลายพิณ:**
- **Guitar285 WordPress**
  - URL: https://guitar285.wordpress.com/category/โน๊ตลายพิณ/
  - มีโน๊ตลายพิณหลากหลายรูปแบบ
  - ใช้ระบบโซลฟาไทย (ด ร ม ฟ ซ ล ท)

#### **การแปลงโน๊ต:**
- โซลฟาไทย → Western Notation:
  - ด (Do) = C
  - ร (Re) = D
  - ม (Mi) = E
  - ฟ (Fa) = F
  - ซ (Sol) = G
  - ล (La) = A
  - ท (Ti) = B

---

## 🔬 **งานวิจัยสำคัญ**

### **1. Automatic Music Transcription for Thai Xylophone (KMUTT)**
- **สถาบัน:** มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี
- **ปี:** 2567 (2024)
- **เทคนิค:** Energy-based Windowed Moving Average (EWMA) + FFT + Peak Detection
- **ผลลัพธ์:**
  - Onset Detection F1-score: **98.54%**
  - Pitch Detection F1-score: **97.34%**
- **ข้อดี:** 
  - ไม่ต้องใช้ Deep Learning (ทำงานบน embedded devices ได้)
  - เหมาะกับดนตรีไทยที่มีโครงสร้างเฉพาะ
- **เอกสาร:** https://inc.kmutt.ac.th/download/capstone_design_projects/2567/10.pdf

### **2. Stepping Towards Transcultural Machine Learning in Music (Google Magenta)**
- **ประเด็น:** AI models ส่วนใหญ่ถูกเทรนด้วยดนตรีตะวันตก (12-tone system)
- **ปัญหา:** ไม่เหมาะกับดนตรีไทยที่ใช้ 7-tone system
- **แนวทาง:** ต้องสร้าง model ที่เข้าใจ cultural context
- **ลิงก์:** https://magenta.withgoogle.com/transcultural

### **3. Deep Learning for Music Genre Classification: Thai Music**
- **วิธีการ:** ใช้ CNN และ RNN
- **เป้าหมาย:** จำแนกประเภทเพลงไทย
- **ลิงก์:** https://dl.acm.org/doi/full/10.1145/3722150.3722157

---

## 💻 **Source Code และ Tools**

### **Open Source Projects สำหรับ Music Transcription:**

#### **1. Spotify Basic Pitch** ⭐ (แนะนำ)
- **GitHub:** https://github.com/spotify/basic-pitch
- **ฟีเจอร์:**
  - Audio-to-MIDI converter แบบ lightweight
  - รองรับ pitch bend detection
  - ทำงานได้กับเครื่องดนตรีหลากหลายชนิด
- **Web Demo:** https://basicpitch.spotify.com/
- **เหมาะสำหรับ:** การแปลงเสียงพิณเป็น MIDI แบบ real-time

#### **2. Omnizart**
- **GitHub:** https://github.com/Music-and-Culture-Technology-Lab/omnizart
- **ฟีเจอร์:**
  - Python library สำหรับ automatic music transcription
  - รองรับ pitched instruments, vocal melody, chords, drums
  - มี pre-trained models หลายแบบ
- **เอกสาร:** https://music-and-culture-technology-lab.github.io/omnizart-doc/

#### **3. NeuralNote**
- **GitHub:** https://github.com/DamRsn/NeuralNote
- **ฟีเจอร์:** Audio plugin (VST3) สำหรับ transcription
- **รูปแบบ:** Standalone, VST3, Component

#### **4. โปรเจคอื่นๆ:**
- **wav2mid:** https://github.com/jsleep/wav2mid (Neural network for polyphonic)
- **automatic_music_transcription:** https://github.com/w4k2/automatic_music_transcription
- **AMT-Deep-Learning:** https://github.com/SboneloMdluli/Automatic-Music-Transcription-using-Deep-Learning

---

## 🎯 **Roadmap: การพัฒนาโมเดล AI**

### **Phase 1: Data Collection (1-2 เดือน)**
1. ดาวน์โหลดวิดีโอจาก YouTube (20-30 วิดีโอต่อลายพิณ)
2. แยกเสียงออกมาเป็นไฟล์ WAV คุณภาพสูง (44.1kHz, 16-bit)
3. รวบรวมโน๊ตดนตรีและแปลงเป็น MIDI
4. สร้าง metadata (ชื่อลายพิณ, ศิลปิน, tempo, key)

### **Phase 2: Data Preprocessing (2-4 สัปดาห์)**
1. ทำ Audio Cleaning (noise reduction, normalization)
2. ตัดเสียงให้เป็น segments (3-5 วินาทีต่อ segment)
3. ใช้ Basic Pitch แปลง audio เป็น MIDI เบื้องต้น
4. Manual verification และแก้ไข MIDI ให้ตรงกับโน๊ตจริง
5. Data Augmentation:
   - Time stretching (±10%)
   - Pitch shifting (±2 semitones)
   - Add background noise (10-20 dB SNR)

### **Phase 3: Model Development (2-3 เดือน)**

#### **Option 1: Transfer Learning (แนะนำสำหรับเริ่มต้น)**
```python
# ใช้ pre-trained model จาก Basic Pitch หรือ Omnizart
# Fine-tune ด้วยข้อมูลพิณ

from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# Load pre-trained model
model = load_model(ICASSP_2022_MODEL_PATH)

# Fine-tune with Phin dataset
model.fit(
    phin_audio_data,
    phin_midi_labels,
    epochs=50,
    batch_size=8,
    validation_split=0.2
)
```

#### **Option 2: Traditional Signal Processing (ตามงานวิจัย KMUTT)**
```python
# ใช้ EWMA + FFT + Peak Detection
# ไม่ต้องใช้ Deep Learning
# เหมาะสำหรับ real-time และ low-resource devices

import librosa
import numpy as np

def detect_notes_ewma(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Energy-based onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units='time'
    )
    
    # FFT-based pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    return onsets, pitches
```

#### **Option 3: Custom Deep Learning Model**
```python
# สร้าง CNN-RNN hybrid model
import tensorflow as tf

def build_phin_transcription_model(input_shape):
    model = tf.keras.Sequential([
        # CNN layers for feature extraction
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                               input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Reshape for RNN
        tf.keras.layers.Reshape((-1, 64)),
        
        # Bidirectional LSTM for temporal modeling
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)
        ),
        
        # Output layer (88 piano keys)
        tf.keras.layers.Dense(88, activation='sigmoid')
    ])
    
    return model
```

### **Phase 4: Evaluation (2-4 สัปดาห์)**
```python
# ใช้ mir_eval สำหรับประเมิน
import mir_eval

def evaluate_transcription(reference_midi, estimated_midi):
    # Onset detection metrics
    onset_precision, onset_recall, onset_f1 = \
        mir_eval.onset.f_measure(ref_onsets, est_onsets)
    
    # Pitch detection metrics
    pitch_precision, pitch_recall, pitch_f1 = \
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches
        )
    
    return {
        'onset_f1': onset_f1,
        'pitch_f1': pitch_f1
    }
```

### **Phase 5: Deployment (1-2 เดือน)**
- สร้าง Web App (Streamlit หรือ Flask)
- พัฒนา Mobile App (React Native + TensorFlow Lite)
- Export model เป็น ONNX สำหรับ cross-platform
- สร้าง API สำหรับนักพัฒนา

---

## 📊 **Datasets เพิ่มเติม (สำหรับ Benchmark)**

### **Public Datasets:**
1. **GigaMIDI Dataset**
   - ขนาด: 1.4 ล้าน MIDI files
   - ลิงก์: https://transactions.ismir.net/articles/10.5334/tismir.203
   - ใช้สำหรับ: Pre-training และ Transfer Learning

2. **Free Music Archive (FMA)**
   - ขนาด: 106,574 tracks
   - GitHub: https://github.com/mdeff/fma
   - ใช้สำหรับ: Audio Classification

3. **MAESTRO Dataset**
   - เสียงเปียโนคุณภาพสูงพร้อม MIDI
   - ลิงก์: https://magenta.tensorflow.org/datasets/maestro
   - ใช้สำหรับ: Transfer Learning

---

## 🛠️ **Python Libraries ที่จำเป็น**

### **สำหรับ Audio Processing:**
```bash
pip install librosa
pip install soundfile
pip install pydub
pip install scipy
pip install numpy
```

### **สำหรับ MIDI Handling:**
```bash
pip install pretty_midi
pip install mido
pip install music21
```

### **สำหรับ Machine Learning:**
```bash
pip install tensorflow
pip install torch torchvision
pip install basic-pitch
pip install omnizart
```

### **สำหรับ Evaluation:**
```bash
pip install mir_eval
pip install jams
```

### **สำหรับ Data Acquisition:**
```bash
pip install yt-dlp
pip install pytube
pip install requests beautifulsoup4
```

---

## ⚠️ **ข้อควรระวัง**

### **1. ลิขสิทธิ์:**
- วิดีโอบน YouTube อาจมีลิขสิทธิ์
- ใช้เพื่อการศึกษาและวิจัยเท่านั้น
- ไม่ควร distribute หรือใช้ในเชิงพาณิชย์โดยไม่ได้รับอนุญาต

### **2. คุณภาพข้อมูล:**
- ต้องกรองเสียงรบกวน (background noise)
- ควรมีข้อมูลจากหลายแหล่งเพื่อความหลากหลาย
- ต้องมี manual verification สำหรับ ground truth

### **3. Cultural Sensitivity:**
- ดนตรีไทยมีระบบเสียง 7 tones ต่อ octave
- ต้องเข้าใจ microtonal nuances
- อย่าบังคับใช้ Western music theory โดยตรง

---

## 📈 **Expected Results**

### **Target Metrics:**
- **Onset Detection F1-score:** > 95%
- **Pitch Detection F1-score:** > 90%
- **Note Duration Accuracy:** > 85%
- **Overall Transcription Accuracy:** > 85%

### **Comparison:**
- งานวิจัย KMUTT (ระนาด): 98.54% onset, 97.34% pitch
- Spotify Basic Pitch (ทั่วไป): ~85-90% accuracy
- **เป้าหมาย (พิณ):** 90-95% accuracy

---

## 🚀 **Next Steps**

### **ทันที (1 สัปดาห์):**
1. ✅ ตั้งค่าโครงสร้างโฟลเดอร์ (เสร็จแล้ว!)
2. ⏳ ดาวน์โหลดวิดีโอ YouTube ที่ระบุไว้
3. ⏳ เริ่มแยกไฟล์เสียง (WAV format)

### **ระยะสั้น (1 เดือน):**
1. รวบรวมโน๊ตดนตรีจากแหล่งต่างๆ
2. ทดลองใช้ Basic Pitch แปลงเสียงพิณ
3. สร้าง proof-of-concept model

### **ระยะกลาง (3 เดือน):**
1. พัฒนา custom model สำหรับพิณโดยเฉพาะ
2. Fine-tune ด้วย augmented data
3. ทำ cross-validation และ optimize hyperparameters

### **ระยะยาว (6+ เดือน):**
1. สร้าง production-ready application
2. เผยแพร่เป็น open source
3. เขียนงานวิจัยและ publish

---

## 🤝 **Contributing**

หากคุณมีข้อมูลเพิ่มเติม เช่น:
- 🎵 โน๊ตลายพิณที่หายาก
- 🎬 วิดีโอสอนพิณคุณภาพสูง
- 💻 Source code ที่เกี่ยวข้อง
- 📚 งานวิจัยดนตรีไทย

กรุณาติดต่อหรือเพิ่มข้อมูลลงใน dataset นี้!

---

## 📞 **Contact & References**

### **แหล่งข้อมูลหลัก:**
- โน๊ตลายพิณ: https://guitar285.wordpress.com/category/โน๊ตลายพิณ/
- งานวิจัย KMUTT: https://inc.kmutt.ac.th/download/capstone_design_projects/2567/10.pdf
- Google Magenta Transcultural: https://magenta.withgoogle.com/transcultural
- Spotify Basic Pitch: https://github.com/spotify/basic-pitch

### **Community:**
- Reddit: r/MusicInformationRetrieval
- Discord: Music & Audio Research Community
- GitHub: awesome-audio-ml

---

**สร้างด้วย ❤️ สำหรับการอนุรักษ์และพัฒนาดนตรีไทย**
**Last Updated: 25 พฤศจิกายน 2025**
