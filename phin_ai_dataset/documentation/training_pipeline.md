# 🤖 Pipeline การเทรนโมเดล AI สำหรับพิณอีสาน

**เอกสารนี้อธิบายขั้นตอนการเทรนโมเดลอย่างละเอียด**

---

## 🎯 **ภาพรวม**

โมเดล AI สำหรับพิณอีสานต้องทำหน้าที่:
1. **Onset Detection** - ตรวจจับจังหวะเริ่มต้นของเสียง
2. **Pitch Detection** - ระบุโน๊ตเสียง (ในระบบ 7-tone ของดนตรีไทย)
3. **Duration Estimation** - ประมาณความยาวของแต่ละโน๊ต
4. **MIDI Export** - แปลงผลลัพธ์เป็นไฟล์ MIDI

---

## 📊 **Pipeline แบบเต็ม**

```
┌─────────────────┐
│ Raw Audio (WAV) │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Audio Preprocessing │
│ - Normalization     │
│ - Noise Reduction   │
│ - Segmentation      │
└────────┬────────────┘
         │
         ▼
┌──────────────────────┐
│ Feature Extraction   │
│ - Mel Spectrogram    │
│ - CQT                │
│ - MFCC               │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Model Inference      │
│ - CNN (Spatial)      │
│ - RNN (Temporal)     │
│ - Attention          │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Post-processing      │
│ - Peak Picking       │
│ - Duration Smoothing │
│ - Threshold Tuning   │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ MIDI Generation      │
│ - Note events        │
│ - Velocity           │
│ - Tempo              │
└──────────────────────┘
```

---

## 📚 **Phase 1: Data Preparation**

### **1.1 Directory Structure**
```
phin_ai_dataset/
├── raw_audio/
│   ├── lai_mahoree/
│   │   ├── steve_01.wav
│   │   ├── steve_02.wav
│   │   └── ...
│   ├── lai_nok_sai/
│   └── lai_tet_khong/
├── processed_audio/
│   ├── train/
│   ├── val/
│   └── test/
├── ground_truth/
│   ├── midi/
│   └── annotations/
└── features/
    ├── spectrograms/
    └── embeddings/
```

### **1.2 Audio Processing Script**

```python
# audio_preprocessing.py
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

class AudioPreprocessor:
    def __init__(self, sr=22050, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def load_and_normalize(self, audio_path):
        """โหลดและทำ normalization"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Normalize to [-1, 1]
        y = librosa.util.normalize(y)
        
        return y, sr
    
    def remove_noise(self, y, noise_threshold=0.02):
        """ลด noise ด้วย spectral gating"""
        # Compute STFT
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = librosa.magphase(D)
        
        # Estimate noise floor (first 1 second)
        noise_frames = int(self.sr / self.hop_length)
        noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply noise gate
        mask = magnitude > (noise_profile * noise_threshold)
        magnitude_clean = magnitude * mask
        
        # Reconstruct
        D_clean = magnitude_clean * phase
        y_clean = librosa.istft(D_clean, hop_length=self.hop_length)
        
        return y_clean
    
    def segment_audio(self, y, segment_length=5.0, overlap=0.5):
        """ตัดเสียงเป็น segments"""
        segment_samples = int(segment_length * self.sr)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        for start in range(0, len(y) - segment_samples, hop_samples):
            segment = y[start:start + segment_samples]
            segments.append(segment)
        
        return segments
    
    def extract_mel_spectrogram(self, y):
        """Extract Mel Spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=128
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_cqt(self, y):
        """Extract Constant-Q Transform (ดีกับเครื่องดนตรี)"""
        cqt = librosa.cqt(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=84,  # 7 octaves
            bins_per_octave=12
        )
        
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        return cqt_db
    
    def process_file(self, input_path, output_dir):
        """Process single audio file"""
        # Load
        y, sr = self.load_and_normalize(input_path)
        
        # Denoise
        y_clean = self.remove_noise(y)
        
        # Segment
        segments = self.segment_audio(y_clean)
        
        # Save segments
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem = Path(input_path).stem
        for i, segment in enumerate(segments):
            output_path = output_dir / f"{stem}_seg{i:03d}.wav"
            sf.write(output_path, segment, sr)
        
        return len(segments)

# Usage
if __name__ == "__main__":
    preprocessor = AudioPreprocessor(sr=22050)
    
    input_file = "raw_audio/lai_mahoree/steve_01.wav"
    output_dir = "processed_audio/train/lai_mahoree"
    
    n_segments = preprocessor.process_file(input_file, output_dir)
    print(f"Created {n_segments} segments")
```

---

## 🔬 **Phase 2: Feature Engineering**

### **2.1 Feature Extractor**

```python
# feature_extraction.py
import librosa
import numpy as np

class PhinFeatureExtractor:
    """
    Feature extractor สำหรับพิณอีสาน
    ออกแบบให้เข้าใจระบบ 7-tone
    """
    
    def __init__(self, sr=22050, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length
        
        # Thai music uses 7-tone scale
        # ปรับ bins_per_octave = 7 แทน 12
        self.bins_per_octave = 7
    
    def extract_all_features(self, y):
        """Extract ฟีเจอร์ทั้งหมด"""
        features = {}
        
        # 1. Mel Spectrogram
        features['mel'] = self._extract_mel(y)
        
        # 2. CQT (Constant-Q Transform)
        features['cqt'] = self._extract_cqt(y)
        
        # 3. Chroma (สำหรับ pitch)
        features['chroma'] = self._extract_chroma(y)
        
        # 4. Onset Strength
        features['onset'] = self._extract_onset(y)
        
        # 5. Spectral Features
        features['spectral'] = self._extract_spectral(y)
        
        return features
    
    def _extract_mel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=128, hop_length=self.hop_length
        )
        return librosa.power_to_db(mel, ref=np.max)
    
    def _extract_cqt(self, y):
        """CQT with 7 bins per octave (Thai scale)"""
        cqt = librosa.cqt(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            bins_per_octave=self.bins_per_octave,
            n_bins=self.bins_per_octave * 6  # 6 octaves
        )
        return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    def _extract_chroma(self, y):
        """Chroma สำหรับ pitch tracking"""
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            bins_per_octave=self.bins_per_octave
        )
        return chroma
    
    def _extract_onset(self, y):
        """Onset strength envelope"""
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        return onset_env
    
    def _extract_spectral(self, y):
        """Spectral features"""
        spectral = {}
        
        # Spectral centroid
        spectral['centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        
        # Spectral rolloff
        spectral['rolloff'] = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        
        # Zero crossing rate
        spectral['zcr'] = librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )
        
        return spectral
```

---

## 🧠 **Phase 3: Model Architecture**

### **3.1 CNN-RNN Hybrid Model**

```python
# model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PhinTranscriptionModel(keras.Model):
    """
    Hybrid CNN-RNN model สำหรับ transcription พิณอีสาน
    
    Architecture:
    1. CNN layers - Extract spatial features from spectrogram
    2. Bidirectional LSTM - Model temporal dependencies
    3. Attention - Focus on important frames
    4. Output heads - Separate heads for onset and pitch
    """
    
    def __init__(self, n_pitches=42, **kwargs):
        super().__init__(**kwargs)
        
        self.n_pitches = n_pitches  # 6 octaves * 7 tones
        
        # CNN Encoder
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.bn3 = layers.BatchNormalization()
        
        # Reshape for RNN
        self.reshape = layers.Reshape((-1, 128))
        
        # Bidirectional LSTM
        self.lstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3)
        )
        self.lstm2 = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.3)
        )
        
        # Attention mechanism
        self.attention = layers.Attention()
        
        # Output heads
        self.onset_head = layers.Dense(1, activation='sigmoid', name='onset')
        self.pitch_head = layers.Dense(
            self.n_pitches, activation='sigmoid', name='pitch'
        )
        self.velocity_head = layers.Dense(
            1, activation='sigmoid', name='velocity'
        )
    
    def call(self, inputs, training=False):
        # CNN feature extraction
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x, training=training)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x, training=training)
        
        # Reshape for RNN
        x = self.reshape(x)
        
        # LSTM temporal modeling
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Attention
        attention_output = self.attention([x, x])
        
        # Output predictions
        onset = self.onset_head(attention_output)
        pitch = self.pitch_head(attention_output)
        velocity = self.velocity_head(attention_output)
        
        return {
            'onset': onset,
            'pitch': pitch,
            'velocity': velocity
        }

# Build model
def build_phin_model(input_shape=(128, None, 1), n_pitches=42):
    """Build and compile model"""
    model = PhinTranscriptionModel(n_pitches=n_pitches)
    
    # Multi-task loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'onset': 'binary_crossentropy',
            'pitch': 'binary_crossentropy',
            'velocity': 'mse'
        },
        loss_weights={
            'onset': 1.0,
            'pitch': 1.0,
            'velocity': 0.5
        },
        metrics={
            'onset': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            'pitch': ['accuracy'],
            'velocity': ['mae']
        }
    )
    
    return model
```

### **3.2 Alternative: EWMA-based Model (ตามงานวิจัย KMUTT)**

```python
# ewma_model.py
import numpy as np
import librosa

class EWMATranscriber:
    """
    Energy-based Windowed Moving Average method
    จากงานวิจัย KMUTT (98.54% F1-score)
    
    ไม่ต้องใช้ Deep Learning!
    """
    
    def __init__(self, sr=22050, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length
    
    def detect_onsets(self, y, alpha=0.8, threshold=0.3):
        """
        Onset detection ด้วย EWMA
        
        alpha: smoothing factor (0.7-0.9)
        threshold: detection threshold (0.2-0.4)
        """
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        
        # Apply EWMA smoothing
        ewma = np.zeros_like(onset_env)
        ewma[0] = onset_env[0]
        
        for i in range(1, len(onset_env)):
            ewma[i] = alpha * ewma[i-1] + (1 - alpha) * onset_env[i]
        
        # Peak detection
        onsets = librosa.onset.onset_detect(
            onset_envelope=ewma,
            sr=self.sr,
            hop_length=self.hop_length,
            backtrack=True,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=threshold,
            wait=10
        )
        
        # Convert to time
        onset_times = librosa.frames_to_time(
            onsets, sr=self.sr, hop_length=self.hop_length
        )
        
        return onset_times
    
    def detect_pitches(self, y, onset_times, fmin=100, fmax=2000):
        """
        Pitch detection ด้วย FFT + Peak Picking
        """
        pitches = []
        
        for onset_time in onset_times:
            # Extract frame around onset
            frame_start = int(onset_time * self.sr)
            frame_end = frame_start + self.hop_length
            frame = y[frame_start:frame_end]
            
            # Compute FFT
            fft = np.fft.rfft(frame)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(frame), 1/self.sr)
            
            # Find peak in frequency range
            mask = (freqs >= fmin) & (freqs <= fmax)
            peak_idx = np.argmax(magnitude[mask])
            peak_freq = freqs[mask][peak_idx]
            
            # Convert to MIDI note
            midi_note = librosa.hz_to_midi(peak_freq)
            pitches.append(midi_note)
        
        return np.array(pitches)
    
    def transcribe(self, audio_path):
        """Complete transcription"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Detect onsets
        onset_times = self.detect_onsets(y)
        
        # Detect pitches
        pitches = self.detect_pitches(y, onset_times)
        
        # Estimate durations
        durations = np.diff(onset_times, append=onset_times[-1] + 0.5)
        
        return {
            'onset_times': onset_times,
            'pitches': pitches,
            'durations': durations
        }
```

---

## 🎓 **Phase 4: Training Loop**

```python
# train.py
import tensorflow as tf
from pathlib import Path
import numpy as np

class PhinTrainer:
    def __init__(self, model, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Callbacks
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1
            )
        ]
    
    def train(self, epochs=100):
        """Train model"""
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_dataset):
        """Evaluate model"""
        results = self.model.evaluate(test_dataset, verbose=1)
        return results

# Data augmentation
def augment_audio(y, sr):
    """Audio augmentation"""
    augmentations = []
    
    # Original
    augmentations.append(y)
    
    # Time stretch
    y_stretch = librosa.effects.time_stretch(y, rate=0.9)
    augmentations.append(y_stretch)
    
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)
    augmentations.append(y_stretch)
    
    # Pitch shift
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    augmentations.append(y_shift)
    
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
    augmentations.append(y_shift)
    
    # Add noise
    noise = np.random.normal(0, 0.005, y.shape)
    y_noisy = y + noise
    augmentations.append(y_noisy)
    
    return augmentations
```

---

## 📈 **Phase 5: Evaluation**

```python
# evaluation.py
import mir_eval
import numpy as np

def evaluate_transcription(ref_midi, est_midi):
    """
    Evaluate transcription using mir_eval
    """
    # Load MIDI files
    ref_intervals, ref_pitches = load_midi_notes(ref_midi)
    est_intervals, est_pitches = load_midi_notes(est_midi)
    
    # Onset detection metrics
    onset_precision, onset_recall, onset_f1 = mir_eval.onset.f_measure(
        ref_intervals[:, 0],
        est_intervals[:, 0],
        window=0.05  # 50ms tolerance
    )
    
    # Pitch detection metrics
    pitch_precision, pitch_recall, pitch_f1 = \
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=0.05,
            pitch_tolerance=50.0  # 50 cents
        )
    
    # Note-level metrics
    note_precision, note_recall, note_f1 = \
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches
        )
    
    return {
        'onset_precision': onset_precision,
        'onset_recall': onset_recall,
        'onset_f1': onset_f1,
        'pitch_precision': pitch_precision,
        'pitch_recall': pitch_recall,
        'pitch_f1': pitch_f1,
        'note_precision': note_precision,
        'note_recall': note_recall,
        'note_f1': note_f1
    }

def load_midi_notes(midi_path):
    """Load notes from MIDI file"""
    import pretty_midi
    
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    intervals = []
    pitches = []
    
    for instrument in midi.instruments:
        for note in instrument.notes:
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)
    
    return np.array(intervals), np.array(pitches)
```

---

## 🎯 **Expected Performance**

### **Target Metrics:**
- **Onset Detection F1:** > 95%
- **Pitch Detection F1:** > 90%
- **Note-level F1:** > 85%

### **Baseline (KMUTT ระนาด):**
- Onset F1: **98.54%**
- Pitch F1: **97.34%**

### **Spotify Basic Pitch (ทั่วไป):**
- Overall accuracy: **85-90%**

---

**Next:** Deploy model และสร้าง web interface!
