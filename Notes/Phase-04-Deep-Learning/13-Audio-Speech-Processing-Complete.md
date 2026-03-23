# 11.6 Audio & Speech Processing (Whisper, Wav2Vec)

## 🎯 Quick Overview
- **Audio Representation**: Waveforms, Spectrograms, and Mel-Frequency Cepstral Coefficients (MFCCs)
- **Feature Extraction**: Converting raw sound into "images" or "tokens" for ML models
- **Speech-to-Text (ASR)**: OpenAI Whisper and Wav2Vec 2.0
- **Speech Synthesis (TTS)**: Generating realistic human voices (Tacotron, VALL-E)
- **Foundation for**: Voice assistants, Automated transcription, and Real-time translation

---

## 1. Representing Audio

Unlike images (pixels) or text (tokens), audio is a continuous wave.

### 1.1 Time Domain (Waveforms)
A graph of amplitude vs. time. It is high-dimensional and contains a lot of noise.

### 1.2 Frequency Domain (Spectrograms)
Using the **Fourier Transform**, we convert waveforms into a heat-map of frequencies.
- **Mel Spectrogram**: Scales frequencies to match how humans perceive pitch (non-linear).
- **MFCCs**: A compressed representation of the Mel Spectrogram, widely used in traditional ASR.

---

## 2. Revolutionary Speech Models

### 2.1 Wav2Vec 2.0 (Self-Supervised)
Learns from raw audio without transcripts. 
- It uses a **Contrastive Task**: the model must predict the correct "latent" representation for a masked segment of audio among several distractors.

### 2.2 OpenAI Whisper (Robust ASR)
Trained on 680,000 hours of multilingual and multitask supervised data.
- **Capabilities**: Transcription (Speech-to-Text), Translation (X-to-English), and Language Identification.
- **Architecture**: A standard Transformer Encoder-Decoder.

---

## 3. Text-to-Speech (TTS)

Generative models that convert text into natural-sounding speech.
- **Tacotron**: Predicts Mel Spectrograms from text.
- **WaveNet/Vocoders**: Converts those spectrograms into the final raw audio wave.
- **VALL-E**: A neural codec language model that can clone a voice with just 3 seconds of audio.

---

## 💻 Python Code Examples

### 1. Simple Audio Visualization (Librosa)
```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load audio
y, sr = librosa.load("audio.wav")

# 2. Compute Mel Spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

# 3. Plot
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()
```

### 2. Transcription with OpenAI Whisper
```python
import whisper

# 1. Load pre-trained model (base, small, medium, large)
model = whisper.load_model("base")

# 2. Transcribe
result = model.transcribe("interview.mp3")

# 3. Print the text
print(result["text"])
```

---

## 📊 Summary Table

| Model | Architecture | Training Style | Best For |
|-------|--------------|----------------|----------|
| **Wav2Vec 2.0** | CNN + Trans. | Self-Supervised | Low-resource languages |
| **Whisper** | Trans. Enc-Dec | Supervised (Large) | Robust real-world ASR |
| **Tacotron 2**| RNN + Attention| Supervised | High-quality TTS |
| **HuBERT** | Transformer | Self-Supervised | General audio features |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Whisper | Live captioning for YouTube/Meetings |
| Speaker Embeddings | Voice biometrics (bank authentication) |
| Audio Augmentation | Improving noise cancellation in earbuds |
| Vocoders | High-fidelity music production AI |

---

## ❓ Quick Check Questions

1. What is the benefit of a Mel Spectrogram over a standard Waveform?
2. How does "Self-Supervised Learning" help in speech processing?
3. What makes Whisper more robust to background noise than previous models?
4. What is a "Vocoder" in the context of TTS?
5. How do humans perceive pitch differently from a linear frequency scale?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. A **Mel Spectrogram** reduces the dimensionality of the data and highlights the patterns that are most important for distinguishing human speech, making it much easier for neural networks to learn from compared to raw amplitudes.
2. Self-supervised learning (like in Wav2Vec) allows models to learn the "structure" of human speech from millions of hours of **unlabeled** audio. This pre-training makes the model much more effective when later fine-tuned on a small amount of expensive labeled transcripts.
3. **Whisper** was trained on a massive and diverse dataset (680k hours) that included audio with heavy accents, background noise, and technical jargon. This makes it far more "zero-shot" robust than models trained on clean laboratory datasets.
4. A **Vocoder** is the final component in a TTS system. While the first part of the model generates a spectrogram (an image of sound), the vocoder translates that spectrogram back into the actual continuous pressure waves (audio) we can hear.
5. Humans perceive pitch **logarithmically**. We are much better at distinguishing between low frequencies (e.g., 100Hz vs 200Hz) than between very high frequencies (e.g., 10,000Hz vs 10,100Hz). The Mel scale accounts for this by spacing out lower frequencies.

</details>

---

**Status:** ✅ Complete
**Next:** Phase 4 Practice Problems (Standardized 5-Level Structure)
