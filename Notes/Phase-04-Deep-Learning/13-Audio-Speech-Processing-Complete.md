# 11.6 Advanced Audio & Speech Processing: Beyond Waveforms

## 🎯 Quick Overview
- **Fourier Transform Math**: Converting time-domain waves to frequency-domain spectra
- **Mel-Filterbanks**: How the Mel-scale mimics human psychoacoustics
- **ASR Loss Functions**: CTC (Connectionist Temporal Classification) and Cross-Entropy
- **Modern Architectures**: Whisper's Multi-task conditioning and Wav2Vec 2.0 self-supervision
- **Foundation for**: Voice cloning, Real-time translation, and Acoustic anomaly detection

---

## 1. The Physics of Digital Audio

### 1.1 The Fourier Transform (STFT)
Audio is originally a continuous waveform. We discretize it via sampling (e.g., 16kHz). To see the frequencies, we use the **Short-Time Fourier Transform (STFT)**.
- **Math**: $X(m, \omega) = \sum_{n=-\infty}^{\infty} x(n) w(n - m) e^{-j\omega n}$
- **Intuition**: We slide a window ($w$) over the audio and perform a Fourier Transform on each segment to see which frequencies are present at that exact moment.

### 1.2 The Mel-Scale
Humans don't hear frequencies linearly. We are much better at distinguishing low-pitched sounds.
- **Mel Conversion**: $M(f) = 2595 \log_{10}(1 + f/700)$
- **Log-Mel Spectrogram**: Taking the log of the Mel-scaled frequencies. This is the "standard" input for most speech models like Whisper.

---

## 2. Automatic Speech Recognition (ASR) Math

The hardest part of ASR is the **alignment** problem: 1 second of audio might correspond to 3 characters or 10 characters.

### 2.1 CTC Loss (Connectionist Temporal Classification)
Allows the model to predict tokens at every time step and then "collapses" them.
- **The Blank Token ($\epsilon$)**: A special character used to separate repeating letters (e.g., "hello" vs "helo").
- **Collapse Rule**: Remove all $\epsilon$ tokens and merge identical adjacent characters.
- **Benefit**: No need for manual word-level alignment in the training data.

---

## 3. Foundation Models for Speech

### 3.1 Wav2Vec 2.0 (Self-Supervised)
Learns by solving a **Contrastive Task**.
1.  **Feature Encoder**: CNN extracts latent representations from raw audio.
2.  **Quantization**: Maps these to a discrete codebook.
3.  **Context Network**: A Transformer predicts masked portions of the latent space.

### 3.2 OpenAI Whisper (Scaling Supervision)
Whisper proved that **Massive Multitask Supervision** (680k hours) beats complex self-supervised pre-training for real-world robustness.
- **Conditioning**: The decoder receives special tokens telling it to: [TRANSCRIBE], [TRANSLATE], or [DETECT_LANGUAGE].

---

## 💻 Professional Implementation

### 1. STFT and Spectrogram Generation (Librosa)
```python
import librosa
import numpy as np

# 1. Load audio
y, sr = librosa.load(librosa.ex('trumpet'))

# 2. Compute STFT
# n_fft: window size, hop_length: stride
D = librosa.stft(y, n_fft=2048, hop_length=512)
magnitude, phase = librosa.magphase(D)

# 3. Apply Mel Filterbank
mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)
mel_spectrogram = np.dot(mel_basis, magnitude)

# 4. Log Scale
log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram)
```

### 2. CTC Decoding Logic (Conceptual)
```python
def ctc_decode(sequence):
    # 1. Remove adjacent duplicates
    res = []
    for i, char in enumerate(sequence):
        if i == 0 or char != sequence[i-1]:
            res.append(char)
    
    # 2. Remove blank tokens
    return "".join([c for c in res if c != "<BLANK>"])

# Input:  [H, H, <B>, E, E, L, L, <B>, L, L, O, O]
# Result: [H, <B>, E, L, <B>, L, O] -> HELLO
```

---

## 📊 Summary Comparison

| Feature | Wav2Vec 2.0 | Whisper | HuBERT |
| :--- | :--- | :--- | :--- |
| **Data Type** | Unlabeled Audio | **Labeled Transcripts**| Unlabeled Audio |
| **Architecture** | CNN + Transformer | Transformer (Enc-Dec)| Transformer |
| **Task** | Contrastive | Multi-task Generative| Clustering/Prediction|
| **Robustness** | Moderate | **Extreme** | Moderate |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **VAD (Voice Activity Detection)**| Automatically cutting out silence to save compute before sending audio to an LLM. |
| **Speaker Diarization** | Identifying "Who spoke when" in a meeting with multiple participants. |
| **Lip-Sync (Wav2Lip)** | Synchronizing a video of a face to match a generated audio file. |
| **Emotion Recognition**| Analyzing the *prosody* (tone/pitch) of a customer call to detect anger or satisfaction. |

---

## ❓ Quick Check Questions

1. Why is the Mel scale non-linear?
2. What is the "Alignment Problem" in speech recognition, and how does CTC solve it?
3. How does Whisper handle translation from French to English without a separate model?
4. What is the purpose of the "Feature Encoder" (CNN) in Wav2Vec 2.0?
5. Explain the difference between "Sample Rate" and "Bit Depth."

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. Because human hearing is non-linear. We can easily distinguish between 100Hz and 200Hz, but 10,000Hz and 10,100Hz sound nearly identical to us. The **Mel scale** warps the frequency axis to give more importance to the lower frequencies where speech information is concentrated.
2. The **Alignment Problem** is that we don't know which audio frames correspond to which characters in the transcript. **CTC** solves this by allowing the model to predict a "blank" token when no character is being said and by mathematically summing over all possible paths that could lead to the correct transcript.
3. Whisper is a **Multitask model**. During training, it was shown pairs of [French Audio] and [English Text] with a special `<|translate|>` token. During inference, by providing this token, the model knows to activate its internal translation mapping rather than just transcribing.
4. Raw audio is extremely high-dimensional (16,000 values per second). The **CNN feature encoder** compresses this raw wave into a more compact latent representation, reducing the sequence length before it is passed to the Transformer.
5. **Sample Rate** is how many times per second we measure the air pressure (determines max frequency). **Bit Depth** is how precisely we measure each sample (determines dynamic range/signal-to-noise ratio).

</details>

---

## 📚 Recommended Resources
- **Paper**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- **Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356)
- **Course**: [HuggingFace Audio Course](https://huggingface.co/learn/audio-course/en/index).

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** Phase 4 Elite Practice Problems (Standardized 5-Level Structure)
