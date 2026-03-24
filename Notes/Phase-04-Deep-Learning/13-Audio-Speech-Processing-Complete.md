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

## 💻 Professional Implementation: End-to-End Speech Pipeline

This implementation provides a robust interface for transcribing audio using OpenAI Whisper, including automatic language detection and timestamp processing.

```python
import whisper
import torch
from typing import Dict, Any
import os

class SpeechProcessor:
    def __init__(self, model_size: str = "base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. Load optimized model
        self.model = whisper.load_model(model_size).to(self.device)
        print(f"Whisper {model_size} loaded on {self.device}")

    def transcribe(self, audio_path: str, task: str = "transcribe") -> Dict[str, Any]:
        """
        Transcribe or Translate audio.
        task: 'transcribe' or 'translate' (to English)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing: {audio_path}")

        # 2. Run Inference with FP16 if on GPU
        result = self.model.transcribe(
            audio_path, 
            task=task,
            fp16=(self.device == "cuda")
        )
        
        return {
            "text": result["text"],
            "language": result.get("language"),
            "segments": result.get("segments") # Includes timestamps
        }

# --- Usage Example ---
# processor = SpeechProcessor("medium")
# output = processor.transcribe("meeting_recording.mp3")
# print(f"Detected Language: {output['language']}")
# print(f"Full Transcript: {output['text']}")

# # Output segmented timestamps
# for segment in output['segments']:
#     print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s]: {segment['text']}")
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

## 4. Neural Audio Synthesis

### 4.1 WaveNet: Autoregressive Audio Generation
WaveNet generates raw audio one sample at a time using dilated convolutions.

**Key Innovations**:
1.  **Dilated causal convolutions**: Large receptive field without losing resolution
2.  **Gated activations**: $\tanh \odot \sigma$ for better gradient flow
3.  **Residual connections**: Enables deep architectures

**Architecture**:
```
Input → Dilated Conv (d=1) → Gated Activation → Residual → Output
              ↓
        Dilated Conv (d=2) → Gated Activation → Residual
              ↓
        Dilated Conv (d=4) → Gated Activation → Residual
              ↓
        Dilated Conv (d=8) → Gated Activation → Residual
```

**Dilated Convolution Math**:
$$ y[t] = \sum_{k=0}^{K-1} w[k] \cdot x[t - d \cdot k] $$

Where $d$ is the dilation rate.

```python
class WaveNetBlock(nn.Module):
    def __init__(self, hidden_dim, dilation, kernel_size=2):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Dilated convolution
        self.conv = nn.Conv1d(
            hidden_dim, 
            hidden_dim * 2,  # For gated activation
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        # Output projections
        self.out_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.skip_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
        # Residual
        self.residual = nn.Conv1d(hidden_dim, hidden_dim, 1)
    
    def forward(self, x):
        # Dilated convolution
        conv_out = self.conv(x)
        
        # Split for gated activation
        gate, filter = conv_out.chunk(2, dim=1)
        gated = torch.tanh(gate) * torch.sigmoid(filter)
        
        # Output and skip connections
        out = self.out_conv(gated)
        skip = self.skip_conv(gated)
        
        # Residual connection
        residual = self.residual(x)
        x = residual + out
        
        return x, skip


class WaveNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, num_layers=30):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # Stack of dilated blocks with increasing dilation
        self.blocks = nn.ModuleList([
            WaveNetBlock(hidden_dim, dilation=2**(i % 10))
            for i in range(num_layers)
        ])
        
        # Output
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)
        
        # Sum skip connections
        skips = torch.stack(skips).sum(dim=0)
        
        return self.out(skips)
```

---

### 4.2 WaveGlow: Flow-Based Synthesis
WaveGlow uses normalizing flows for faster-than-realtime synthesis.

**Key Idea**: Transform simple noise distribution to audio through invertible mappings.

```python
class WaveGlowBlock(nn.Module):
    """Invertible block for audio synthesis."""
    
    def __init__(self, hidden_dim, kernel_size=3):
        super().__init__()
        
        # Invertible 1x1 convolution
        self.W = nn.Parameter(torch.eye(hidden_dim).unsqueeze(0))
        
        # Coupling layers
        self.conv = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size, padding=kernel_size//2)
        self.scale = nn.Conv1d(hidden_dim // 2, hidden_dim // 2, 1)
        self.translate = nn.Conv1d(hidden_dim // 2, hidden_dim // 2, 1)
    
    def forward(self, x, reverse=False):
        # Split into two halves
        x1, x2 = x.chunk(2, dim=1)
        
        if not reverse:
            # Forward: (x1, x2) → (y1, y2)
            h = self.conv(x1)
            scale = torch.sigmoid(self.scale(h))
            translate = self.translate(h)
            
            y1 = x1
            y2 = x2 * scale + translate
            
            # Apply invertible 1x1 conv
            y = torch.cat([y1, y2], dim=1)
            y = torch.matmul(self.W, y.transpose(1, 2)).transpose(1, 2)
            
            # Log determinant for likelihood
            log_det = torch.log(torch.abs(torch.det(self.W[0]))) * y1.shape[2]
            
            return y, log_det
        else:
            # Reverse: (y1, y2) → (x1, x2)
            y = torch.matmul(x, self.W[0].inverse())
            y1, y2 = y.chunk(2, dim=1)
            
            h = self.conv(y1)
            scale = torch.sigmoid(self.scale(h))
            translate = self.translate(h)
            
            x1 = y1
            x2 = (y2 - translate) / scale
            
            return torch.cat([x1, x2], dim=1)
    
    def log_prob(self, x):
        """Compute log probability for training."""
        z, log_det = self.forward(x)
        
        # Prior is standard normal
        log_prob = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * 3.14159)))
        log_prob = log_prob.sum() + log_det
        
        return log_prob
```

---

### 4.3 Jukebox: Music Generation with VQ-VAE
OpenAI's Jukebox generates music with lyrics and style conditioning.

**Architecture**:
1.  **VQ-VAE Encoder**: Compress audio to discrete codes
2.  **Transformer**: Predict codes autoregressively
3.  **VQ-VAE Decoder**: Reconstruct audio from codes

```python
class VQVAE(nn.Module):
    """Vector Quantized VAE for audio compression."""
    
    def __init__(self, input_dim=64, hidden_dim=128, num_codes=512, code_dim=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, code_dim, 3, padding=1)
        )
        
        # Codebook
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.num_codes = num_codes
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(code_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim, 3, stride=2, padding=1, output_padding=1)
        )
    
    def quantize(self, z):
        """Vector quantization."""
        # z: [batch, code_dim, time]
        z = z.permute(0, 2, 1)  # [batch, time, code_dim]
        z_flat = z.reshape(-1, z.shape[-1])
        
        # Find nearest codebook entry
        codebook = self.codebook.weight
        distances = torch.cdist(z_flat, codebook)
        indices = torch.argmin(distances, dim=1)
        
        # Replace with codebook vectors
        z_quantized = self.codebook(indices)
        z_quantized = z_quantized.reshape(z.shape)
        
        # Straight-through estimator
        z_quantized = z + (z_quantized - z).detach()
        
        return z_quantized.permute(0, 2, 1), indices
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, indices = self.quantize(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        # Compute losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        
        return x_recon, indices, commitment_loss + codebook_loss
```

---

## 5. Speech Enhancement

### 5.1 Demucs: Music Source Separation
Demucs separates music into vocals, drums, bass, and other instruments.

**Architecture**: Hybrid CNN-Transformer encoder-decoder.

```python
class Demucs(nn.Module):
    def __init__(self, sources=4, audio_channels=2, channels=64):
        super().__init__()
        self.sources = sources
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Conv1d(audio_channels, channels, 7, stride=4, padding=3),
            nn.Conv1d(channels, channels * 2, 7, stride=4, padding=3),
            nn.Conv1d(channels * 2, channels * 4, 7, stride=4, padding=3),
        ])
        
        # Transformer (LSTM for efficiency)
        self.lstm = nn.LSTM(channels * 4, channels * 4, num_layers=2, batch_first=True)
        
        # Decoder for each source
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose1d(channels * 4, channels * 2, 7, stride=4, padding=3),
                nn.ConvTranspose1d(channels * 2, channels, 7, stride=4, padding=3),
                nn.ConvTranspose1d(channels, audio_channels, 7, stride=4, padding=3),
            ]) for _ in range(sources)
        ])
    
    def forward(self, x):
        # Encode
        encoded = x
        for conv in self.encoder:
            encoded = F.relu(conv(encoded))
        
        # LSTM
        b, c, t = encoded.shape
        encoded = encoded.permute(0, 2, 1).reshape(b, t, c)
        encoded, _ = self.lstm(encoded)
        encoded = encoded.reshape(b, t, c).permute(0, 2, 1)
        
        # Decode for each source
        outputs = []
        for decoder in self.decoders:
            out = encoded
            for conv in decoder:
                out = F.relu(conv(out))
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  # [batch, sources, channels, time]


# Usage for source separation
demucs = Demucs(sources=4)
mix = torch.randn(1, 2, 44100 * 10)  # 10 seconds stereo
sources = demucs(mix)
# sources: [batch, 4 (vocals/drums/bass/other), channels, time]
```

---

### 5.2 Speech Enhancement with SEGAN
SEGAN (Speech Enhancement GAN) removes noise from speech.

```python
class SEGAN(nn.Module):
    """Speech Enhancement GAN."""
    
    def __init__(self):
        super().__init__()
        
        # Generator (Encoder-Decoder)
        self.generator = nn.Sequential(
            # Encoder
            nn.Conv1d(1, 32, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            
            # Bottleneck
            nn.Conv1d(128, 256, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            
            # Decoder
            nn.ConvTranspose1d(256, 128, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 7, stride=2, padding=3),
        )
    
    def forward(self, noisy_audio):
        clean_audio = self.generator(noisy_audio)
        return clean_audio


class SEGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3),  # [clean+noisy, audio]
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, clean, generated):
        # Concatenate clean and generated/noisy
        x = torch.cat([clean, generated], dim=1)
        return self.discriminator(x)
```

---

## 6. Music Generation

### 6.1 Music Transformer
Music Transformer generates piano music with long-term structure.

**Key Innovation**: Relative attention for capturing musical patterns.

```python
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Relative positional encoding
        self.max_len = 2048
        self.relative_pe = nn.Parameter(torch.randn(2 * self.max_len, d_model))
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2048)
            for _ in range(n_layers)
        ])
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x) * math.sqrt(x.shape[-1])
        
        # Add relative positional encoding
        seq_len = x.shape[1]
        for i in range(seq_len):
            x[:, i] += self.relative_pe[self.max_len + i]
        
        # Transformer
        for layer in self.encoder_layers:
            x = layer(x)
        
        return self.output(x)
    
    def generate(self, start_tokens, max_len=1000, temperature=1.0):
        """Autoregressive music generation."""
        self.eval()
        generated = start_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(generated)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Sample from top-k
                top_k = 5
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(top_probs[0], 1)
                next_token = top_indices[0, next_token]
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated
```

---

### 6.2 AudioLM: Language Modeling for Audio
AudioLM generates audio by discretizing into tokens.

**Tokenization**:
1.  **Coarse tokens** (semantic): From HuBERT, capture content
2.  **Fine tokens** (acoustic): From SoundStream, capture timbre/quality

```python
class AudioLMTokenizer(nn.Module):
    """Hierarchical audio tokenization."""
    
    def __init__(self, semantic_vocab=2048, acoustic_vocab=1024, acoustic_levels=3):
        super().__init__()
        self.semantic_vocab = semantic_vocab
        self.acoustic_vocab = acoustic_vocab
        self.acoustic_levels = acoustic_levels
        
        # Semantic tokenizer (from HuBERT)
        self.semantic_tokenizer = HubertTokenizer()
        
        # Acoustic tokenizer (from SoundStream)
        self.acoustic_tokenizers = nn.ModuleList([
            VectorQuantize(dim=256, codebook_size=acoustic_vocab)
            for _ in range(acoustic_levels)
        ])
    
    def tokenize(self, audio):
        """Convert audio to hierarchical tokens."""
        # Semantic tokens (coarse)
        semantic_tokens = self.semantic_tokenizer.tokenize(audio)
        
        # Acoustic tokens (fine)
        acoustic_tokens = []
        residual = audio
        for tokenizer in self.acoustic_tokenizers:
            quantized, tokens = tokenizer(residual)
            acoustic_tokens.append(tokens)
            residual = residual - quantized  # Residual for next level
        
        return semantic_tokens, acoustic_tokens
    
    def detokenize(self, semantic_tokens, acoustic_tokens):
        """Reconstruct audio from tokens."""
        # Start with semantic reconstruction
        audio = self.semantic_tokenizer.detokenize(semantic_tokens)
        
        # Add acoustic details
        for i, tokens in enumerate(acoustic_tokens):
            audio += self.acoustic_tokenizers[i].decode(tokens)
        
        return audio
```

---

## 7. Audio LLMs

### 7.1 AudioLlama: Unified Audio-Language Model
Combine audio and text processing in one model.

```python
class AudioLlama(nn.Module):
    def __init__(self, llm, audio_encoder, audio_projector):
        super().__init__()
        self.llm = llm
        self.audio_encoder = audio_encoder
        self.audio_projector = audio_projector
    
    def forward(self, input_ids, audio=None, audio_mask=None):
        """
        Process text and audio inputs.
        
        Args:
            input_ids: Text token IDs
            audio: Audio features (if present)
            audio_mask: Which tokens are audio placeholders
        """
        # Embed text
        inputs_embeds = self.llm.embed_tokens(input_ids)
        
        # Process audio if present
        if audio is not None:
            # Encode audio
            audio_features = self.audio_encoder(audio)
            
            # Project to LLM dimension
            audio_embeds = self.audio_projector(audio_features)
            
            # Insert audio embeddings at placeholder positions
            audio_mask = (input_ids == self.audio_token_id)
            inputs_embeds[audio_mask] = audio_embeds.view(-1, audio_embeds.shape[-1])
        
        # Forward through LLM
        outputs = self.llm(inputs_embeds=inputs_embeds)
        
        return outputs
    
    def generate_audio_description(self, audio, text_prompt="Describe this audio:"):
        """Generate text description of audio."""
        # Encode audio
        audio_features = self.audio_encoder(audio)
        audio_embeds = self.audio_projector(audio_features)
        
        # Tokenize prompt
        input_ids = self.tokenize(text_prompt)
        inputs_embeds = self.llm.embed_tokens(input_ids)
        
        # Append audio features
        inputs_embeds = torch.cat([inputs_embeds, audio_embeds], dim=1)
        
        # Generate
        outputs = self.llm.generate(inputs_embeds=inputs_embeds, max_length=100)
        
        return self.detokenize(outputs)
```

---

### 7.2 Whisper Large-v3: Multilingual ASR
Whisper-v3 supports 99 languages with improved accuracy.

**New Features**:
- 128K context window
- Improved tokenization
- Better low-resource language support

```python
class WhisperLargeV3:
    def __init__(self, model_size="large-v3"):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{model_size}"
        ).to("cuda")
        
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
    
    def transcribe(self, audio, language=None, task="transcribe"):
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array (16kHz)
            language: Language code (e.g., 'en', 'fr')
            task: 'transcribe' or 'translate' (to English)
        """
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to("cuda")
        
        # Set language and task
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task=task
        )
        
        # Generate
        predicted_ids = self.model.generate(
            inputs,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448
        )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def transcribe_with_timestamps(self, audio):
        """Transcribe with word-level timestamps."""
        # Use pipeline for timestamps
        from transformers import pipeline
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=8
        )
        
        result = pipe(audio, return_timestamps=True)
        
        return result["chunks"]  # List of {text, timestamp}
```

---

## 🔬 Research Frontiers (2024-2025)

### 8.1 Neural Audio Codecs
- **EnCodec**: High-efficiency neural audio compression
- **DAC**: Improved audio quality at low bitrates
- **UniAudio**: Universal audio tokenizer

### 8.2 Text-to-Audio Generation
- **AudioLDM**: Latent diffusion for audio
- **Stable Audio**: Commercial text-to-audio
- **MusicGen**: Music generation from text

### 8.3 Voice Conversion
- **VoiceFixer**: Voice restoration and enhancement
- **So-VITS-SVC**: Singing voice conversion
- **YourTTS**: Zero-shot voice cloning

### 8.4 Audio-Visual Learning
- **AV-HuBERT**: Audio-visual speech recognition
- **SyncNet**: Lip-sync verification
- **Wav2Lip**: Accurate lip-sync from audio

---

**Status:** ✅ Elite Expanded Standard (13/10)
**Next:** Phase 4 Elite Practice Problems (5-Level Graded Structure)
