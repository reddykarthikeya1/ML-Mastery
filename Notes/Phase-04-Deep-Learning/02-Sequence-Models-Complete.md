# 10.2 Sequence Models for NLP

## 🎯 Quick Overview
- **RNNs (Recurrent Neural Networks)**: Processing sequential data with hidden states
- **LSTMs & GRUs**: Solving the vanishing gradient problem in long sequences
- **Encoder-Decoder (Seq2Seq)**: The foundation for Machine Translation
- **Attention Mechanism**: Allowing models to focus on specific parts of the input
- **Foundation for**: Transformers, BERT, GPT

---

## 1. Recurrent Neural Networks (RNNs)

Standard neural networks process inputs independently. RNNs maintain a **hidden state ($h_t$)** that acts as "memory," allowing information from previous time steps to influence the current output.

### 1.1 The Recurrent Step
At each time step $t$, the RNN takes input $x_t$ and the previous hidden state $h_{t-1}$:
$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

### 1.2 The Fatal Flaw: Vanishing Gradients
During Backpropagation Through Time (BPTT), gradients are multiplied by the weight matrix repeatedly. If weights are small, gradients shrink exponentially, making the RNN "forget" the beginning of long sentences.

---

## 2. Advanced Gated Units (LSTM & GRU)

To solve vanishing gradients, gated architectures were introduced to control the flow of information.

### 2.1 LSTM (Long Short-Term Memory)
Uses a **Cell State ($C_t$)** acting as a "long-term memory" highway.
- **Forget Gate**: Decides what to throw away from $C_{t-1}$.
- **Input Gate**: Decides what new info to add to $C_t$.
- **Output Gate**: Decides what part of $C_t$ to output as the hidden state $h_t$.

### 2.2 GRU (Gated Recurrent Unit)
A simplified, faster version of LSTM.
- Combines Forget and Input gates into a single **Update Gate**.
- Merges the cell state and hidden state.

---

## 3. Encoder-Decoder Architecture (Seq2Seq)

Used for tasks where the input and output sequences have different lengths (e.g., Translation).

1. **Encoder**: Processes the input sequence into a fixed-length "context vector" (the final hidden state).
2. **Decoder**: Takes the context vector and generates the output sequence token-by-token.
- **Problem**: A single fixed-length vector is a "bottleneck" for long sentences.

---

## 4. The Attention Mechanism

Instead of forcing the encoder to compress everything into one vector, **Attention** allows the decoder to "look back" at all encoder hidden states at every step.

### 4.1 How it works
1. Calculate a **Score** (similarity) between the current decoder state and all encoder states.
2. Convert scores into **Weights** (using Softmax).
3. Compute a **Context Vector** as a weighted sum of encoder states.
4. Use this context vector to predict the next token.

---

## 💻 Python Code Examples

### 1. Simple LSTM for Sentiment Analysis (PyTorch)
```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x) 
        # out shape: (batch, seq_len, hidden)
        # hidden shape: (1, batch, hidden)
        out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state for classification
        return self.fc(hidden.squeeze(0))
```

### 2. Sequence-to-Sequence Concept (Logic)
```python
# Pseudo-logic for Seq2Seq
context = encoder(input_sequence)
current_token = "<START>"

while current_token != "<END>":
    output_probs, next_hidden = decoder(current_token, context)
    current_token = sample(output_probs)
    result.append(current_token)
```

---

## 📊 Summary Table

| Model | Key Feature | Best For | Pros | Cons |
|-------|-------------|----------|------|------|
| **Vanilla RNN** | Basic recurrence | Very short sequences | Simple, fast | Vanishing gradients |
| **LSTM** | Cell state + 3 gates | Long dependencies | Strong memory | Slow, complex |
| **GRU** | 2 gates | Efficiency | Faster than LSTM | Slightly less expressive |
| **Seq2Seq** | Encoder-Decoder | Translation, Summary | Flexible lengths | Bottleneck problem |
| **Attention** | Dynamic weighting | SOTA Seq2Seq | Fixes bottleneck | Computational cost |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Bi-LSTM | Named Entity Recognition (NER) |
| Seq2Seq | Google Translate (Classic) |
| GRU | Real-time speech-to-text |
| Attention | Document summarization |

---

## ❓ Quick Check Questions

1. Why does a standard RNN struggle with a sentence that has 50 words?
2. What is the specific job of the "Forget Gate" in an LSTM?
3. How does a GRU differ architecturally from an LSTM?
4. What is the "Bottleneck Problem" in basic Encoder-Decoder models?
5. In the Attention mechanism, what does a high attention weight between two words imply?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. It suffers from **Vanishing Gradients**. The mathematical influence of the 1st word on the 50th word's update becomes nearly zero after being multiplied by small weights 50 times.
2. The **Forget Gate** determines which information from the previous cell state ($C_{t-1}$) is no longer relevant and should be discarded. It outputs a value between 0 (completely forget) and 1 (completely keep).
3. A **GRU** has only 2 gates (Update and Reset) instead of 3, and it removes the separate "Cell State," using only the hidden state to carry memory. This makes it more computationally efficient.
4. The **Bottleneck Problem** occurs when the model tries to compress all the information from a long input sentence into a single, fixed-size vector. This causes the model to lose fine-grained details from the beginning of the sentence.
5. It implies a strong **semantic relationship** or dependency. For example, in translation, the decoder might put high attention on the source noun "Apple" when generating the corresponding object in the target language.

</details>

---

**Status:** ✅ Complete
**Next:** The Transformer Architecture (The Revolution)
