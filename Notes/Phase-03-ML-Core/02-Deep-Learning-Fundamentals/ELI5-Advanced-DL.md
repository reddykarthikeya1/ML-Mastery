
---

#### 🧒 ELI5: Advanced DL Topics - Metric Learning, Siamese Networks, Online Learning

> Imagine you're a judge, a twin detector, and a lifelong learner.
>
> **Metric Learning** (Learning similarity):
>
> **Problem**: What does "similar" mean?
> - Is a husky similar to a wolf?
> - Is a Chihuahua similar to a husky?
> - Distance in pixels doesn't work!
>
> **Metric Learning Solution**: Learn a BETTER distance!
> - Same class → CLOSE together
> - Different class → FAR apart
>
> **Triplet Loss**:
> ```
> Anchor (your photo)
>     ↓
> Positive (another photo of you) → Should be CLOSE
>     ↓
> Negative (photo of stranger) → Should be FAR
> ```
>
> **Loss function**:
> - Distance(Anchor, Positive) + margin < Distance(Anchor, Negative)
> - "Make sure positive is closer than negative by at least margin!"
>
> **Like**: Training a bouncer
> - "These two photos = same person (let them in)"
> - "These two photos = different people (keep out)"
>
> **Applications**:
> - Face recognition (phone unlock)
> - Signature verification (bank checks)
> - Product recommendations ("similar items")
>
> **Siamese Networks** (Twin networks):
>
> **Architecture**:
> ```
> Image A → [Network] → Features A
>                         ↓
>                     Compare
>                         ↓
> Image B → [Network] → Features B
> ```
>
> **Key idea**: SAME network for both images!
> - Share weights (twins!)
> - If A and B similar → features close
> - If A and B different → features far
>
> **Why Siamese?**:
> - One-shot learning (learn from ONE example!)
> - "I've seen you once, I'll recognize you again"
> - No retraining needed!
>
> **Applications**:
> - Signature verification (one real signature → detect fakes)
> - Face verification (one photo → recognize person)
> - Duplicate detection (is this question already asked?)
>
> **Online Learning** (Learn continuously):
>
> **Problem**: World changes!
> - Spam emails evolve
> - User preferences change
> - New classes appear
> - Can't retrain from scratch every day!
>
> **Online Learning Solution**: Update incrementally!
> - New data arrives → small update
> - No full retraining
> - Adapts to changes!
>
> **Like**: Learning from experience
> - Day 1: Learn pattern A
> - Day 2: Pattern shifts slightly → adjust
> - Day 3: Pattern shifts more → adjust more
> - Always current!
>
> **Challenges**:
> - **Catastrophic forgetting**: New info overwrites old!
>   - Solution: Replay buffer (remember old examples)
> - **Concept drift**: Patterns change over time
>   - Solution: Weight decay (forget old gradually)
>
> **Applications**:
> - Stock prediction (market changes daily)
> - Recommendation systems (preferences evolve)
> - Spam filtering (spammers adapt)
>
> **Self-Supervised Learning** (Labels from data itself):
>
> **Problem**: Labels are expensive!
> - 1 billion images on internet
> - 0 labeled
> - Can't train supervised!
>
> **Self-Supervised Solution**: Create your OWN labels!
>
> **Methods**:
> - **Rotation prediction**: Rotate image 0°, 90°, 180°, 270°
>   - Task: "Which rotation was applied?"
>   - Model learns image features!
> - **Jigsaw puzzle**: Cut image into 9 pieces, shuffle
>   - Task: "Put pieces back together"
>   - Model learns spatial relationships!
> - **Masked prediction** (BERT): Mask 15% of words
>   - Task: "Predict masked words"
>   - Model learns language!
>
> **Like**: Creating your own homework
> - Teacher doesn't give questions
> - You make up questions, answer them
> - Still learn!
>
> **Why it works**:
> - Tasks require understanding
> - "To predict rotation, must understand objects"
> - Features transfer to real tasks!
>
> **Applications**:
> - BERT (masked language modeling)
> - GPT (next word prediction)
> - SimCLR (contrastive learning)
> - MAE (masked autoencoders)
>
> **Graph Neural Networks (GNNs)** (Learning on graphs):
>
> **Problem**: Data is connected!
> - Social networks (friends connected)
> - Molecules (atoms connected)
> - Knowledge graphs (facts connected)
> - CNNs/RNNs can't handle this!
>
> **GNN Solution**: Message passing!
> ```
>   Node A
>   /   \
>  B --- C
> ```
> - Each node collects info from neighbors
> - A collects from B and C
> - B collects from A and C
> - Update features based on neighbors!
>
> **Like**: Gossip spreading
> - "What I know = what I knew + what neighbors told me"
> - After few rounds, everyone knows everything!
>
> **Applications**:
> - Social network analysis (influence, communities)
> - Drug discovery (molecule properties)
> - Recommendation (user-item graphs)
> - Fraud detection (transaction networks)
>
> **When to use which**:
> - **Metric Learning**: Need similarity measure
> - **Siamese Networks**: One-shot learning, verification
> - **Online Learning**: Data arrives continuously, patterns change
> - **Self-Supervised**: No labels, want to pretrain
> - **GNNs**: Data has graph structure

</details>
