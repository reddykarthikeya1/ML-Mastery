# Classical ML Algorithms - Complete Notes

## 📚 Complete Topic List (6 Topics)

### 7.1 Linear Models ✅
- [x] Linear Regression (OLS, Gradient Descent)
- [x] Logistic Regression (Binary, Multi-class)
- [x] Generalized Linear Models (Ridge, Lasso, Elastic Net, GLM)

### 7.2 Tree-Based Models ✅
- [x] Decision Trees (ID3, C4.5, CART)
- [x] Ensemble Methods (Bagging, Boosting, Random Forest)
- [x] Advanced Boosting (XGBoost, LightGBM, CatBoost)

### 7.3 Instance-Based Learning ✅
- [x] K-Nearest Neighbors (KNN)
- [x] Distance Metrics (Euclidean, Manhattan, Minkowski, Cosine)
- [x] KD-Trees, Ball Trees
- [x] Learning Vector Quantization (LVQ)

### 7.4 Support Vector Machines ✅
- [x] Maximum Margin Classifier
- [x] Hard/Soft Margin SVM
- [x] Kernel Functions (Linear, Polynomial, RBF, Sigmoid)
- [x] Support Vector Regression (SVR)
- [x] SMO Algorithm

### 7.5 Naive Bayes ✅
- [x] Bayes Theorem
- [x] Gaussian Naive Bayes
- [x] Multinomial Naive Bayes
- [x] Bernoulli Naive Bayes
- [x] Laplace Smoothing

### 7.6 Clustering Algorithms ✅
- [x] K-Means (with K-Means++)
- [x] DBSCAN, HDBSCAN
- [x] Hierarchical Clustering
- [x] Gaussian Mixture Models (GMM)
- [x] Expectation-Maximization (EM) Algorithm

---

#### 🧒 ELI5: Advanced ML Topics - Association Rules, Semi-Supervised, Multi-Task Learning

> Imagine you're a detective, a teacher, and a multitasker.
>
> **Association Rules** (Market Basket Analysis):
>
> **Problem**: What items are bought together?
> - "People who buy diapers also buy beer"
> - "People who buy phones also buy cases"
> - How to find these patterns?
>
> **Apriori Algorithm**:
> - **Support**: How often does {diaper, beer} appear together?
>   - Support = 100/1000 = 10% (10% of transactions)
> - **Confidence**: If someone buys diaper, how likely to buy beer?
>   - Confidence = 100/200 = 50% (half of diaper buyers buy beer)
> - **Lift**: Is this correlation or coincidence?
>   - Lift = 50% / 20% = 2.5 (2.5× more likely than random!)
>
> **Like**: Matchmaking for products
> - "You bought X, you might like Y!"
> - Amazon recommendations!
>
> **Applications**:
> - Supermarket layout (put related items together)
> - Netflix: "Because you watched..."
> - Website recommendations
>
> **Semi-Supervised Learning** (Best of both worlds):
>
> **Problem**: Labels are EXPENSIVE!
> - 1 million images
> - Only 1000 labeled (0.1%)
> - Labeling rest: $10,000 and 100 hours!
>
> **Semi-Supervised Solution**:
> - Use 1000 labeled + 999,000 unlabeled
> - Learn from BOTH!
>
> **Methods**:
> - **Self-Training**: 
>   1. Train on labeled data
>   2. Predict on unlabeled (pseudo-labels)
>   3. Add confident predictions to training set
>   4. Repeat!
> - **Co-Training**:
>   1. Train TWO models on different views
>   2. Model 1 labels data for Model 2
>   3. Model 2 labels data for Model 1
>   4. Both improve!
>
> **Like**: Teacher with TA
> - Teacher labels 100 problems
> - TA (model) labels rest
> - Teacher checks TA's work
> - Everyone learns!
>
> **When to use**:
> - ✅ Lots of unlabeled data
> - ✅ Few labeled examples
> - ✅ Labeling is expensive
>
> **Multi-Task Learning** (Learn multiple things at once):
>
> **Problem**: Training separate models is inefficient!
> - Model 1: Detect cars
> - Model 2: Detect pedestrians
> - Model 3: Detect traffic lights
> - 3× training, 3× memory!
>
> **Multi-Task Solution**: ONE model, multiple outputs!
> ```
>           Input (image)
>               ↓
>        Shared layers (learn features)
>         ↓    ↓    ↓
>      Cars  People  Lights
> ```
>
> **Why it works**:
> - Shared layers learn GENERAL features
> - "Edges, textures, shapes" useful for ALL tasks
> - Like: Learning math helps physics AND chemistry!
>
> **Benefits**:
> - ✅ Faster training (one model)
> - ✅ Better generalization (shared knowledge)
> - ✅ Less memory (shared layers)
>
> **Real examples**:
> - Self-driving: Detect cars + people + signs (one model)
> - Medical: Diagnose disease A + disease B (one model)
> - NLP: POS tagging + named entities (one model)
>
> **Curriculum Learning** (Teach in order):
>
> **Problem**: Random training is inefficient!
> - Show hard examples first → model confused
> - Like: Teaching calculus before addition!
>
> **Curriculum Solution**: Easy → Hard!
> - Phase 1: Simple examples (clear patterns)
> - Phase 2: Medium examples (some noise)
> - Phase 3: Hard examples (ambiguous)
>
> **Like**: School curriculum
> - Grade 1: Addition
> - Grade 5: Multiplication
> - Grade 10: Calculus
> -循序渐进!
>
> **Why it works**:
> - Build foundation first
> - Gradually increase difficulty
> - Model gains confidence
>
> **Applications**:
> - Language learning: Simple sentences → complex
> - Image recognition: Clear objects → occluded
> - Speech: Clean audio → noisy audio
>
> **Contrastive Learning** (Learn by comparison):
>
> **Problem**: How to learn without labels?
> - Millions of images, no labels
> - Can't train classifier
>
> **Contrastive Solution**: Learn what's SIMILAR vs DIFFERENT!
> - "These two augmented views of same image = SIMILAR"
> - "These two different images = DIFFERENT"
>
> **Like**: Twin identification
> - "These two photos = same person (different angles)"
> - "These two photos = different people"
> - Learn features that distinguish!
>
> **SimCLR / MoCo**:
> - Take image, create two augmented views
> - "These should be close in feature space"
> - Other images should be far
> - Learn without ANY labels!
>
> **When to use which**:
> - **Association Rules**: Market basket, recommendations
> - **Semi-Supervised**: Few labels, lots of unlabeled data
> - **Multi-Task**: Related tasks, efficiency needed
> - **Curriculum**: Complex tasks, gradual learning helps
> - **Contrastive**: Unlabeled data, want representations

</details>

---

## 📝 Complete Notes Index

| # | Topic | File | Status |
|---|-------|------|--------|
| 1 | 7.1 Linear Models | `01-Linear-Models-Complete.md` | ✅ |
| 2 | 7.2 Tree-Based Models | `02-Tree-Based-Models-Complete.md` | ✅ |
| 3 | 7.3 Instance-Based Learning | `03-Instance-Based-Learning.md` | ✅ |
| 4 | 7.4 Support Vector Machines | `04-Support-Vector-Machines.md` | ✅ |
| 5 | 7.5 Naive Bayes | `05-Naive-Bayes.md` | ✅ |
| 6 | 7.6 Clustering Algorithms | `06-Clustering-Algorithms.md` | ✅ |
| 7 | Practice Problems | `Practice-Problems.md` | ✅ |

---

## 📊 Summary

| Section | Topics | Files |
|---------|--------|-------|
| Linear Models | 15 subtopics | 1 file |
| Tree-Based Models | 25 subtopics | 1 file |
| Instance-Based Learning | 10 subtopics | 1 file |
| Support Vector Machines | 15 subtopics | 1 file |
| Naive Bayes | 10 subtopics | 1 file |
| Clustering Algorithms | 15 subtopics | 1 file |
| **Total** | **90 subtopics** | **7 files** |

---

## 🎯 What's Next?

After Classical ML Algorithms:
1. ✅ Deep Learning Fundamentals
2. ✅ Deep Learning Frameworks
3. ✅ Phase 4: Specialization (NLP, Computer Vision)

---

## 🔗 Related Topics
- [[../00-ML-Fundamentals/README|ML Fundamentals]]
- [[../02-Deep-Learning-Fundamentals/README|Deep Learning Fundamentals]]

---
**Phase:** 03 - Core ML & Deep Learning
**Status:** 🟢 Complete
**Last Updated:** 2026-03-23
