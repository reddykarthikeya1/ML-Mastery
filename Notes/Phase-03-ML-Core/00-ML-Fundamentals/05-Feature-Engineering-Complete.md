# 6.5 Feature Engineering

## 🎯 Quick Overview
- **Feature Selection**: Choose relevant features
- **Feature Extraction**: Transform to lower dimensions
- **Feature Creation**: Create new features
- **Foundation for**: Improved model performance

---

#### 🧒 ELI5: Feature Engineering, Scaling & Encoding

> Imagine you're packing for a trip and need to organize your stuff.
>
> **Feature Engineering** (Preparing data for ML):
>
> **Raw data** is like a messy room:
> - Clothes everywhere
> - Some items broken
> - Important stuff hidden
>
> **Feature Engineering** = Organizing before the trip!
>
> **Feature Selection** (What to pack):
>
> **Filter Methods** (Quick check):
> - "Do I use this often?"
> - Keep items with high usage score
> - Fast but simple
>
> **Wrapper Methods** (Try combinations):
> - "Let me try packing 10 items"
> - "Now try 11 items - better?"
> - "Now try 9 - worse?"
> - Slow but finds best combination!
>
> **Embedded Methods** (Smart packing):
> - Uses ML model's opinion
> - "Random Forest says this feature is important"
> - Best of both worlds!
>
> **Feature Extraction** (Compressing):
>
> **PCA** (Combining related items):
> - You have: Shampoo, conditioner, body wash, face wash
> - PCA: "These are all BATH products!"
> - Combine into: "Bath score" (1 feature instead of 4)
> - Like: Rolling clothes instead of folding - same stuff, less space!
>
> **Why scale features?** (Making everything same size):
>
> **Problem**: Features have different scales!
> - Age: 0-100
> - Salary: 0-1,000,000
> - Model thinks salary is MORE important (bigger numbers!)
>
> **StandardScaler** (Make mean=0, std=1):
> - Age 50 → 0 (average age)
> - Age 25 → -1 (below average)
> - Salary 500k → 0 (average salary)
> - Now both equally important!
>
> **MinMaxScaler** (Squish to 0-1 range):
> - Youngest person → 0
> - Oldest person → 1
> - Everyone else → 0.3, 0.7, etc.
> - Like: Making everyone's height fit in a box!
>
> **Encoding** (Converting text to numbers):
>
> **One-Hot Encoding** (Create yes/no columns):
> - Color: Red, Blue, Green
> - Becomes 3 columns:
>   - Is_Red: 1 or 0
>   - Is_Blue: 1 or 0
>   - Is_Green: 1 or 0
> - Like: Checking boxes on a form!
>
> **Label Encoding** (Assign numbers):
> - Color: Red=0, Blue=1, Green=2
> - Simpler but implies order (2 > 1 > 0)
> - Good for: Ordinal data (Small, Medium, Large)
>
> **Data Leakage** (Cheating on the test):
>
> **Problem**: Using info you shouldn't have!
> - Like: Seeing test questions before studying
> - Model does GREAT on training
> - TERRIBLE on real data (no peeking!)
>
> **Common leaks**:
> - Scaling BEFORE train/test split (peeking at test data!)
> - Using future info to predict past
> - Including ID columns (unique to each row!)
>
> **Rule**: Train/test split FIRST, then scale/encode!

</details>

---

## 1. Feature Selection

---

#### 🧒 ELI5: Class Imbalance Handling - SMOTE, Class Weights, Anomaly Detection

> Imagine you're a bouncer at a club trying to spot fake IDs.
>
> **Class Imbalance Problem** (Rare events):
>
> **Scenario**: 99% real IDs, 1% fake IDs
> - Model: "Everyone is REAL!"
> - Accuracy: 99%! 🎉
> - Problem: Caught ZERO fake IDs! 😱
>
> **Why imbalance breaks ML**:
> - Model takes the easy path
> - "Why try hard when 99% works?"
> - Like: "Guess 'rain' in desert - 99% right!"
>
> **Solution 1: SMOTE** (Synthetic Minority Oversampling):
>
> **Problem**: Only 100 fake IDs in dataset!
> - Model doesn't see enough examples
> - Can't learn the patterns!
>
> **SMOTE Solution** (Create fake fakes!):
> - Take 2 fake IDs
> - Blend their features: "Half of this one, half of that one"
> - Create NEW synthetic fake ID!
> - Repeat until you have 1000 fake IDs!
>
> **How SMOTE works**:
> - Pick a minority sample
> - Find its K nearest neighbors
> - Create new sample BETWEEN them
> - "You're 70% like sample A, 30% like sample B"
>
> **Why SMOTE works**:
> - More examples for minority class
> - Synthetic samples are realistic (between real samples)
> - Model sees balanced data!
>
> **Solution 2: Class Weights** (Punish mistakes differently):
>
> **Normal training**:
> - Mistake on real ID: -1 point
> - Mistake on fake ID: -1 point
> - Model: "Whatever, I'll just guess real"
>
> **Weighted training**:
> - Mistake on real ID: -1 point
> - Mistake on fake ID: -99 points! 😨
> - Model: "I BETTER catch those fake IDs!"
>
> **How to set weights**:
> - Inverse of class frequency
> - Real (99%): weight = 1
> - Fake (1%): weight = 99
> - "Rare class = Higher penalty!"
>
> **Solution 3: Threshold Moving** (Change the bar):
>
> **Default threshold**:
> - "If P(fake) > 0.5 → FAKE"
> - Model rarely predicts > 0.5 for fake
> - Catches nothing!
>
> **Lower threshold**:
> - "If P(fake) > 0.1 → FAKE"
> - Catches MORE fake IDs!
> - More false alarms, but catches the real fakes!
>
> **Solution 4: Anomaly Detection** (Different approach):
>
> **Instead of classification**:
> - Don't learn "fake vs real"
> - Learn "what normal looks like"
> - Flag anything weird as anomaly!
>
> **Like**:
> - Normal: "ID has photo, name, birthdate"
> - Anomaly: "This ID has NO photo!" → FAKE!
> - Works even with ZERO fake examples!
>
> **Anomaly Detection Methods**:
>
> **Isolation Forest** (Isolating weirdos):
> - Randomly split data
> - Anomalies get isolated QUICKLY (fewer splits)
> - Like: "Weird person stands out in crowd"
>
> **One-Class SVM** (Drawing boundary):
> - Draw tight boundary around normal data
> - Anything outside = anomaly!
> - Like: "If you're not in the club, you're out!"
>
> **Autoencoders** (Reconstruction error):
> - Train to reconstruct normal IDs
> - Fake IDs → HIGH reconstruction error!
> - "This doesn't look like what I learned!"
>
> **When to use what**:
> - **SMOTE**: 100-1000 minority samples, need more data
> - **Class Weights**: Simple, works with any model
> - **Threshold Moving**: Post-processing, no retraining
> - **Anomaly Detection**: VERY rare (< 0.1%) or zero examples
>
> **Real-world examples**:
> - **Fraud detection**: Class weights + threshold moving
> - **Medical diagnosis**: SMOTE + ensemble
> - **Defect detection**: Anomaly detection (rare defects)
> - **Spam filtering**: Class weights (spam is rare)
>
> **Combining techniques**:
> - SMOTE + Class Weights = Best of both!
> - Anomaly Detection + Threshold = Maximum sensitivity!
> - Ensemble of all = State-of-the-art!

</details>

---

### Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select K best features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected features
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
```

### Wrapper Methods

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

# RFE
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
rfe.fit(X, y)

# RFECV
rfecv = RFECV(estimator=RandomForestClassifier(), cv=5)
rfecv.fit(X, y)

print(f"Optimal features: {rfecv.n_features_}")
```

### Embedded Methods

```python
from sklearn.feature_selection import SelectFromModel

# Feature importance
model = RandomForestClassifier()
model.fit(X, y)

# Select features
selector = SelectFromModel(model, threshold='mean')
X_selected = selector.fit_transform(X, y)
```

---

#### 🧒 ELI5: Feature Selection - Filter, Wrapper, Embedded Methods

> Imagine you're packing for a trip but your suitcase is too small.
>
> **Feature Selection Problem** (Too many features!):
> - You have 1000 features (clothes, gadgets, books)
> - Model gets confused (too many choices!)
> - Overfits (memorizes instead of learning)
> - Slow training (processing everything!)
>
> **Solution**: Pack only what you NEED!
>
> **Filter Methods** (Quick screening):
>
> **How it works**:
> - Score EACH feature independently
> - "How much does this feature correlate with target?"
> - Top K features win!
>
> **Methods**:
> - **Correlation**: "Does feature move with target?"
> - **Chi-square**: "Are feature and target dependent?"
> - **ANOVA F-test**: "Do feature values differ across classes?"
> - **Mutual Information**: "How much does feature tell us about target?"
>
> **Pros**:
> - ✅ Fast (score once, done!)
> - ✅ Model-agnostic (works with any model)
> - ✅ Scales to millions of features
>
> **Cons**:
> - ❌ Ignores feature interactions
> - ❌ "Height" and "Weight" might both score high, but redundant!
>
> **Like**: Job applications
> - Filter: "Must have 5+ years experience" (quick screen)
> - Fast but might miss great candidates!
>
> **Wrapper Methods** (Try combinations):
>
> **How it works**:
> - Try DIFFERENT feature subsets
> - Train model on each subset
> - Keep the best!
>
> **Methods**:
> - **Forward Selection**: Start with 0 features, add best one at a time
> - **Backward Elimination**: Start with ALL features, remove worst one at a time
> - **Recursive Feature Elimination (RFE)**: Train model, remove weakest, repeat!
>
> **Pros**:
> - ✅ Considers feature interactions
> - ✅ Finds best subset for YOUR model
>
> **Cons**:
> - ❌ SLOW (train model many times!)
> - ❌ 1000 features → 2¹⁰⁰⁰ combinations! (impossible!)
>
> **Like**: Trying outfit combinations
> - "Does this shirt go with these pants?"
> - Takes forever but looks great!
>
> **Embedded Methods** (Built-in selection):
>
> **How it works**:
> - Feature selection happens DURING training!
> - Model learns which features are useful
> - Automatically ignores useless ones!
>
> **Methods**:
> - **Lasso (L1)**: Forces some feature weights to ZERO
> - **Random Forest**: "Feature importance" - how much each feature helps
> - **XGBoost/LightGBM**: Built-in feature selection
>
> **Pros**:
> - ✅ Fast (no extra training!)
> - ✅ Model-specific (optimized for your model)
> - ✅ Considers interactions
>
> **Cons**:
> - ❌ Tied to specific model
>
> **Like**: Smart packing
> - "I'll only pack what I'll actually USE"
> - Learn while packing!
>
> **When to use which**:
> - **Filter**: Millions of features, need quick reduction
> - **Wrapper**: Small dataset (< 100 features), want best accuracy
> - **Embedded**: Default choice! (best balance)
>
> **Real workflow**:
> 1. Filter: 10,000 → 1,000 features (quick screen)
> 2. Embedded (Lasso): 1,000 → 100 features (model-based)
> 3. Wrapper (RFE): 100 → 50 features (fine-tune)
> - Best of all worlds!

</details>

---

## 2. Feature Extraction

### PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)

print(f"Original: {X.shape[1]}, PCA: {X_pca.shape[1]}")
```

### LDA

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)
```

---

## 3. Feature Transformation

### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
```

### Encoding

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-Hot Encoding
ohe = OneHotEncoder()
X_ohe = ohe.fit_transform(X_categorical)

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

---

## 💻 Python Code Examples

```python
# === Complete Feature Engineering Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=15)),
    ('classifier', RandomForestClassifier())
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
```

---

## 📊 Summary Tables

### Feature Selection Methods

| Method | Type | Pros | Cons |
|--------|------|------|------|
| Filter | Univariate | Fast | Ignores interactions |
| RFE | Wrapper | Considers interactions | Slow |
| Feature Importance | Embedded | Fast | Biased to high cardinality |

### Feature Scaling

| Method | Formula | Use Case |
|--------|---------|----------|
| Standard | (x - mean) / std | Most algorithms |
| MinMax | (x - min) / (max - min) | Neural networks |

---

## 🎯 ML Applications

| Technique | ML Application |
|-----------|----------------|
| Feature Selection | Dimensionality reduction |
| PCA | Visualization, compression |
| Encoding | Categorical variables |

---

---

## ❓ Quick Check Questions

1. What is the difference between Feature Selection and Feature Extraction?
2. How does One-Hot Encoding differ from Label Encoding?
3. Why do we scale features for algorithms like KNN?
4. What is the "Recursive Feature Elimination" (RFE) technique?
5. When would you use a Pipeline in Scikit-Learn?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Feature Selection** involves choosing a subset of the original features. **Feature Extraction** involves transforming the original features into a new, lower-dimensional set of features (e.g., PCA).
2. **One-Hot Encoding** creates new binary columns for each unique category (no order implied). **Label Encoding** assigns a unique integer to each category (implies an ordinal relationship).
3. We scale features because **KNN** relies on distance calculations. If one feature has a much larger range than others, it will dominate the distance metric and bias the model.
4. **RFE** is a wrapper-based feature selection method that fits a model and removes the weakest feature(s) until the specified number of features is reached.
5. You use a **Pipeline** to bundle preprocessing steps (like scaling and encoding) together with an estimator, ensuring that the same transformations are applied to both training and test data, which helps prevent data leakage.

</details>

---

**Status:** ✅ Complete
**Next:** Classical ML Algorithms

