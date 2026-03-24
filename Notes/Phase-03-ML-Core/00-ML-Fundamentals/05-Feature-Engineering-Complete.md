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

