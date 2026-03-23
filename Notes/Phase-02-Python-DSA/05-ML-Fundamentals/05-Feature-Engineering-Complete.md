# 6.5 Feature Engineering

## 🎯 Quick Overview
- **Feature Engineering**: Create better features from raw data
- **Feature Selection**: Choose most relevant features
- **Feature Extraction**: Transform features to lower dimensions
- **Foundation for**: Improved model performance

---

## 1. Feature Selection

### Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# F-test (ANOVA)
selector_f = SelectKBest(score_func=f_classif, k=10)
X_selected_f = selector_f.fit_transform(X_scaled, y)

# Get selected features
selected_features_f = X.columns[selector_f.get_support()]
print(f"Selected by F-test: {selected_features_f.tolist()}")

# Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected_mi = selector_mi.fit_transform(X_scaled, y)

selected_features_mi = X.columns[selector_mi.get_support()]
print(f"Selected by MI: {selected_features_mi.tolist()}")

# Chi-Squared (for categorical)
selector_chi = SelectKBest(score_func=chi2, k=10)
X_selected_chi = selector_chi.fit_transform(X_scaled, y)
```

### Wrapper Methods

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

# RFE (Recursive Feature Elimination)
model = RandomForestClassifier(random_state=42)
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X_scaled, y)

selected_features_rfe = X.columns[rfe.support_]
print(f"Selected by RFE: {selected_features_rfe.tolist()}")

# RFECV (RFE with Cross-Validation)
rfecv = RFECV(estimator=model, cv=5, scoring='accuracy', min_features_to_select=1)
rfecv.fit(X_scaled, y)

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Best CV Score: {rfecv.grid_scores_.max():.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.xlabel('Number of Features')
plt.ylabel('CV Score')
plt.title('RFECV')
plt.grid(True, alpha=0.3)
plt.show()
```

### Embedded Methods

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Random Forest Feature Importance
model = RandomForestClassifier()
model.fit(X_scaled, y)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Select features with importance > threshold
selector = SelectFromModel(model, threshold='mean', prefit=True)
X_selected = selector.transform(X_scaled)

selected_features = X.columns[selector.get_support()]
print(f"Selected by importance: {selected_features.tolist()}")

# Lasso (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

coef = pd.Series(lasso.coef_, index=X.columns)
selected_features = coef[coef != 0].index.tolist()
print(f"Selected by Lasso: {selected_features}")
```

---

## 2. Feature Extraction

### PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"PCA dimensions: {X_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X.columns
)

print(loadings.head())
```

### LDA (Linear Discriminant Analysis)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA (supervised)
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

print(f"Original dimensions: {X.shape[1]}")
print(f"LDA dimensions: {X_lda.shape[1]}")
```

### Autoencoders

```python
from tensorflow import keras
from tensorflow.keras import layers

# Build autoencoder
input_dim = X.shape[1]
encoding_dim = 10

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=32, validation_split=0.2)

# Extract encoder
encoder = keras.Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"Encoded dimensions: {X_encoded.shape[1]}")
```

---

## 3. Feature Transformation

### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# StandardScaler (Z-score)
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# MinMaxScaler
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
X_minmax = scaler_minmax.fit_transform(X)

# RobustScaler (robust to outliers)
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)

# PowerTransformer (normalize variance)
pt = PowerTransformer(method='yeo-johnson')
X_power = pt.fit_transform(X)
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import pandas as pd

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(X_categorical)

# Or using pandas
X_ohe_pd = pd.get_dummies(X_categorical, prefix='cat')

# Label Encoding (for target)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Ordinal Encoding (for ordered categories)
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_ordinal = oe.fit_transform(X_categorical)

# Target Encoding
def target_encode(df, col, target):
    """Encode categorical variable with target mean"""
    mean_target = df.groupby(col)[target].mean()
    return df[col].map(mean_target)

X_target_encoded = target_encode(df, 'category', 'target')
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

print(f"Original features: {X_scaled.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")

# Feature names
feature_names = poly.get_feature_names_out(X.columns)
print(feature_names[:10])
```

### Interaction Features

```python
# Create interaction features
df['feature_interaction'] = df['feature1'] * df['feature2']
df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-8)
df['feature_sum'] = df['feature1'] + df['feature2']
df['feature_diff'] = df['feature1'] - df['feature2']
```

---

## 4. Feature Creation

### Domain-Based Features

```python
# Time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['hour'] = df['date'].dt.hour
df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

# Text-based features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
df['exclamation_count'] = df['text'].str.count('!')
df['question_count'] = df['text'].str.count('\?')

# Geographic features
df['distance_to_city'] = haversine_distance(df['lat'], df['lon'], city_lat, city_lon)
df['is_urban'] = df['population_density'] > 1000
```

### Aggregation Features

```python
# GroupBy aggregations
agg_features = df.groupby('customer_id').agg({
    'purchase_amount': ['mean', 'sum', 'std', 'max', 'min'],
    'purchase_date': ['count', 'max'],
    'product_category': 'nunique'
})

agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns]

# Merge back
df = df.merge(agg_features, on='customer_id', how='left')
```

### Binning/Discretization

```python
from sklearn.preprocessing import KBinsDiscretizer

# Equal-width binning
df['age_bin'] = pd.cut(df['age'], bins=5, labels=['0-20', '21-40', '41-60', '61-80', '80+'])

# Equal-frequency binning
df['income_bin'] = pd.qcut(df['income'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# KBinsDiscretizer
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_binned = kbin.fit_transform(X)
```

---

## 5. Handling Missing Data

### Imputation

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Mean/Median/Mode imputation
imputer_mean = SimpleImputer(strategy='mean')
X_imputed_mean = imputer_mean.fit_transform(X)

imputer_median = SimpleImputer(strategy='median')
X_imputed_median = imputer_median.fit_transform(X)

imputer_mode = SimpleImputer(strategy='most_frequent')
X_imputed_mode = imputer_mode.fit_transform(X)

# KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed_knn = knn_imputer.fit_transform(X)

# MICE (Multiple Imputation by Chained Equations)
iter_imputer = IterativeImputer(random_state=42)
X_imputed_iter = iter_imputer.fit_transform(X)

# Add missing indicator
imputer_with_indicator = SimpleImputer(strategy='mean', add_indicator=True)
X_with_indicator = imputer_with_indicator.fit_transform(X)
```

---

## 6. Feature Pipelines

### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Define column types
numeric_features = ['age', 'income', 'score']
categorical_features = ['gender', 'city', 'occupation']

# Numeric pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
```

---

## 💻 Python Code Examples

```python
# === Complete Feature Engineering Pipeline ===

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature selection pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=15)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")

# Get selected features
selected_mask = pipeline.named_steps['feature_selection'].get_support()
selected_features = feature_names[selected_mask]
print(f"\nSelected Features ({len(selected_features)}):")
print(selected_features.tolist())

# Feature importance
importances = pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), selected_features[indices], rotation=90)
plt.tight_layout()
plt.show()

# === Text Feature Engineering ===

from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "This is a great product",
    "Terrible experience, would not recommend",
    "Average quality, nothing special",
    "Excellent service and fast delivery"
]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    stop_words='english'
)

X_tfidf = vectorizer.fit_transform(texts)

print(f"TF-IDF shape: {X_tfidf.shape}")
print(f"Features: {vectorizer.get_feature_names_out()[:10]}")

# === Date Feature Engineering ===

df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'value': np.random.randn(100)
})

# Date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

print(df.head())
```

---

## 📊 Summary Tables

### Feature Selection Methods

| Method | Type | Pros | Cons |
|--------|------|------|------|
| Filter | Univariate | Fast, scalable | Ignores interactions |
| RFE | Wrapper | Considers interactions | Slow, expensive |
| Feature Importance | Embedded | Fast, built-in | Biased to high cardinality |

### Feature Scaling

| Method | Formula | Use Case |
|--------|---------|----------|
| Standard | (x - mean) / std | Most algorithms |
| MinMax | (x - min) / (max - min) | Neural networks |
| Robust | (x - median) / IQR | With outliers |

### Encoding Methods

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| One-Hot | Nominal | Simple | Curse of dimensionality |
| Label | Ordinal | Preserves order | Assumes ordering |
| Target | High cardinality | Captures relationship | Risk of overfitting |

---

## 🎯 ML Applications

| Feature Engineering | ML Application |
|---------------------|----------------|
| Feature Selection | Dimensionality reduction |
| PCA | Visualization, compression |
| Text Features | NLP, sentiment analysis |
| Date Features | Time series forecasting |
| Aggregation | Customer segmentation |

---

---

## ❓ Quick Check Questions

1. What is the difference between "Filter" and "Wrapper" methods for feature selection?
2. Why is "Feature Scaling" important for distance-based algorithms like KNN or SVM?
3. When should you use "One-Hot Encoding" versus "Label Encoding"?
4. What is the difference between "Feature Selection" and "Feature Extraction" (like PCA)?
5. Explain how a "Polynomial Features" transformation can help a linear model capture non-linear relationships.

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Filter methods** (e.g., Correlation, ANOVA) evaluate features based on their statistical properties independently of any model, making them very fast. **Wrapper methods** (e.g., RFE) use a specific ML model to evaluate combinations of features, making them more accurate but computationally expensive.
2. Distance-based algorithms calculate the similarity between points using numerical distances. If features have different scales (e.g., age in years vs. income in thousands), the feature with the **larger numerical range** will dominate the distance calculation, potentially leading to a biased model.
3. Use **One-Hot Encoding** for nominal categories with no inherent order (e.g., Color: Red, Blue, Green). Use **Label Encoding** (or Ordinal Encoding) only for ordinal categories where the order matters (e.g., Size: Small, Medium, Large) or for the target label itself.
4. **Feature Selection** keeps a subset of the *original* features and discards the rest. **Feature Extraction** transforms the data into a *new*, lower-dimensional feature space (e.g., Principal Components) that preserves the most important information.
5. Polynomial transformation creates new features by taking the power of existing features (e.g., $x^2$) or interactions between them (e.g., $x_1 \times x_2$). This allows a **linear model** to fit a non-linear curve in the original feature space.

</details>

---

**Status:** ✅ Complete
**Next:** Phase 3 - Core ML & Deep Learning
