# 4.3 Data Preprocessing and Cleaning

## 🎯 Quick Overview
- **Data Quality**: Ensure data is accurate and usable
- **Missing Data**: Handle incomplete records
- **Outliers**: Detect and treat anomalies
- **Transformation**: Scale and encode features
- **Foundation for**: ML model performance, data analysis

---

## 1. Data Quality

### Dimensions of Data Quality

```
1. Accuracy: Data correctly represents reality
2. Completeness: No missing values
3. Consistency: Data is uniform across sources
4. Timeliness: Data is up-to-date
5. Validity: Data conforms to rules/constraints
6. Uniqueness: No duplicate records
```

### Data Profiling

```python
import pandas as pd
import numpy as np

def profile_data(df):
    """Comprehensive data profiling"""
    
    profile = pd.DataFrame()
    
    # Basic info
    profile['type'] = df.dtypes
    profile['count'] = df.count()
    profile['missing'] = df.isnull().sum()
    profile['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
    profile['unique'] = df.nunique()
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    profile.loc[numeric_cols, 'mean'] = df[numeric_cols].mean()
    profile.loc[numeric_cols, 'std'] = df[numeric_cols].std()
    profile.loc[numeric_cols, 'min'] = df[numeric_cols].min()
    profile.loc[numeric_cols, 'max'] = df[numeric_cols].max()
    
    return profile

# Usage
profile = profile_data(df)
print(profile)
```

### Data Validation Rules

```python
def validate_data(df):
    """Apply validation rules"""
    
    errors = []
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate rows")
    
    # Check for missing values
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            errors.append(f"Column '{col}' has {count} missing values")
    
    # Check value ranges
    if 'age' in df.columns:
        invalid_ages = df[(df['age'] < 0) | (df['age'] > 150)]
        if len(invalid_ages) > 0:
            errors.append(f"Found {len(invalid_ages)} invalid ages")
    
    # Check email format
    if 'email' in df.columns:
        invalid_emails = df[~df['email'].str.contains('@', na=False)]
        if len(invalid_emails) > 0:
            errors.append(f"Found {len(invalid_emails)} invalid emails")
    
    return errors

# Usage
errors = validate_data(df)
for error in errors:
    print(f"ERROR: {error}")
```

---

## 2. Handling Missing Data

### Types of Missing Data

```
MCAR (Missing Completely At Random):
- Missingness is independent of both observed and unobserved data
- Example: Equipment randomly fails

MAR (Missing At Random):
- Missingness depends on observed data but not unobserved
- Example: Men are less likely to report salary

MNAR (Missing Not At Random):
- Missingness depends on unobserved data
- Example: People with high salaries don't report salary
```

### Detection

```python
import missingno as msno
import matplotlib.pyplot as plt

# Count missing values
df.isnull().sum()

# Percentage missing
(df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

# Visualize missing patterns
msno.matrix(df)  # Missing data matrix
msno.heatmap(df)  # Correlation of missingness
msno.dendrogram(df)  # Hierarchical clustering of missing patterns

plt.show()
```

### Deletion Methods

```python
# Listwise deletion (drop rows with any missing)
df_clean = df.dropna()

# Drop rows with all missing
df_clean = df.dropna(how='all')

# Drop columns with missing
df_clean = df.dropna(axis=1)

# Drop rows with threshold (keep rows with at least n non-missing)
df_clean = df.dropna(thresh=5)

# Drop based on specific columns
df_clean = df.dropna(subset=['important_col1', 'important_col2'])
```

### Imputation Methods

```python
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
import numpy as np

# Mean/Median/Mode imputation
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
df_imputed = imputer.fit_transform(df)

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)

# Forward/Backward fill (time series)
df.fillna(method='ffill')  # Forward fill
df.fillna(method='bfill')  # Backward fill

# Interpolation
df.interpolate(method='linear')  # Linear interpolation
df.interpolate(method='polynomial', order=2)  # Polynomial

# Multiple Imputation (MICE)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
df_imputed = imputer.fit_transform(df)

# Add missing indicator
df['col_missing'] = df['col'].isnull().astype(int)
```

---

## 3. Handling Outliers

### Detection Methods

```python
from scipy import stats

# Z-score method
z_scores = np.abs(stats.zscore(df['column']))
outliers = np.where(z_scores > 3)[0]

# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]

# Isolation Forest
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = clf.fit_predict(df[['column']])
outliers = df[outlier_labels == -1]

# DBSCAN for outlier detection
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(df[['column1', 'column2']])
outliers = df[labels == -1]
```

### Treatment Methods

```python
# Capping/Winsorizing
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1

df['column_capped'] = df['column'].clip(
    lower=Q1 - 1.5 * IQR,
    upper=Q3 + 1.5 * IQR
)

# Transformation
df['column_log'] = np.log1p(df['column'])  # Log transform
df['column_sqrt'] = np.sqrt(df['column'])  # Square root
df['column_boxcox'], _ = stats.boxcox(df['column'] + 1)  # Box-Cox

# Binning
df['column_binned'] = pd.cut(
    df['column'],
    bins=[0, 25, 50, 75, 100],
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

# Removal (use cautiously)
df_clean = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]
```

---

## 4. Data Transformation

### Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# StandardScaler (Z-score normalization)
# Result: mean=0, std=1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# MinMaxScaler
# Result: range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# RobustScaler (robust to outliers)
# Uses median and IQR
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)

# MaxAbsScaler
# Result: range [-1, 1] if data has both positive and negative
scaler = MaxAbsScaler()
df_scaled = scaler.fit_transform(df)

# When to use:
# - StandardScaler: Most cases, especially for neural networks
# - MinMaxScaler: When bounds are important
# - RobustScaler: When outliers are present
# - MaxAbsScaler: When data is already centered at 0
```

### Power Transformations

```python
from sklearn.preprocessing import PowerTransformer

# Log transform (for right-skewed data)
df['col_log'] = np.log1p(df['col'])

# Box-Cox (positive values only)
pt = PowerTransformer(method='box-cox')
df['col_transformed'] = pt.fit_transform(df[['col']])

# Yeo-Johnson (handles negative values)
pt = PowerTransformer(method='yeo-johnson')
df['col_transformed'] = pt.fit_transform(df[['col']])
```

---

## 5. Encoding Categorical Variables

### Nominal Encoding (No Order)

```python
# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Or using sklearn
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
df_encoded = ohe.fit_transform(df[['category']])

# Drop first to avoid multicollinearity
ohe = OneHotEncoder(drop='first')
df_encoded = ohe.fit_transform(df[['category']])
```

### Ordinal Encoding (With Order)

```python
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# OrdinalEncoder (for features)
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['level_encoded'] = oe.fit_transform(df[['level']])

# LabelEncoder (for target variable)
le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['target'])
```

### Advanced Encoding

```python
import category_encoders as ce

# Binary Encoding
encoder = ce.BinaryEncoder(cols=['category'])
df_encoded = encoder.fit_transform(df)

# Target/Mean Encoding
encoder = ce.TargetEncoder(cols=['category'])
df_encoded = encoder.fit_transform(df, y=df['target'])

# Frequency Encoding
freq_encoder = df['category'].value_counts().to_dict()
df['category_freq'] = df['category'].map(freq_encoder)

# Hashing Trick
from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=10, input_type='string')
df_hashed = fh.transform(df['category'].astype(str))
```

---

## 6. Feature Scaling and Normalization

### When to Scale

```
Scale when using:
✓ Distance-based algorithms (KNN, K-Means, SVM)
✓ Neural Networks
✓ PCA
✓ Gradient Descent optimization
✓ Regularization (L1, L2)

Don't scale when using:
✗ Tree-based models (Decision Trees, Random Forests, XGBoost)
✗ Naive Bayes
```

### Normalization Techniques

```python
# L1 Normalization (Manhattan)
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l1')
df_normalized = normalizer.fit_transform(df)

# L2 Normalization (Euclidean)
normalizer = Normalizer(norm='l2')
df_normalized = normalizer.fit_transform(df)

# Pipeline example
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

---

## 7. Data Integration

### Combining Multiple Sources

```python
# Merge DataFrames
df_merged = pd.merge(df1, df2, on='key', how='inner')
df_merged = pd.merge(df1, df2, left_on='key1', right_on='key2', how='left')

# Concatenate
df_combined = pd.concat([df1, df2], axis=0)  # Stack vertically
df_combined = pd.concat([df1, df2], axis=1)  # Stack horizontally

# Join (index-based)
df_joined = df1.join(df2, how='left')
```

### Handling Redundancy

```python
# Find and remove duplicates
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=['col1', 'col2'], inplace=True)

# Find correlated features
corr_matrix = df.corr()
high_corr = np.where(np.abs(corr_matrix) > 0.9)

# Remove highly correlated features
to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            to_drop.add(corr_matrix.columns[i])

df_reduced = df.drop(columns=to_drop)
```

---

## 8. Data Reduction

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Filter methods
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Mutual Information
mi_scores = mutual_info_classif(X, y)

# Wrapper methods (RFE)
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X, y)
selected_features = rfe.support_

# Embedded methods
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

### Sampling Techniques

```python
# Random sampling
df_sample = df.sample(n=1000, random_state=42)
df_sample = df.sample(frac=0.1, random_state=42)  # 10% of data

# Stratified sampling (maintain class distribution)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Reservoir sampling (for streaming data)
import random

def reservoir_sampling(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
```

---

## 💻 Python Code Examples

```python
# === Complete Preprocessing Pipeline ===

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def create_preprocessing_pipeline(df, target_col):
    """Create complete preprocessing pipeline"""
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor, X, y

# Usage
preprocessor, X, y = create_preprocessing_pipeline(df, 'target')
X_processed = preprocessor.fit_transform(X)

# === Outlier Detection and Treatment ===

def handle_outliers(df, columns, method='iqr'):
    """Detect and treat outliers"""
    
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < 3]
    
    return df_clean

# === Missing Data Analysis ===

def analyze_missing_data(df):
    """Comprehensive missing data analysis"""
    
    # Count and percentage
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    
    # Create report
    report = pd.DataFrame({
        'missing_count': missing_count,
        'missing_pct': missing_pct,
        'dtype': df.dtypes
    })
    
    # Sort by missing percentage
    report = report[report['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    # Recommendations
    report['recommendation'] = report['missing_pct'].apply(lambda x: 
        'Drop column' if x > 50 else
        'Impute' if x > 5 else
        'Drop rows' if x > 0 else
        'OK'
    )
    
    return report

# Usage
missing_report = analyze_missing_data(df)
print(missing_report)
```

---

## 📊 Summary Tables

### Missing Data Handling

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| Deletion | < 5% missing, MCAR | Simple | Loss of data |
| Mean/Median | Numeric, MAR | Simple | Reduces variance |
| KNN Imputation | Numeric, MAR | Preserves relationships | Computationally expensive |
| MICE | Complex patterns | Most accurate | Complex, slow |

### Encoding Methods

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| One-Hot | Nominal, few categories | Simple, interpretable | Curse of dimensionality |
| Label | Ordinal | Preserves order | Assumes ordering |
| Target | High cardinality | Captures relationship | Risk of overfitting |
| Binary | High cardinality | Efficient | Less interpretable |

### Scaling Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| Standard | (x - mean) / std | Most cases |
| MinMax | (x - min) / (max - min) | Neural networks, bounds important |
| Robust | (x - median) / IQR | With outliers |
| L1/L2 Norm | x / ||x|| | Text classification, clustering |

---

## 🎯 ML Applications

| Preprocessing Step | ML Application |
|-------------------|----------------|
| Missing Data Imputation | Complete feature matrices |
| Outlier Treatment | Robust models |
| Scaling | Distance-based algorithms |
| Encoding | Categorical features |
| Feature Selection | Dimensionality reduction |

---

## ❓ Quick Check Questions

1. What is the difference between MCAR and MNAR missing data?
2. When should you use RobustScaler over StandardScaler?
3. Why might you choose not to scale features when using a Random Forest model?
4. What is the "Curse of Dimensionality" in the context of One-Hot Encoding?
5. How does Stratified Sampling differ from Random Sampling?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **MCAR (Missing Completely At Random)** means the missing data has no relationship to any other data (observed or unobserved). **MNAR (Missing Not At Random)** means the missingness is related to the unobserved value itself (e.g., people with very high debt are less likely to report it).
2. Use **RobustScaler** when your dataset contains significant outliers. It uses the median and IQR instead of the mean and variance, making it insensitive to extreme values.
3. **Tree-based models** (like Random Forest) partition the feature space using orthogonal splits (greater than/less than specific values). The absolute scale of the feature does not affect how the tree makes these splits.
4. **Curse of Dimensionality** occurs when One-Hot Encoding a categorical feature with many unique categories creates too many new columns (dimensions), making the dataset sparse, computationally expensive to process, and prone to overfitting.
5. **Random Sampling** selects items completely by chance. **Stratified Sampling** ensures that the sample maintains the same proportions of subgroups (classes) as the original dataset, which is crucial for imbalanced classification tasks.

</details>
---

**Status:** ✅ Complete
**Next:** ETL Concepts
