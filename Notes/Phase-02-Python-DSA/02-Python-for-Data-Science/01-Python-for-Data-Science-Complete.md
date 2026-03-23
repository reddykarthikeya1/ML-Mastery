# 3.1-3.4 Python for Data Science - Complete Reference

## 🎯 Quick Overview
- **Jupyter**: Interactive computing environment
- **NumPy**: Numerical computing with arrays
- **Pandas**: Data manipulation and analysis
- **Visualization**: Matplotlib and Seaborn for plotting

---

## Part 1: Jupyter Notebooks (3.1)

### 1.1 Jupyter Basics

#### What is Jupyter?
```
Jupyter = Interactive web-based notebook for code, visualizations, and text

Components:
- Jupyter Notebook: Original interface
- JupyterLab: Next-gen interface with tabs, terminals
- Jupyter Kernel: Executes code
```

#### Installation and Setup
```bash
# Install Jupyter
pip install jupyter

# Install JupyterLab
pip install jupyterlab

# Start notebook
jupyter notebook

# Start JupyterLab
jupyter lab
```

#### Cell Types
```
1. Code cells: Execute Python code (Shift+Enter)
2. Markdown cells: Formatted text
3. Raw cells: Unformatted text
```

#### Magic Commands
```python
# Line magics (single line)
%timeit sum(range(1000))      # Time execution
%matplotlib inline            # Display plots inline
%load script.py               # Load code from file
%run script.py                # Run script
%who                          # List variables

# Cell magics (entire cell)
%%time
for i in range(1000):
    pass

%%writefile output.txt
Content to write to file
```

---

## Part 2: NumPy (3.2)

### 2.1 NumPy Arrays

#### Creating Arrays
```python
import numpy as np

# From lists
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# Built-in functions
np.zeros((3, 4))           # Array of zeros
np.ones((2, 3))            # Array of ones
np.full((2, 2), 7)         # Array filled with value
np.eye(3)                  # Identity matrix
np.arange(0, 10, 2)        # Like range: [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)       # Evenly spaced: [0, 0.25, 0.5, 0.75, 1]

# Random arrays
np.random.rand(3, 3)       # Uniform [0, 1)
np.random.randn(3, 3)      # Standard normal
np.random.randint(0, 10, (3, 3))  # Random integers
np.random.choice([1, 2, 3], 5)    # Random choice
```

#### Array Attributes
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.ndim      # 2 (number of dimensions)
arr.shape     # (2, 3) (dimensions)
arr.size      # 6 (total elements)
arr.dtype     # dtype('int64') (data type)
arr.itemsize  # 8 (bytes per element)
arr.nbytes    # 48 (total bytes)
```

#### Indexing and Slicing
```python
arr = np.array([1, 2, 3, 4, 5])

# Basic indexing
arr[0]      # 1
arr[-1]     # 5
arr[1:4]    # [2, 3, 4]

# 2D indexing
matrix = np.array([[1, 2, 3], [4, 5, 6]])
matrix[0, 1]    # 2
matrix[:, 1]    # [2, 5] (second column)
matrix[0, :]    # [1, 2, 3] (first row)

# Boolean indexing
arr[arr > 3]        # [4, 5]
arr[arr % 2 == 0]   # [2, 4]

# Fancy indexing
arr[[0, 2, 4]]  # [1, 3, 5]
```

---

### 2.2 Array Operations

#### Element-wise Operations
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b      # [5, 7, 9]
a * b      # [4, 10, 18]
a ** 2     # [1, 4, 9]
np.sqrt(a) # [1, 1.414, 1.732]
```

#### Broadcasting
```python
# Broadcasting rules:
# 1. Align shapes from right
# 2. Dimensions must be equal or one must be 1

a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
b = np.array([10, 20, 30])             # Shape (3,)

a + b  # [[11, 22, 33], [14, 25, 36]]
# b is broadcast to [[10, 20, 30], [10, 20, 30]]
```

#### Aggregation Functions
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

np.sum(arr)        # 21
np.mean(arr)       # 3.5
np.std(arr)        # Standard deviation
np.min(arr)        # 1
np.max(arr)        # 6
np.argmax(arr)     # Index of max (5)
np.argmin(arr)     # Index of min (0)

# Axis parameter
np.sum(arr, axis=0)  # [5, 7, 9] (column sums)
np.sum(arr, axis=1)  # [6, 15] (row sums)
```

---

### 2.3 Array Manipulation

#### Reshaping
```python
arr = np.arange(12)  # [0, 1, ..., 11]

arr.reshape(3, 4)     # Reshape to 3x4
arr.ravel()           # Flatten to 1D
arr.flatten()         # Flatten (returns copy)
arr.T                 # Transpose
```

#### Stacking and Splitting
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Stacking
np.vstack([a, b])    # [[1, 2, 3], [4, 5, 6]]
np.hstack([a, b])    # [1, 2, 3, 4, 5, 6]
np.stack([a, b])     # Stack along new axis

# Splitting
arr = np.arange(10)
np.split(arr, 5)     # Split into 5 equal parts
np.hsplit(arr, 5)    # Horizontal split
```

---

### 2.4 Linear Algebra with NumPy

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
A @ B           # Matrix multiply
np.dot(A, B)    # Same as @
A.dot(B)        # Method form

# Linear algebra functions
np.linalg.det(A)     # Determinant
np.linalg.inv(A)     # Inverse
np.linalg.trace(A)   # Trace (sum of diagonal)
np.linalg.norm(A)    # Frobenius norm

# Eigenvalues/eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solve linear system: Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)  # [1, 2]

# SVD
U, S, Vh = np.linalg.svd(A)
```

---

## Part 3: Pandas (3.3)

### 3.1 Series and DataFrame Basics

#### Creating Series
```python
import pandas as pd

# From list
s = pd.Series([1, 2, 3, 4])

# With custom index
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# From dictionary
s = pd.Series({'a': 1, 'b': 2, 'c': 3})
```

#### Creating DataFrames
```python
# From dictionary of lists
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# From list of dictionaries
df = pd.DataFrame([
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
])

# From CSV
df = pd.read_csv('data.csv')

# From Excel
df = pd.read_excel('data.xlsx')
```

#### DataFrame Attributes
```python
df.shape      # (rows, columns)
df.columns    # Column names
df.index      # Index
df.dtypes     # Data types
df.head()     # First 5 rows
df.tail()     # Last 5 rows
df.info()     # Summary info
df.describe() # Statistical summary
```

---

### 3.2 Selecting and Indexing

#### Selecting Columns
```python
df['name']      # Single column (Series)
df[['name', 'age']]  # Multiple columns (DataFrame)
```

#### Selecting Rows
```python
# By label
df.loc[0]        # Row with label 0
df.loc[0:2]      # Rows 0 to 2 (inclusive)

# By position
df.iloc[0]       # First row
df.iloc[0:3]     # First 3 rows

# Boolean indexing
df[df['age'] > 30]
df[(df['age'] > 25) & (df['city'] == 'NYC')]
```

#### Setting Values
```python
df['salary'] = [50000, 60000, 70000]  # New column
df.loc[0, 'age'] = 26                  # Single value
df.loc[df['age'] > 30, 'city'] = 'SF'  # Conditional
```

---

### 3.3 Data Cleaning

#### Handling Missing Data
```python
df.isna()           # Check for missing values
df.isna().sum()     # Count missing per column

# Remove missing
df.dropna()         # Drop rows with any NA
df.dropna(axis=1)   # Drop columns with any NA

# Fill missing
df.fillna(0)                    # Fill with 0
df.fillna(df.mean())            # Fill with mean
df.fillna(method='ffill')       # Forward fill
df.fillna(method='bfill')       # Backward fill
```

#### Handling Duplicates
```python
df.duplicated()         # Check for duplicates
df.drop_duplicates()    # Remove duplicates
```

#### Data Type Conversion
```python
df['age'] = df['age'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df['value'] = pd.to_numeric(df['value'], errors='coerce')
```

---

### 3.4 Data Transformation

#### Sorting
```python
df.sort_values('age')           # Sort by column
df.sort_values(['age', 'name']) # Sort by multiple
df.sort_index()                 # Sort by index
```

#### Applying Functions
```python
# Apply to column
df['age_squared'] = df['age'].apply(lambda x: x**2)

# Apply to DataFrame
df.apply(np.mean)  # Apply to each column

# Map (Series only)
df['city_code'] = df['city'].map({'NYC': 1, 'LA': 2, 'Chicago': 3})

# Replace
df.replace('NYC', 'New York')
```

#### Binning and Encoding
```python
# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 65, 100],
                         labels=['Child', 'Young', 'Adult', 'Senior'])

# One-hot encoding
pd.get_dummies(df['city'])
```

---

### 3.5 Data Aggregation (GroupBy)

#### GroupBy Operations
```python
# Split-Apply-Combine
df.groupby('city')['age'].mean()      # Mean age per city
df.groupby('city').agg(['mean', 'sum'])  # Multiple aggregations

# Multiple columns
df.groupby(['city', 'gender'])['salary'].mean()

# Custom aggregation
df.groupby('city').agg({
    'age': ['mean', 'min', 'max'],
    'salary': 'sum'
})
```

#### Transform and Filter
```python
# Transform (returns same shape)
df['age_zscore'] = df.groupby('city')['age'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Filter
df.groupby('city').filter(lambda x: x['age'].mean() > 30)
```

#### Pivot Tables
```python
pd.pivot_table(df, values='salary', index='city', 
               columns='gender', aggfunc='mean')

# Cross-tabulation
pd.crosstab(df['city'], df['gender'])
```

---

### 3.6 Merging and Joining

#### Concatenation
```python
pd.concat([df1, df2], axis=0)  # Stack vertically
pd.concat([df1, df2], axis=1)  # Stack horizontally
```

#### Merging (SQL-style joins)
```python
pd.merge(df1, df2, on='key')           # Inner join
pd.merge(df1, df2, on='key', how='left')   # Left join
pd.merge(df1, df2, on='key', how='right')  # Right join
pd.merge(df1, df2, on='key', how='outer')  # Outer join
```

#### Join (index-based)
```python
df1.join(df2, how='left')
```

---

### 3.7 Time Series

#### Creating Datetime
```python
pd.to_datetime(['2024-01-01', '2024-01-02'])
pd.date_range('2024-01-01', periods=10, freq='D')
```

#### Time-based Indexing
```python
df['2024-01']  # All January 2024 data
df['2024']     # All 2024 data
```

#### Resampling
```python
df.resample('M').mean()  # Monthly mean
df.resample('W').sum()   # Weekly sum
```

#### Rolling Windows
```python
df.rolling(window=7).mean()   # 7-day rolling mean
df.expanding().mean()         # Expanding mean
```

---

## Part 4: Data Visualization (3.4)

### 4.1 Matplotlib Fundamentals

#### Basic Plotting
```python
import matplotlib.pyplot as plt

# Line plot
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Title')
plt.show()

# Scatter plot
plt.scatter(x, y)

# Bar plot
plt.bar(['A', 'B', 'C'], [1, 2, 3])

# Histogram
plt.hist(data, bins=20)

# Box plot
plt.boxplot(data)
```

#### Multiple Subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].bar(categories, values)
axes[1, 1].hist(data)

plt.tight_layout()
plt.show()
```

#### Customization
```python
plt.plot(x, y, color='red', linestyle='--', linewidth=2, 
         marker='o', markersize=5, label='Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

---

### 4.2 Seaborn

#### Distribution Plots
```python
import seaborn as sns

sns.histplot(data, bins=20)
sns.kdeplot(data)
sns.boxplot(data)
sns.violinplot(data)
```

#### Categorical Plots
```python
sns.barplot(x='category', y='value', data=df)
sns.countplot(x='category', data=df)
sns.stripplot(x='category', y='value', data=df)
sns.boxplot(x='category', y='value', data=df)
```

#### Relational Plots
```python
sns.scatterplot(x='x', y='y', hue='category', data=df)
sns.lineplot(x='time', y='value', hue='group', data=df)
```

#### Matrix Plots
```python
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
sns.clustermap(data)  # Clustered heatmap
```

#### Pair Plots
```python
sns.pairplot(df, hue='category')
```

---

## 💻 Python Code Examples

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Complete Data Analysis Example ===

# Generate sample data
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.normal(50000, 15000, n),
    'score': np.random.normal(75, 10, n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n),
    'gender': np.random.choice(['M', 'F'], n)
})

# Data cleaning
df['income'] = df['income'].clip(lower=0)  # Remove negative incomes
df = df.dropna()

# Descriptive statistics
print(df.describe())

# Group analysis
group_stats = df.groupby('city').agg({
    'age': 'mean',
    'income': ['mean', 'std'],
    'score': 'mean'
})
print(group_stats)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
sns.histplot(df['age'], bins=20, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution')

# Income by city
sns.boxplot(x='city', y='income', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Income by City')

# Age vs Income scatter
sns.scatterplot(x='age', y='income', hue='gender', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Age vs Income')

# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()
```

---

## 📊 Summary Tables

### NumPy Functions

| Function | Purpose | Example |
|----------|---------|---------|
| np.array() | Create array | np.array([1,2,3]) |
| np.zeros() | Array of zeros | np.zeros((3,3)) |
| np.arange() | Range array | np.arange(0,10,2) |
| np.linspace() | Evenly spaced | np.linspace(0,1,5) |
| np.sum() | Sum | np.sum(arr, axis=0) |
| np.mean() | Mean | np.mean(arr) |
| np.std() | Standard deviation | np.std(arr) |

### Pandas Operations

| Operation | Method | Example |
|-----------|--------|---------|
| Read CSV | pd.read_csv() | pd.read_csv('file.csv') |
| Select column | df['col'] | df['name'] |
| Filter rows | df[condition] | df[df['age']>30] |
| Group by | df.groupby() | df.groupby('city').mean() |
| Merge | pd.merge() | pd.merge(df1, df2, on='key') |
| Handle NA | df.fillna() | df.fillna(0) |

### Visualization Types

| Plot Type | Function | Use Case |
|-----------|----------|----------|
| Line | plt.plot() | Trends over time |
| Scatter | plt.scatter() | Relationship between variables |
| Bar | plt.bar() | Compare categories |
| Histogram | plt.hist() | Distribution |
| Box | plt.boxplot() | Distribution and outliers |
| Heatmap | sns.heatmap() | Correlation matrix |

---

## 🎯 ML Applications

| Tool | ML Application |
|------|---------------|
| NumPy | Feature matrices, numerical operations |
| Pandas | Data preprocessing, feature engineering |
| Matplotlib | Model evaluation plots |
| Seaborn | Exploratory data analysis |

---

**Status:** ✅ Complete
**Next:** Data Engineering (SQL, NoSQL, ETL)
