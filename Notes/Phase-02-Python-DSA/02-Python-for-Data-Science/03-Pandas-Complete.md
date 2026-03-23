# 3.3 Pandas

## 🎯 Quick Overview
- **Series**: 1D labeled array
- **DataFrame**: 2D labeled table
- **Data manipulation**: Clean, transform, analyze
- **Foundation for**: Data preprocessing in ML

---

## 1. Series

### Creating Series

```python
import pandas as pd

# From list
s = pd.Series([1, 2, 3, 4])

# With custom index
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# From dictionary
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# Attributes
s.values    # Array of values
s.index     # Index object
s.dtype     # Data type
```

---

## 2. DataFrame Basics

### Creating DataFrames

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

# Other formats
df = pd.read_json('data.json')
df = pd.read_sql('SELECT * FROM table', connection)
df = pd.read_parquet('data.parquet')
```

### DataFrame Attributes

```python
df.shape      # (rows, columns)
df.columns    # Column names
df.index      # Index
df.dtypes     # Data types
df.head()     # First 5 rows
df.tail()     # Last 5 rows
df.info()     # Summary info
df.describe() # Statistical summary
df.sample(5)  # Random 5 rows
```

---

## 3. Selecting and Indexing

### Selecting Columns

```python
df['name']      # Single column (Series)
df[['name', 'age']]  # Multiple columns (DataFrame)
```

### Selecting Rows

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
df[df['name'].isin(['Alice', 'Bob'])]
```

### Setting Values

```python
df['salary'] = [50000, 60000, 70000]  # New column
df.loc[0, 'age'] = 26                  # Single value
df.loc[df['age'] > 30, 'city'] = 'SF'  # Conditional
```

---

## 4. Data Cleaning

### Handling Missing Data

```python
df.isna()           # Check for missing values
df.isna().sum()     # Count missing per column

# Remove missing
df.dropna()         # Drop rows with any NA
df.dropna(axis=1)   # Drop columns with any NA
df.dropna(thresh=3) # Keep rows with at least 3 non-NA

# Fill missing
df.fillna(0)                    # Fill with 0
df.fillna(df.mean())            # Fill with mean
df.fillna(method='ffill')       # Forward fill
df.fillna(method='bfill')       # Backward fill
df.interpolate()                # Interpolate
```

### Handling Duplicates

```python
df.duplicated()         # Check for duplicates
df.drop_duplicates()    # Remove duplicates
df.drop_duplicates(subset=['name'])  # By column
```

### Data Type Conversion

```python
df['age'] = df['age'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['category'] = df['type'].astype('category')
```

---

## 5. Data Transformation

### Sorting

```python
df.sort_values('age')           # Sort by column
df.sort_values(['age', 'name']) # Sort by multiple
df.sort_values('age', ascending=False)
df.sort_index()                 # Sort by index
```

### Applying Functions

```python
# Apply to column
df['age_squared'] = df['age'].apply(lambda x: x**2)

# Apply to DataFrame
df.apply(np.mean)  # Apply to each column

# Apply row-wise
df.apply(lambda row: row['age'] + row['salary'], axis=1)

# Map (Series only)
df['city_code'] = df['city'].map({'NYC': 1, 'LA': 2, 'Chicago': 3})

# Replace
df.replace('NYC', 'New York')
df.replace([1, 2, 3], ['one', 'two', 'three'])
```

### Binning and Encoding

```python
# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 65, 100],
                         labels=['Child', 'Young', 'Adult', 'Senior'])

# One-hot encoding
pd.get_dummies(df['city'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])
```

---

## 6. Data Aggregation (GroupBy)

### GroupBy Operations

```python
# Split-Apply-Combine
df.groupby('city')['age'].mean()      # Mean age per city
df.groupby('city').agg(['mean', 'sum'])  # Multiple aggregations

# Multiple columns
df.groupby(['city', 'gender'])['salary'].mean()

# Multiple aggregations
df.groupby('city').agg({
    'age': ['mean', 'min', 'max'],
    'salary': 'sum'
})

# Named aggregations (pandas 0.25+)
df.groupby('city').agg(
    avg_age=('age', 'mean'),
    total_salary=('salary', 'sum')
)
```

### Transform and Filter

```python
# Transform (returns same shape)
df['age_zscore'] = df.groupby('city')['age'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Filter
df.groupby('city').filter(lambda x: x['age'].mean() > 30)
```

### Pivot Tables

```python
pd.pivot_table(df, values='salary', index='city', 
               columns='gender', aggfunc='mean')

# Cross-tabulation
pd.crosstab(df['city'], df['gender'])
```

---

## 7. Merging and Joining

### Concatenation

```python
pd.concat([df1, df2], axis=0)  # Stack vertically
pd.concat([df1, df2], axis=1)  # Stack horizontally
pd.concat([df1, df2], ignore_index=True)  # Reset index
```

### Merging (SQL-style joins)

```python
pd.merge(df1, df2, on='key')           # Inner join
pd.merge(df1, df2, on='key', how='left')   # Left join
pd.merge(df1, df2, on='key', how='right')  # Right join
pd.merge(df1, df2, on='key', how='outer')  # Outer join

# Different column names
pd.merge(df1, df2, left_on='key1', right_on='key2')

# Multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])
```

### Join (index-based)

```python
df1.join(df2, how='left')
df1.join(df2, on='key', how='inner')
```

---

## 8. Time Series

### Creating Datetime

```python
pd.to_datetime(['2024-01-01', '2024-01-02'])
pd.date_range('2024-01-01', periods=10, freq='D')
pd.date_range('2024-01-01', '2024-12-31', freq='M')
```

### Time-based Indexing

```python
df['2024-01']  # All January 2024 data
df['2024']     # All 2024 data
df.loc['2024-01-01':'2024-01-31']
```

### Resampling

```python
df.resample('M').mean()  # Monthly mean
df.resample('W').sum()   # Weekly sum
df.resample('Q', on='date').mean()  # Quarterly
```

### Rolling Windows

```python
df.rolling(window=7).mean()   # 7-day rolling mean
df.expanding().mean()         # Expanding mean
df.ewm(span=7).mean()         # Exponential weighted mean
```

---

## 9. Advanced Pandas

### MultiIndex

```python
# Create MultiIndex
df.set_index(['city', 'year'])

# Access
df.loc['NYC']
df.loc['NYC', 2024]

# Reset index
df.reset_index()
```

### Categorical Data

```python
df['category'] = df['type'].astype('category')
df['category'].cat.categories  # Get categories
df['category'].cat.codes       # Get codes
```

### Performance Tips

```python
# Use vectorized operations
df['new'] = df['a'] + df['b']  # Fast

# Avoid iterrows
# Bad:
for idx, row in df.iterrows():
    df.loc[idx, 'new'] = row['a'] + row['b']

# Good:
df['new'] = df['a'] + df['b']

# Use eval for complex expressions
df.eval('new = a + b * c')

# Use appropriate dtypes
df['category'] = df['type'].astype('category')
```

---

## 💻 Python Code Examples

```python
import pandas as pd
import numpy as np

# === Example 1: Complete Data Analysis ===

# Load data
df = pd.read_csv('sales.csv')

# Explore
print(df.shape)
print(df.info())
print(df.describe())

# Clean
df = df.dropna()
df = df.drop_duplicates()

# Transform
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['revenue'] = df['price'] * df['quantity']

# Aggregate
monthly_sales = df.groupby('month')['revenue'].sum()

# Merge
products = pd.read_csv('products.csv')
df = df.merge(products, on='product_id', how='left')

# === Example 2: Time Series Analysis ===

# Create time series data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
sales = np.random.randn(365).cumsum() + 100

df = pd.DataFrame({'date': dates, 'sales': sales})
df.set_index('date', inplace=True)

# Resample
monthly = df.resample('M').mean()
weekly = df.resample('W').sum()

# Rolling average
df['rolling_7d'] = df.rolling(window=7).mean()

# Seasonality
df['day_of_week'] = df.index.dayofweek
daily_pattern = df.groupby('day_of_week')['sales'].mean()

# === Example 3: Data Pipeline ===

def process_data(input_file, output_file):
    """Complete data processing pipeline"""
    
    # Load
    df = pd.read_csv(input_file)
    
    # Validate
    assert df.shape[0] > 0, "Empty file"
    assert 'id' in df.columns, "Missing id column"
    
    # Clean
    df = df.dropna(subset=['id', 'value'])
    df = df.drop_duplicates(subset=['id'])
    
    # Transform
    df['value_normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
    df['category'] = pd.cut(df['value'], bins=5, labels=['A', 'B', 'C', 'D', 'E'])
    
    # Aggregate
    summary = df.groupby('category').agg({
        'value': ['mean', 'std', 'count']
    })
    
    # Save
    df.to_csv(output_file, index=False)
    summary.to_csv(output_file.replace('.csv', '_summary.csv'))
    
    return df, summary

# === Example 4: Advanced GroupBy ===

# Multiple groupby operations
df.groupby(['city', 'category']).agg({
    'sales': ['sum', 'mean', 'count'],
    'profit': 'sum'
}).round(2)

# Custom aggregation
def range(x):
    return x.max() - x.min()

df.groupby('city')['sales'].agg(['mean', range])

# Groupby with transform
df['sales_pct_of_city'] = df.groupby('city')['sales'].transform(
    lambda x: x / x.sum()
)
```

---

## 📊 Summary Tables

### DataFrame Operations

| Operation | Method | Example |
|-----------|--------|---------|
| Read CSV | pd.read_csv() | pd.read_csv('file.csv') |
| Select column | df['col'] | df['name'] |
| Filter rows | df[condition] | df[df['age']>30] |
| Group by | df.groupby() | df.groupby('city').mean() |
| Merge | pd.merge() | pd.merge(df1, df2, on='key') |
| Handle NA | df.fillna() | df.fillna(0) |

### GroupBy Aggregations

| Aggregation | Method | Example |
|-------------|--------|---------|
| Sum | .sum() | df.groupby('city').sum() |
| Mean | .mean() | df.groupby('city').mean() |
| Count | .count() | df.groupby('city').count() |
| Multiple | .agg() | df.groupby('city').agg(['sum', 'mean']) |
| Custom | .agg() | df.groupby('city').agg(lambda x: x.max()-x.min()) |

### Time Series

| Operation | Method | Example |
|-----------|--------|---------|
| Create range | pd.date_range() | pd.date_range('2024-01-01', periods=10) |
| Resample | .resample() | df.resample('M').mean() |
| Rolling | .rolling() | df.rolling(7).mean() |
| Shift | .shift() | df.shift(1) |
| Diff | .diff() | df.diff() |

---

## 🎯 ML Applications

| Pandas Feature | ML Application |
|----------------|----------------|
| DataFrame | Feature storage |
| GroupBy | Feature aggregation |
| merge/join | Data integration |
| get_dummies | One-hot encoding |
| Time series | Temporal features |
| fillna | Imputation |

---

---

## ❓ Quick Check Questions

1. What is the difference between a Pandas Series and a DataFrame?
2. How do `loc` and `iloc` differ in how they select data?
3. What is the difference between `df.dropna()` and `df.fillna()`?
4. Explain the "Split-Apply-Combine" pattern in the context of `groupby()`.
5. Which Pandas function is used to perform a SQL-style join between two DataFrames?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. A **Series** is a 1D labeled array capable of holding any data type. A **DataFrame** is a 2D labeled data structure (like a table) with columns of potentially different types. You can think of a DataFrame as a dictionary of Series objects.
2. **`loc`** is **label-based**, meaning you use the names of the rows and columns to select data. **`iloc`** is **integer-position based**, meaning you use the numerical index (starting from 0) to select data.
3. **`df.dropna()`** removes any rows (or columns) that contain missing values. **`df.fillna()`** replaces missing values with a specific value (like 0, the mean, or a string).
4. The **Split-Apply-Combine** pattern involves: **Splitting** the data into groups based on some criteria (e.g., city), **Applying** a function to each group independently (e.g., calculating the mean), and **Combining** the results back into a single data structure.
5. **`pd.merge()`** is the primary function used for database-style joins (inner, left, right, and outer) between two DataFrames based on a common key.

</details>
---

**Status:** ✅ Complete
**Next:** Data Visualization
