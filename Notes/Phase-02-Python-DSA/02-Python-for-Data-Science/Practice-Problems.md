# Python for Data Science - Practice Problems

## Topic 1: Jupyter Notebooks

### Level 1: Basic

**1.1** Create a notebook with:
- Markdown cell with headings, lists, and a table
- Code cell that prints "Hello, Jupyter!"
- Use %timeit to time a simple operation

**1.2** Practice magic commands:
- Use %%writefile to write content to a file
- Use %load to load the file back
- Use %run to execute a Python script

---

## Topic 2: NumPy

### Level 1: Basic

**2.1** Array creation:
```python
# Create:
# 1. Array of zeros (5x5)
# 2. Array of ones (3x4)
# 3. Identity matrix (4x4)
# 4. Array with values 0-20
# 5. Array with 10 evenly spaced values from 0 to 100
```

**2.2** Array operations:
```python
arr = np.array([1, 2, 3, 4, 5])

# 1. Square all elements
# 2. Find sum
# 3. Find mean
# 4. Find standard deviation
# 5. Normalize (subtract mean, divide by std)
```

### Level 2: Intermediate

**2.3** Matrix operations:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1. Matrix multiplication
# 2. Element-wise multiplication
# 3. Find inverse of A
# 4. Find determinant of A
# 5. Solve Ax = b where b = [5, 11]
```

**2.4** Broadcasting practice:
```python
# Create a 5x5 matrix where each row is [0, 1, 2, 3, 4]
# Hint: Use broadcasting with np.arange(5)
```

### Level 3: Advanced

**2.5** NumPy Practice:
```python
def normalize_matrix(matrix):
    """Normalize each row of a matrix"""
    # Your code here
    pass

def find_outliers(data, threshold=3):
    """Find outliers using z-score"""
    # Your code here
    pass
```

---

## Topic 3: Pandas

### Level 1: Basic

**3.1** Create a DataFrame:
```python
# Create DataFrame with columns: name, age, city
# Add 5 rows of data
# Display first 3 rows
# Display summary statistics
```

**3.2** Data selection:
```python
# Given df:
# 1. Select 'name' column
# 2. Select rows where age > 25
# 3. Select name and city for people over 30
# 4. Set index to 'name'
```

### Level 2: Intermediate

**3.3** Data cleaning:
```python
# Given a DataFrame with missing values:
# 1. Count missing values per column
# 2. Fill numeric missing with mean
# 3. Drop rows with any missing
# 4. Remove duplicates
```

**3.4** GroupBy operations:
```python
# Given sales data with columns: date, product, region, sales
# 1. Group by region, find total sales
# 2. Group by product, find mean sales
# 3. Group by region and product, find sum and mean
# 4. Add a column with sales as percentage of region total
```

### Level 3: Advanced

**3.5** Time series analysis:
```python
def analyze_time_series(df, date_col, value_col):
    """
    Perform time series analysis:
    1. Convert date column to datetime
    2. Set as index
    3. Resample to monthly
    4. Calculate rolling 7-day average
    5. Find month with highest average
    """
    # Your code here
    pass
```

**3.6** Data pipeline:
```python
def process_sales_data(input_file, output_file):
    """
    Complete data processing pipeline:
    1. Load CSV
    2. Handle missing values
    3. Create new features (month, year, day_of_week)
    4. Aggregate by month
    5. Save to CSV
    """
    # Your code here
    pass
```

---

## Topic 4: Data Visualization

### Level 1: Basic

**4.1** Create basic plots:
```python
# Given data:
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 1. Line plot
# 2. Scatter plot
# 3. Bar chart
# 4. Histogram
```

### Level 2: Intermediate

**4.2** Create a dashboard:
```python
# Create a 2x2 subplot with:
# 1. Line plot (top-left)
# 2. Scatter plot (top-right)
# 3. Bar chart (bottom-left)
# 4. Histogram (bottom-right)
```

**4.3** Seaborn visualizations:
```python
# Using tips dataset:
# 1. Distribution of total_bill (histogram + KDE)
# 2. Box plot of total_bill by day
# 3. Scatter plot of total_bill vs tip
# 4. Correlation heatmap
```

### Level 3: Advanced

**4.4** Complete EDA visualization:
```python
def create_eda_report(df):
    """
    Create comprehensive EDA visualizations:
    1. Distribution plots for all numeric columns
    2. Correlation heatmap
    3. Pair plot for key variables
    4. Save all plots to files
    """
    # Your code here
    pass
```

---

## Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 2.2
```python
arr = np.array([1, 2, 3, 4, 5])

# 1. Square
arr_squared = arr ** 2

# 2. Sum
total = np.sum(arr)

# 3. Mean
mean = np.mean(arr)

# 4. Std
std = np.std(arr)

# 5. Normalize
normalized = (arr - mean) / std
```

### 2.3
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1. Matrix multiplication
A @ B

# 2. Element-wise
A * B

# 3. Inverse
np.linalg.inv(A)

# 4. Determinant
np.linalg.det(A)

# 5. Solve
b = np.array([5, 11])
x = np.linalg.solve(A, b)
```

### 3.4
```python
# Group by region
df.groupby('region')['sales'].sum()

# Group by product
df.groupby('product')['sales'].mean()

# Multiple aggregations
df.groupby(['region', 'product']).agg({
    'sales': ['sum', 'mean']
})

# Percentage of region total
df['sales_pct'] = df.groupby('region')['sales'].transform(
    lambda x: x / x.sum() * 100
)
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
**Status:** ✅ Python for Data Science Complete!
