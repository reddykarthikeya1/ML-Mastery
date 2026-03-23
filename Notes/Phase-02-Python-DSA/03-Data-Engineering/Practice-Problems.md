# Data Engineering - Practice Problems

## Topic 1: SQL Fundamentals

### Level 1: Basic

**1.1** Write SQL queries for:
```sql
-- 1. Select all employees with salary > 50000
-- 2. Count employees per department
-- 3. Find average salary by department
-- 4. List employees hired in 2024
```

**1.2** JOIN practice:
```sql
-- Given tables: employees(id, name, dept_id), departments(id, name)
-- 1. INNER JOIN to get employee with department
-- 2. LEFT JOIN to get all employees with their departments
-- 3. Find employees without departments
```

### Level 2: Intermediate

**2.1** Window functions:
```sql
-- 1. Rank employees by salary within each department
-- 2. Calculate running total of sales
-- 3. Find top 3 salaries per department
```

**2.2** Subqueries and CTEs:
```sql
-- 1. Find employees earning more than department average
-- 2. Using CTE, calculate monthly sales totals
-- 3. Find departments with above-average sales
```

---

## Topic 2: NoSQL

### Level 2: Intermediate

**2.1** MongoDB queries:
```javascript
// 1. Find all users over 25
// 2. Update user's email
// 3. Aggregate: group by city, count users
```

---

## Topic 3: Data Preprocessing

### Level 2: Intermediate

**3.1** Handle missing data:
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, np.nan],
    'B': [5, np.nan, np.nan, 8, 10],
    'C': ['a', 'b', 'c', np.nan, 'e']
})

# 1. Count missing values
# 2. Fill numeric with mean
# 3. Drop rows with any missing
# 4. Forward fill
```

**3.2** Encoding practice:
```python
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue'],
    'size': ['S', 'M', 'L', 'XL'],
    'price': [10, 15, 20, 25]
})

# 1. One-hot encode 'color'
# 2. Label encode 'size' (ordinal)
# 3. Create pipeline with encoding
```

---

## Topic 4: ETL

### Level 3: Advanced

**4.1** Build ETL pipeline:
```python
def etl_pipeline(source_file, target_table):
    """
    Complete ETL pipeline:
    1. Extract from CSV
    2. Clean data (handle missing, remove duplicates)
    3. Transform (create features)
    4. Load to SQLite
    """
    # Your code here
    pass
```

---

## Topic 5: Data Pipelines

### Level 3: Advanced

**5.1** Airflow DAG:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator

# Create DAG with:
# 1. Extract task
# 2. Transform task
# 3. Load task
# 4. Proper dependencies
```

---

## Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
```sql
-- 1
SELECT * FROM employees WHERE salary > 50000;

-- 2
SELECT department, COUNT(*) as emp_count 
FROM employees 
GROUP BY department;

-- 3
SELECT department, AVG(salary) as avg_salary 
FROM employees 
GROUP BY department;

-- 4
SELECT * FROM employees 
WHERE EXTRACT(YEAR FROM hire_date) = 2024;
```

### 2.1
```sql
-- 1. Rank by salary
SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;

-- 2. Running total
SELECT date, sales,
       SUM(sales) OVER (ORDER BY date) as running_total
FROM daily_sales;

-- 3. Top 3 per department
SELECT * FROM (
    SELECT name, department, salary,
           DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
    FROM employees
) ranked
WHERE rank <= 3;
```

### 3.1
```python
# 1. Count missing
df.isnull().sum()

# 2. Fill numeric with mean
df['A'] = df['A'].fillna(df['A'].mean())
df['B'] = df['B'].fillna(df['B'].mean())

# 3. Drop rows with any missing
df_clean = df.dropna()

# 4. Forward fill
df_ffill = df.fillna(method='ffill')
```

### 3.2
```python
# 1. One-hot encode
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')

# 2. Label encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])

# 3. Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

pipeline = Pipeline([
    ('encoder', OneHotEncoder(columns=['color']))
])
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23
