# Data Engineering - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Concept Recall
**1.1** What is the difference between a `WHERE` clause and a `HAVING` clause in SQL?
**1.2** List the four main types of NoSQL databases and provide one common use case for each.
**1.3** Define the three main types of missing data mechanisms (MCAR, MAR, MNAR).
**1.4** What are the three phases of an ETL pipeline?

### Level 2: Intermediate Operations
**2.1** Write a SQL query to find the second highest salary from an `employees` table without using Window Functions (e.g., using `MAX` or `LIMIT`/`OFFSET`).
**2.2** In MongoDB, write an aggregation pipeline to group users by `city`, calculate the average age per city, and sort the results by the average age in descending order.
**2.3** Given a dataset with severe outliers in a `salary` column, why might `RobustScaler` be a better choice than `StandardScaler`?
**2.4** Explain the difference between Full Extraction and Incremental Extraction. Why is CDC (Change Data Capture) important?

### Level 3: Advanced Queries and Design
**3.1** Write a SQL query using Window Functions to assign a dense rank to employees based on their sales performance, partitioned by their department.
**3.2** Compare the Lambda and Kappa data pipeline architectures. In what scenario would you explicitly choose Kappa over Lambda?
**3.3** You have a categorical feature with 1,000 unique string values (high cardinality). Why is One-Hot Encoding a poor choice here, and what encoding technique would you use instead?

### Level 4: Python Implementation Practice
**4.1** Using `pandas`, write a concise script to:
1. Load a CSV file named `data.csv`.
2. Fill missing numeric values with the column median.
3. Drop any rows where the `email` column is missing.
4. Apply a One-Hot Encoding to the `department` column.

**4.2** Write a simple Python function to simulate a Redis cache check: It should take a `user_id`, check if `user:{user_id}` exists in a dictionary (acting as Redis). If it does, return the data. If not, simulate a DB call, store the result in the dictionary, and return it.

### Level 5: Real-world System Design
**5.1** **Scenario:** You are building a real-time recommendation engine for an e-commerce platform. 
- User clicks stream in at 10,000 events per second.
- You need to update user profiles in real-time.
- You also need to run a nightly batch job to retrain your Machine Learning models using all historical data.
**Task:** Outline the data architecture for this system. Specify which databases (SQL vs NoSQL types) you would use for the real-time cache vs. the historical data, and whether you would use an ETL, ELT, Lambda, or Kappa pipeline approach.

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.1
`WHERE` filters individual rows before any grouping or aggregations are applied. `HAVING` filters the resulting groups after the `GROUP BY` clause and aggregations have been processed.

### 2.1
```sql
SELECT MAX(salary) 
FROM employees 
WHERE salary < (SELECT MAX(salary) FROM employees);
```
*(Alternatively, using ORDER BY and LIMIT/OFFSET depending on the SQL dialect).*

### 2.2
```javascript
db.users.aggregate([
    { $group: { _id: "$city", avg_age: { $avg: "$age" } } },
    { $sort: { avg_age: -1 } }
])
```

### 3.1
```sql
SELECT 
    employee_id, 
    department_id, 
    sales,
    DENSE_RANK() OVER (PARTITION BY department_id ORDER BY sales DESC) as sales_rank
FROM employees;
```

### 3.3
One-Hot Encoding 1,000 unique values would create 1,000 new sparse columns, leading to the Curse of Dimensionality (massive memory footprint, overfitting risk). Instead, you should use **Target Encoding** (Mean Encoding), **Frequency Encoding**, or the **Hashing Trick** to keep the dimensional footprint small.

### 4.1
```python
import pandas as pd

df = pd.read_csv('data.csv')
# Fill numeric missing with median
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Drop missing emails
df = df.dropna(subset=['email'])

# One-hot encode department
df = pd.get_dummies(df, columns=['department'], drop_first=True)
```

</details>

---

## 📝 Notes Section

### My Practice Problems:


### Mistakes to Review:


### Key Insights:


---
**Last Updated:** 2026-03-23