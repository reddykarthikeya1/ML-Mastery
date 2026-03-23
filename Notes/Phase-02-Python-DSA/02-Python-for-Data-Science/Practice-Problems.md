# Python for Data Science - Practice Problems

## 📊 Graded Practice Levels

### Level 1: Basic Tooling and NumPy
**1.1** In a Jupyter Notebook, what is the difference between a "Code" cell and a "Markdown" cell? Name one "Magic Command" and its purpose.
**1.2** Create a $5 \times 5$ NumPy array of all zeros, then change all the elements on the main diagonal to 1 (Identity matrix) without using `np.eye`.
**1.3** Given an array `arr = np.array([10, 20, 30, 40, 50])`, calculate the Z-score normalization for every element: $z = (x - \mu) / \sigma$.
**1.4** What is "Broadcasting" in NumPy? Give a simple example where a scalar is added to a matrix.

### Level 2: Intermediate Pandas Operations
**2.1** Load a CSV file into a DataFrame. Filter the rows where 'Age' is greater than 25 and 'City' is 'London'.
**2.2** Given a DataFrame with missing values in the 'Salary' column, fill the missing values with the median salary of the entire dataset.
**2.3** Use `groupby()` to find the average 'Sales' and total 'Quantity' for each 'Category' in a retail dataset.
**2.4** Explain the difference between `pd.merge()` and `pd.concat()`. When would you use one over the other?

### Level 3: Advanced Data Manipulation and Viz
**3.1** **Time Series:** Given a DataFrame with a 'Timestamp' column, convert it to datetime objects, set it as the index, and resample the data to find the weekly average of a 'Price' column.
**3.2** **Multi-Indexing:** Create a DataFrame with a hierarchical index (e.g., 'State' and 'City') and select all data for a specific 'State'.
**3.3** **Matplotlib:** Create a figure with two subplots side-by-side. The first should be a line plot of $\sin(x)$ and the second a scatter plot of random noise.
**3.4** **Seaborn:** Using the 'Titanic' dataset (or similar), create a box plot showing the distribution of 'Age' for each 'Class', colored by whether the passenger 'Survived'.

### Level 4: Python Implementation Practice
**4.1** Write a NumPy function that takes a matrix and returns a new matrix where every element is replaced by its rank within its row.
**4.2** Write a Pandas function to clean a "dirty" dataset:
1. Remove duplicates.
2. Convert 'Date' string column to datetime.
3. One-hot encode all categorical columns with fewer than 10 unique values.
4. Scale all numeric columns to the range [0, 1] (Min-Max scaling).

### Level 5: Real-world Data Analysis Scenario
**5.1** **Scenario:** You are analyzing website traffic data.
- Input: A CSV with `timestamp`, `user_id`, `page_visited`, and `time_spent`.
- Goal: Identify "Power Users" (top 5% by total time spent) and visualize their daily activity patterns compared to average users.
**Task:** Describe the steps (load, clean, aggregate, visualize) and the specific Python functions/plots you would use to deliver this insight.

---

## 📝 Solutions (Selected)

<details>
<summary>Click to reveal solutions</summary>

### 1.2
```python
arr = np.zeros((5, 5))
np.fill_diagonal(arr, 1)
```

### 2.3
```python
df.groupby('Category').agg({
    'Sales': 'mean',
    'Quantity': 'sum'
})
```

### 3.1
```python
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
weekly_avg = df['Price'].resample('W').mean()
```

### 4.2
```python
def clean_data(df):
    df = df.drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'])
    # One-hot
    cat_cols = [c for c in df.select_dtypes('object') if df[c].nunique() < 10]
    df = pd.get_dummies(df, columns=cat_cols)
    # Min-Max
    num_cols = df.select_dtypes('number').columns
    df[num_cols] = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())
    return df
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
