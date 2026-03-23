# 3.4 Data Visualization

## 🎯 Quick Overview
- **Matplotlib**: Foundation plotting library
- **Seaborn**: Statistical visualization
- **Best practices**: Clear, effective visualizations
- **Foundation for**: EDA, model evaluation, presentations

---

## 1. Matplotlib Fundamentals

### Basic Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Sine Wave')
plt.show()

# Scatter plot
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()

# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [10, 24, 36, 18]

plt.bar(categories, values)
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart')
plt.show()

# Histogram
data = np.random.randn(1000)

plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Box plot
data = [np.random.randn(100) for _ in range(4)]

plt.boxplot(data, labels=['A', 'B', 'C', 'D'])
plt.ylabel('Value')
plt.title('Box Plot')
plt.show()
```

### Figure and Axes

```python
# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot on axes
ax.plot(x, y, 'b-', linewidth=2, label='Sine')

# Customize
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Title', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Save figure
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Multiple Subplots

```python
# 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine')

axes[0, 1].scatter(x, np.cos(x))
axes[0, 1].set_title('Cosine')

axes[1, 0].bar(categories, values)
axes[1, 0].set_title('Bar')

axes[1, 1].hist(data, bins=20)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

---

## 2. Matplotlib Advanced

### Customization

```python
# Line styles
plt.plot(x, y, linestyle='-', linewidth=2)   # Solid
plt.plot(x, y, linestyle='--', linewidth=2)  # Dashed
plt.plot(x, y, linestyle='-.', linewidth=2)  # Dash-dot
plt.plot(x, y, linestyle=':', linewidth=2)   # Dotted

# Markers
plt.plot(x, y, marker='o', markersize=8, markerfacecolor='red')

# Colors
plt.plot(x, y, color='red')
plt.plot(x, y, color='#FF5733')
plt.plot(x, y, color=(0.1, 0.2, 0.5))

# Fill between
plt.fill_between(x, np.sin(x), alpha=0.3)

# Annotations
plt.annotate('Peak', xy=(np.pi/2, 1), xytext=(2, 1.2),
             arrowprops=dict(arrowstyle='->'))
plt.text(5, 0.5, 'Some text', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
```

### Color Maps

```python
# Sequential
plt.scatter(x, y, c=y, cmap='viridis')
plt.colorbar(label='Value')

# Diverging
plt.scatter(x, y, c=y, cmap='coolwarm')

# Categorical
plt.scatter(x, y, c=categories, cmap='tab10')

# Available colormaps
# Sequential: viridis, plasma, inferno, magma, blues, greens
# Diverging: coolwarm, seismic, RdBu
# Categorical: tab10, tab20, Set1, Set2
```

### 3D Plotting

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Scatter 3D
ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten(), cmap='viridis')
plt.show()
```

---

## 3. Seaborn

### Setup

```python
import seaborn as sns

# Set theme
sns.set_theme(style='whitegrid')

# Set palette
sns.set_palette('husl')

# Load example dataset
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
```

### Distribution Plots

```python
# Histogram with KDE
sns.histplot(data=tips, x='total_bill', kde=True)

# KDE plot
sns.kdeplot(data=tips, x='total_bill', fill=True)

# Box plot
sns.boxplot(data=tips, x='day', y='total_bill')

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill')

# ECDF plot
sns.ecdfplot(data=tips, x='total_bill')

# Joint plot
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')

# Pair plot
sns.pairplot(iris, hue='species')
```

### Categorical Plots

```python
# Bar plot
sns.barplot(data=tips, x='day', y='total_bill')

# Count plot
sns.countplot(data=tips, x='day')

# Strip plot
sns.stripplot(data=tips, x='day', y='total_bill', jitter=True)

# Swarm plot
sns.swarmplot(data=tips, x='day', y='total_bill')

# Point plot
sns.pointplot(data=tips, x='day', y='total_bill')

# Boxen plot
sns.boxenplot(data=tips, x='day', y='total_bill')
```

### Relational Plots

```python
# Scatter plot
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species')

# Line plot
sns.lineplot(data=tips, x='total_bill', y='tip', hue='time')

# Relplot (figure-level)
sns.relplot(data=iris, x='sepal_length', y='sepal_width', 
            hue='species', col='species')
```

### Matrix Plots

```python
# Heatmap
corr = iris.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)

# Clustermap
sns.clustermap(corr, annot=True, cmap='coolwarm')
```

### Regression Plots

```python
# Regression plot
sns.regplot(data=tips, x='total_bill', y='tip')

# LM plot (with hue)
sns.lmplot(data=tips, x='total_bill', y='tip', hue='time')

# Residual plot
sns.residplot(data=tips, x='total_bill', y='tip')
```

### Facet Grids

```python
# FacetGrid
g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(plt.scatter, 'total_bill', 'tip')
g.add_legend()

# Catplot
sns.catplot(data=tips, x='day', y='total_bill', 
            kind='box', col='time', row='smoker')

# Displot
sns.displot(data=tips, x='total_bill', col='time', kde=True)
```

---

## 4. Visualization Best Practices

### Choosing the Right Plot

| Goal | Plot Type |
|------|-----------|
| Show distribution | Histogram, KDE, Box plot |
| Compare categories | Bar chart |
| Show relationship | Scatter plot, Line plot |
| Show correlation | Heatmap |
| Show composition | Pie chart (use sparingly), Stacked bar |
| Show change over time | Line plot |

### Design Principles

```
1. Maximize data-ink ratio
2. Avoid chart junk
3. Use appropriate scales
4. Label clearly
5. Use color purposefully
6. Consider colorblind accessibility
7. Provide context
```

### Color Accessibility

```python
# Colorblind-friendly palettes
sns.set_palette('colorblind')

# Or use specific palettes
sns.color_palette('tab10')  # Good for categories
sns.color_palette('viridis')  # Good for sequential
```

---

## 💻 Python Code Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as np

# === Example 1: Complete EDA Visualization ===

def create_eda_report(df, target_column=None):
    """Create comprehensive EDA visualizations"""
    
    # Setup
    fig = plt.figure(figsize=(20, 15))
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for i, col in enumerate(numeric_cols[:6], 1):
        plt.subplot(2, 3, i)
        
        # Histogram with KDE
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel('')
    
    plt.tight_layout()
    plt.savefig('eda_distributions.png', dpi=300)
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.show()

# === Example 2: Model Evaluation Plots ===

def plot_model_evaluation(y_true, y_pred, y_pred_proba=None):
    """Create model evaluation visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist='norm', plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300)
    plt.show()

# === Example 3: Time Series Visualization ===

def plot_time_series(df, date_column, value_column):
    """Create time series visualizations"""
    
    # Convert date
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Main time series
    axes[0].plot(df[date_column], df[value_column], linewidth=2)
    axes[0].set_title(f'{value_column} Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel(value_column)
    axes[0].grid(True, alpha=0.3)
    
    # Rolling average
    df[f'{value_column}_7d'] = df[value_column].rolling(window=7).mean()
    df[f'{value_column}_30d'] = df[value_column].rolling(window=30).mean()
    
    axes[1].plot(df[date_column], df[value_column], alpha=0.3, label='Daily')
    axes[1].plot(df[date_column], df[f'{value_column}_7d'], 
                 label='7-day MA', linewidth=2)
    axes[1].plot(df[date_column], df[f'{value_column}_30d'], 
                 label='30-day MA', linewidth=2)
    axes[1].set_title('With Moving Averages')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(value_column)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Seasonality (by month)
    df['month'] = df[date_column].dt.month
    monthly_avg = df.groupby('month')[value_column].mean()
    
    axes[2].bar(monthly_avg.index, monthly_avg.values, edgecolor='black')
    axes[2].set_title('Monthly Average')
    axes[2].set_xlabel('Month')
    axes[2].set_ylabel(f'Average {value_column}')
    axes[2].set_xticks(range(1, 13))
    axes[2].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300)
    plt.show()

# === Example 4: Comparison Plot ===

def create_comparison_plot():
    """Create multi-style comparison plot"""
    
    # Generate data
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    values1 = [23, 45, 56, 78]
    values2 = [12, 34, 45, 67]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Grouped bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, values1, width, label='Series 1')
    axes[0, 0].bar(x + width/2, values2, width, label='Series 2')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories, rotation=45)
    axes[0, 0].set_title('Grouped Bar Chart')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Stacked bar chart
    axes[0, 1].bar(categories, values1, label='Series 1')
    axes[0, 1].bar(categories, values2, bottom=values1, label='Series 2')
    axes[0, 1].set_title('Stacked Bar Chart')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Horizontal bar chart
    axes[1, 0].barh(categories, values1)
    axes[1, 0].set_title('Horizontal Bar Chart')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Pie chart
    axes[1, 1].pie(values1, labels=categories, autopct='%1.1f%%', 
                   startangle=90, explode=(0.1, 0, 0, 0))
    axes[1, 1].set_title('Pie Chart')
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png', dpi=300)
    plt.show()
```

---

## 📊 Summary Tables

### Plot Types

| Plot | Function | Use Case |
|------|----------|----------|
| Line | plt.plot() | Trends over time |
| Scatter | plt.scatter() | Relationship |
| Bar | plt.bar() | Compare categories |
| Histogram | plt.hist() | Distribution |
| Box | plt.boxplot() | Distribution + outliers |
| Heatmap | sns.heatmap() | Correlation matrix |

### Seaborn Plot Categories

| Category | Functions |
|----------|-----------|
| Distribution | histplot, kdeplot, boxplot, violinplot |
| Categorical | barplot, countplot, stripplot, swarmplot |
| Relational | scatterplot, lineplot, relplot |
| Matrix | heatmap, clustermap |
| Regression | regplot, lmplot, residplot |

---

## 🎯 ML Applications

| Visualization | ML Application |
|---------------|----------------|
| Scatter plots | Feature relationships |
| Heatmaps | Correlation analysis |
| Box plots | Outlier detection |
| Learning curves | Model diagnostics |
| Confusion matrix | Classification evaluation |
| ROC curves | Model comparison |

---

---

## ❓ Quick Check Questions

1. What is the hierarchy of objects in Matplotlib (from the largest container down)?
2. What is the difference between `plt.plot()` and `plt.scatter()`?
3. Which Seaborn plot is best for visualizing the distribution of a single numeric variable along with its density?
4. What information does a Box Plot provide about a dataset?
5. When would you use a `FacetGrid` or `pairplot` in Seaborn?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. The Matplotlib hierarchy is: **Figure** (the entire window/page) → **Axes** (an individual plot/subplot) → **Axis** (the x and y lines) → **Tick/Label**.
2. **`plt.plot()`** is primarily for line plots (connecting points), while **`plt.scatter()`** is for individual data points (useful for showing correlations or clusters).
3. **`sns.histplot(kde=True)`** or **`sns.displot(kde=True)`** is best for showing both the frequency (histogram) and the estimated probability density (KDE).
4. A **Box Plot** shows the five-number summary: the **minimum**, **first quartile (Q1)**, **median (Q2)**, **third quartile (Q3)**, and **maximum**, while also explicitly identifying **outliers**.
5. Use a **`FacetGrid`** to create a grid of plots based on the values of categorical variables (e.g., separate plots for "Smoker" vs. "Non-smoker"). Use a **`pairplot`** to automatically plot pairwise relationships across an entire dataset (scatter plots for every combination of numeric features).

</details>
---

**Status:** ✅ Complete
**Next:** Practice Problems
