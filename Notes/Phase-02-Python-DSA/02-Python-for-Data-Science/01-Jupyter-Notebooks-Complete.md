# 3.1 Jupyter Notebooks

## 🎯 Quick Overview
- **Jupyter**: Interactive computing environment
- **Notebook cells**: Code, markdown, raw
- **Magic commands**: Special commands for productivity
- **Foundation for**: Data exploration, prototyping, documentation

---

## 1. Jupyter Basics

### What is Jupyter?

```
Jupyter = Interactive web-based notebook for code, visualizations, and text

Components:
- Jupyter Notebook: Original interface
- JupyterLab: Next-gen interface with tabs, terminals
- Jupyter Kernel: Executes code
- IPython: Interactive Python shell
```

### Installation and Setup

```bash
# Install Jupyter
pip install jupyter

# Install JupyterLab
pip install jupyterlab

# Start notebook
jupyter notebook

# Start JupyterLab
jupyter lab

# Install kernel for specific Python version
python -m ipykernel install --user --name=myenv
```

### Interface Overview

```
JupyterLab Interface:
├── File Browser
├── Running Terminals/Notebooks
├── Main Work Area (tabs)
└── Menu Bar

Notebook Interface:
├── Menu Bar (File, Edit, Cell, Kernel, etc.)
├── Toolbar (Save, Add, Cut, Copy, Run, etc.)
└── Cells (Code, Markdown, Raw)
```

---

## 2. Cell Types

### Code Cells

```python
# Execute Python code
print("Hello, Jupyter!")

# Shift+Enter to run
# Ctrl+Enter to run and stay in cell
# Alt+Enter to run and insert new cell below
```

### Markdown Cells

```markdown
# Heading 1
## Heading 2
### Heading 3

**Bold text**
*Italic text*
~~Strikethrough~~

- Bullet point
- Another point

1. Numbered list
2. Second item

[Link](https://jupyter.org)

![Image](image.png)

> Blockquote

`Inline code`

```python
# Code block
def hello():
    print("World")
```

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2   |
```

### Raw Cells

```
Raw text without formatting
Useful for:
- LaTeX equations
- Custom formatting
- Export control
```

---

## 3. Magic Commands

### Line Magics (single line, %prefix)

```python
# Timing
%timeit sum(range(1000))      # Time execution (multiple runs)
%time sum(range(1000))        # Time execution (single run)

# Display
%matplotlib inline            # Display plots inline
%matplotlib notebook          # Interactive plots

# File operations
%load script.py               # Load code from file
%run script.py                # Run script
%save mycode.py 1-10          # Save cells to file

# Information
%who                          # List variables
%whos                         # Detailed variable info
%env                          # Environment variables
%pwd                          # Print working directory
%ls                           # List directory

# History
%history                      # Command history
%paste                        # Paste from clipboard
```

### Cell Magics (entire cell, %%prefix)

```python
%%time
for i in range(1000):
    pass

%%writefile output.txt
Content to write to file
Multiple lines

%%html
<div style="color: red">HTML content</div>

%%bash
echo "Bash command"
ls -la

%%python3
# Explicit Python 3

%%latex
\sum_{i=1}^n i = \frac{n(n+1)}{2}
```

---

## 4. Notebook Features

### Keyboard Shortcuts

```
Command Mode (Esc):
- A: Insert cell above
- B: Insert cell below
- D,D: Delete cell
- Z: Undo delete
- C: Copy cell
- V: Paste cell
- X: Cut cell
- Y: Change to code cell
- M: Change to markdown cell
- Shift+M: Toggle markdown
- O: Toggle output
- H: Show shortcuts

Edit Mode (Enter):
- Tab: Code completion
- Shift+Tab: Show docstring
- Ctrl+]: Indent
- Ctrl+[: Dedent
- Ctrl+/: Toggle comment
- Ctrl+Z: Undo
- Ctrl+Y: Redo
```

### Kernel Management

```python
# Kernel menu options:
# - Interrupt: Stop current execution
# - Restart: Restart kernel (clears variables)
# - Restart & Run All: Restart and execute all cells
# - Change kernel: Switch to different kernel

# Check kernel
import sys
print(sys.executable)
print(sys.version)

# List installed kernels
!jupyter kernelspec list
```

### Output Management

```python
# Suppress output
;  # Add semicolon to suppress output

# Multiple outputs
print("First")
print("Second")  # Both show

# Last expression only (no print)
"First"
"Second"  # Only "Second" shows

# Clear output
from IPython.display import clear_output
clear_output()

# Display multiple items
from IPython.display import display
display(obj1, obj2, obj3)
```

---

## 5. Rich Output

### HTML Display

```python
from IPython.display import HTML, display

display(HTML('<h1 style="color: blue">Hello</h1>'))
```

### Images

```python
from IPython.display import Image, display

# From file
display(Image(filename='image.png'))

# From URL
display(Image(url='https://example.com/image.png'))

# Resize
display(Image(filename='image.png', width=200, height=200))
```

### Videos

```python
from IPython.display import YouTubeVideo

YouTubeVideo('video_id', width=640, height=480)
```

### LaTeX

```python
from IPython.display import Math, Latex

display(Math(r'\sum_{i=1}^n i = \frac{n(n+1)}{2}'))
display(Latex(r'\begin{equation} E = mc^2 \end{equation}'))
```

---

## 6. Widgets (ipywidgets)

### Basic Widgets

```python
import ipywidgets as widgets
from IPython.display import display

# Slider
slider = widgets.IntSlider(min=0, max=100, value=50)
display(slider)

# Text input
text = widgets.Text(value='Hello', description='Name:')
display(text)

# Dropdown
dropdown = widgets.Dropdown(options=['A', 'B', 'C'], value='A')
display(dropdown)

# Checkbox
checkbox = widgets.Checkbox(value=True, description='Check me')
display(checkbox)

# Button
button = widgets.Button(description='Click me')
display(button)

def on_click(b):
    print('Button clicked!')

button.on_click(on_click)
```

### Interactive Functions

```python
from ipywidgets import interact, interactive

@interact(x=(0, 10, 1), y=(0, 10, 1))
def plot_point(x=5, y=5):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.plot(x, y, 'ro', markersize=15)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.show()
```

---

## 7. Best Practices

### Organizing Notebooks

```markdown
1. Use markdown cells for documentation
2. Start with overview/explanation
3. Group related code in sections
4. Use clear cell ordering
5. Add headings for navigation
6. Include example outputs
7. Document assumptions and limitations
```

### Version Control

```bash
# Strip outputs before committing
jupyter nbconvert --clear-output notebook.ipynb

# Convert to script
jupyter nbconvert --to script notebook.ipynb

# Use nbdime for diffing
pip install nbdime
nbdime diff notebook1.ipynb notebook2.ipynb

# Jupyter Git extension
pip install jupyterlab-git
```

### Performance Tips

```python
# Use %load_ext for extensions
%load_ext autoreload
%autoreload 2  # Auto-reload modules

# Profile code
%prun function_call()

# Memory usage
%memit function_call()  # Requires memory_profiler

# Avoid large outputs
# Use logging instead of print for debugging
# Clear intermediate outputs
```

---

## 💻 Python Code Examples

```python
# === Example 1: Complete Data Analysis Notebook Structure ===

"""
# Data Analysis Template

## Overview
This notebook analyzes [dataset] to answer [questions].

## Setup
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure settings
%matplotlib inline
plt.style.use('seaborn')
pd.set_option('display.max_columns', 50)

"""
## Data Loading
"""

# Load data
df = pd.read_csv('data.csv')
print(f"Shape: {df.shape}")
df.head()

"""
## Exploratory Data Analysis
"""

# Summary statistics
df.describe()

# Missing values
df.isnull().sum()

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... plotting code

"""
## Conclusions
1. Key finding 1
2. Key finding 2
3. Recommendations
"""

# === Example 2: Interactive Dashboard ===

import ipywidgets as widgets
from ipywidgets import interact

@interact(
    n=widgets.IntSlider(min=10, max=1000, value=100, description='Samples'),
    dist=widgets.Dropdown(options=['normal', 'uniform', 'exponential'], value='normal')
)
def plot_distribution(n=100, dist='normal'):
    """Interactive distribution plot"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    if dist == 'normal':
        data = np.random.randn(n)
    elif dist == 'uniform':
        data = np.random.rand(n)
    else:
        data = np.random.exponential(size=n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(data, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_title(f'{dist.title()} Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    ax2.boxplot(data, vert=False)
    ax2.set_title('Box Plot')
    ax2.set_xlabel('Value')
    
    plt.tight_layout()
    plt.show()

# === Example 3: Timing Comparison ===

# Compare different approaches
import numpy as np

arr = np.random.rand(10000)

# Method 1: Loop
%timeit
result1 = []
for x in arr:
    result1.append(x * 2)

# Method 2: List comprehension
%timeit
result2 = [x * 2 for x in arr]

# Method 3: NumPy vectorization
%timeit
result3 = arr * 2
```

---

## 📊 Summary Tables

### Magic Commands

| Command | Type | Purpose |
|---------|------|---------|
| %timeit | Line | Time execution |
| %run | Line | Run script |
| %load | Line | Load code |
| %%time | Cell | Time cell |
| %%writefile | Cell | Write to file |
| %%html | Cell | Render HTML |

### Keyboard Shortcuts

| Shortcut | Mode | Action |
|----------|------|--------|
| Esc | Command | Exit edit mode |
| Enter | Edit | Enter edit mode |
| A | Command | Cell above |
| B | Command | Cell below |
| D,D | Command | Delete cell |
| M | Command | Markdown cell |
| Y | Command | Code cell |
| Shift+Enter | Both | Run cell |

---

## 🎯 ML Applications

| Jupyter Feature | ML Application |
|-----------------|----------------|
| Interactive cells | Model experimentation |
| Markdown | Documentation |
| Widgets | Hyperparameter tuning UI |
| Magic commands | Performance profiling |
| Rich output | Model visualization |

---

## ❓ Quick Check Questions

1. What's the difference between Jupyter Notebook and JupyterLab?
2. How do you display plots inline?
3. What does %timeit do?
4. How do you convert a cell to markdown?
5. What's the shortcut to delete a cell?
6. How do you suppress cell output?

---

## 📝 Answers to Quick Check

1. **Notebook vs Lab:**
   - Notebook: Classic interface
   - Lab: Modern, tabbed interface

2. **Display plots inline:**
   - %matplotlib inline

3. **%timeit:**
   - Times execution over multiple runs
   - Reports average time

4. **Convert to markdown:**
   - Command mode + M
   - Or Cell menu → Cell Type → Markdown

5. **Delete cell:**
   - D,D (press D twice)

6. **Suppress output:**
   - Add semicolon at end of line

---

**Status:** ✅ Complete
**Next:** NumPy
