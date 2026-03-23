# 4.1 SQL Fundamentals

## 🎯 Quick Overview
- **SQL**: Language for relational databases
- **Queries**: Retrieve and manipulate data
- **Joins**: Combine data from multiple tables
- **Foundation for**: Data extraction, analysis, backend systems

---

## 1. Database Basics

### What is a Database?

```
Database = Organized collection of data

Relational Database:
- Data stored in tables
- Tables have rows (records) and columns (fields)
- Relationships between tables via keys
```

### Key Concepts

```
Table: Collection of related data
Row/Record: Single entry in a table
Column/Field: Attribute of a table
Primary Key: Unique identifier for a row
Foreign Key: Reference to primary key in another table
Index: Improves query performance
```

### Data Types

```sql
-- Numeric
INT, INTEGER, BIGINT, SMALLINT
DECIMAL, NUMERIC, FLOAT, REAL

-- String
CHAR(n), VARCHAR(n), TEXT

-- Date/Time
DATE, TIME, DATETIME, TIMESTAMP

-- Boolean
BOOLEAN, BOOL

-- Other
BLOB, JSON, XML
```

---

## 2. Basic SQL Queries

### SELECT Statement

```sql
-- Select all columns
SELECT * FROM employees;

-- Select specific columns
SELECT first_name, last_name, salary FROM employees;

-- Select with DISTINCT
SELECT DISTINCT department FROM employees;

-- Select with LIMIT
SELECT * FROM employees LIMIT 10;
```

### WHERE Clause

```sql
-- Basic conditions
SELECT * FROM employees WHERE salary > 50000;

-- Multiple conditions
SELECT * FROM employees 
WHERE salary > 50000 AND department = 'Engineering';

-- IN operator
SELECT * FROM employees WHERE department IN ('Engineering', 'Sales');

-- BETWEEN operator
SELECT * FROM employees WHERE salary BETWEEN 50000 AND 100000;

-- LIKE operator (pattern matching)
SELECT * FROM employees WHERE first_name LIKE 'J%';  -- Starts with J
SELECT * FROM employees WHERE first_name LIKE '%n';  -- Ends with n
SELECT * FROM employees WHERE first_name LIKE '%oh%';  -- Contains 'oh'

-- IS NULL
SELECT * FROM employees WHERE manager_id IS NULL;

-- IS NOT NULL
SELECT * FROM employees WHERE manager_id IS NOT NULL;
```

### ORDER BY

```sql
-- Single column
SELECT * FROM employees ORDER BY salary;

-- Descending
SELECT * FROM employees ORDER BY salary DESC;

-- Multiple columns
SELECT * FROM employees ORDER BY department ASC, salary DESC;
```

---

## 3. Aggregation and Grouping

### Aggregate Functions

```sql
-- COUNT
SELECT COUNT(*) FROM employees;
SELECT COUNT(DISTINCT department) FROM employees;

-- SUM
SELECT SUM(salary) FROM employees;

-- AVG
SELECT AVG(salary) FROM employees;

-- MIN/MAX
SELECT MIN(salary), MAX(salary) FROM employees;
```

### GROUP BY

```sql
-- Group by single column
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department;

-- Group by multiple columns
SELECT department, location, COUNT(*) as emp_count
FROM employees
GROUP BY department, location;
```

### HAVING Clause

```sql
-- Filter groups
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 60000;

-- HAVING vs WHERE
-- WHERE filters rows before grouping
-- HAVING filters groups after grouping
```

### ROLLUP, CUBE, GROUPING SETS

```sql
-- ROLLUP (subtotals)
SELECT department, location, SUM(salary)
FROM employees
GROUP BY ROLLUP(department, location);

-- CUBE (all combinations)
SELECT department, location, SUM(salary)
FROM employees
GROUP BY CUBE(department, location);
```

---

## 4. Joins

### INNER JOIN

```sql
-- Basic INNER JOIN
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.id;

-- Multiple JOINs
SELECT e.first_name, d.department_name, l.city
FROM employees e
INNER JOIN departments d ON e.department_id = d.id
INNER JOIN locations l ON d.location_id = l.id;
```

### OUTER JOINs

```sql
-- LEFT JOIN (all from left, matching from right)
SELECT e.first_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id;

-- RIGHT JOIN (all from right, matching from left)
SELECT e.first_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.id;

-- FULL OUTER JOIN (all from both)
SELECT e.first_name, d.department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.id;
```

### CROSS JOIN

```sql
-- Cartesian product
SELECT * FROM colors CROSS JOIN sizes;
```

### SELF JOIN

```sql
-- Join table to itself
SELECT e.first_name as employee, m.first_name as manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

---

## 5. Subqueries

### Subquery in WHERE

```sql
-- Find employees with above average salary
SELECT * FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- IN with subquery
SELECT * FROM employees
WHERE department_id IN (SELECT id FROM departments WHERE location = 'NYC');
```

### Subquery in SELECT

```sql
-- Add aggregate as column
SELECT first_name, salary,
       (SELECT AVG(salary) FROM employees) as avg_salary
FROM employees;
```

### Subquery in FROM

```sql
-- Derived table
SELECT dept, avg_salary
FROM (
    SELECT department as dept, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
) as dept_stats
WHERE avg_salary > 60000;
```

### Correlated Subquery

```sql
-- Find employees earning more than their department average
SELECT e.first_name, e.salary, e.department_id
FROM employees e
WHERE e.salary > (
    SELECT AVG(salary) 
    FROM employees 
    WHERE department_id = e.department_id
);

-- EXISTS with correlated subquery
SELECT * FROM employees e
WHERE EXISTS (
    SELECT 1 FROM departments d 
    WHERE d.id = e.department_id AND d.location = 'NYC'
);
```

---

## 6. Set Operations

### UNION

```sql
-- Combine results (remove duplicates)
SELECT first_name FROM employees
UNION
SELECT first_name FROM contractors;

-- UNION ALL (keep duplicates)
SELECT first_name FROM employees
UNION ALL
SELECT first_name FROM contractors;
```

### INTERSECT

```sql
-- Common rows
SELECT product_id FROM orders_2023
INTERSECT
SELECT product_id FROM orders_2024;
```

### EXCEPT (MINUS in Oracle)

```sql
-- Rows in first but not second
SELECT product_id FROM orders_2023
EXCEPT
SELECT product_id FROM orders_2024;
```

---

## 7. Data Modification

### INSERT

```sql
-- Insert single row
INSERT INTO employees (first_name, last_name, salary)
VALUES ('John', 'Doe', 75000);

-- Insert multiple rows
INSERT INTO employees (first_name, last_name, salary)
VALUES ('John', 'Doe', 75000),
       ('Jane', 'Smith', 80000);

-- INSERT from SELECT
INSERT INTO high_earners (first_name, last_name, salary)
SELECT first_name, last_name, salary
FROM employees
WHERE salary > 100000;
```

### UPDATE

```sql
-- Update single row
UPDATE employees
SET salary = 80000
WHERE id = 1;

-- Update multiple columns
UPDATE employees
SET salary = salary * 1.1, title = 'Senior'
WHERE department = 'Engineering';

-- Update with JOIN
UPDATE employees e
SET salary = e.salary * 1.1
FROM departments d
WHERE e.department_id = d.id AND d.performance = 'Excellent';
```

### DELETE

```sql
-- Delete specific rows
DELETE FROM employees WHERE id = 1;

-- Delete with condition
DELETE FROM employees WHERE salary < 30000;

-- TRUNCATE (faster, removes all rows)
TRUNCATE TABLE employees;
```

### UPSERT

```sql
-- INSERT ... ON CONFLICT (PostgreSQL)
INSERT INTO employees (id, first_name, salary)
VALUES (1, 'John', 75000)
ON CONFLICT (id) 
DO UPDATE SET salary = EXCLUDED.salary;

-- INSERT ... ON DUPLICATE KEY (MySQL)
INSERT INTO employees (id, first_name, salary)
VALUES (1, 'John', 75000)
ON DUPLICATE KEY UPDATE salary = VALUES(salary);
```

---

## 8. Schema and Table Management

### CREATE TABLE

```sql
CREATE TABLE employees (
    id INT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    salary DECIMAL(10, 2) CHECK (salary > 0),
    department_id INT,
    hire_date DATE DEFAULT CURRENT_DATE,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);
```

### ALTER TABLE

```sql
-- Add column
ALTER TABLE employees ADD COLUMN phone VARCHAR(20);

-- Drop column
ALTER TABLE employees DROP COLUMN phone;

-- Modify column
ALTER TABLE employees MODIFY COLUMN salary DECIMAL(12, 2);

-- Add constraint
ALTER TABLE employees ADD CONSTRAINT chk_salary CHECK (salary > 0);

-- Rename table
ALTER TABLE employees RENAME TO staff;
```

### DROP TABLE

```sql
DROP TABLE employees;
DROP TABLE IF EXISTS employees;
```

### CREATE/DROP INDEX

```sql
-- Create index
CREATE INDEX idx_department ON employees(department_id);

-- Create unique index
CREATE UNIQUE INDEX idx_email ON employees(email);

-- Drop index
DROP INDEX idx_department;
```

### CREATE/DROP VIEW

```sql
-- Create view
CREATE VIEW high_earners AS
SELECT first_name, last_name, salary
FROM employees
WHERE salary > 100000;

-- Use view
SELECT * FROM high_earners;

-- Drop view
DROP VIEW high_earners;
```

### Common Table Expressions (CTEs)

```sql
-- Basic CTE
WITH dept_stats AS (
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
)
SELECT * FROM dept_stats WHERE avg_salary > 60000;

-- Recursive CTE
WITH RECURSIVE numbers AS (
    SELECT 1 as n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 10
)
SELECT * FROM numbers;
```

---

## 9. Advanced SQL

### Window Functions

```sql
-- ROW_NUMBER
SELECT first_name, salary, department,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;

-- RANK and DENSE_RANK
SELECT first_name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank,
       DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;

-- NTILE
SELECT first_name, salary,
       NTILE(4) OVER (ORDER BY salary) as quartile
FROM employees;

-- LEAD and LAG
SELECT first_name, salary,
       LAG(salary) OVER (ORDER BY salary) as prev_salary,
       LEAD(salary) OVER (ORDER BY salary) as next_salary
FROM employees;

-- FIRST_VALUE and LAST_VALUE
SELECT first_name, salary, department,
       FIRST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary DESC) as highest
FROM employees;
```

### OVER Clause

```sql
-- Window frame
SELECT first_name, salary,
       AVG(salary) OVER (
           PARTITION BY department 
           ORDER BY hire_date
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ) as moving_avg
FROM employees;
```

---

## 10. SQL for Data Science

### Date/Time Functions

```sql
-- Current date/time
SELECT CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP;

-- Date extraction
SELECT EXTRACT(YEAR FROM hire_date) as hire_year FROM employees;

-- Date arithmetic
SELECT hire_date, hire_date + INTERVAL '1 year' as anniversary FROM employees;

-- Date difference
SELECT first_name, DATEDIFF(CURRENT_DATE, hire_date) as days_employed FROM employees;
```

### String Functions

```sql
-- Concatenation
SELECT CONCAT(first_name, ' ', last_name) as full_name FROM employees;

-- Substring
SELECT SUBSTRING(first_name, 1, 3) as initials FROM employees;

-- Case conversion
SELECT UPPER(first_name), LOWER(last_name) FROM employees;

-- Trim
SELECT TRIM(first_name) as trimmed FROM employees;

-- Length
SELECT first_name, LENGTH(first_name) as name_length FROM employees;
```

### CASE Expressions

```sql
-- Simple CASE
SELECT first_name, salary,
       CASE 
           WHEN salary > 100000 THEN 'High'
           WHEN salary > 60000 THEN 'Medium'
           ELSE 'Low'
       END as salary_level
FROM employees;

-- CASE with aggregation
SELECT department,
       SUM(CASE WHEN salary > 100000 THEN 1 ELSE 0 END) as high_earners,
       SUM(CASE WHEN salary <= 100000 THEN 1 ELSE 0 END) as low_earners
FROM employees
GROUP BY department;
```

### Pivoting

```sql
-- Pivot rows to columns
SELECT department,
       SUM(CASE WHEN year = 2022 THEN salary ELSE 0 END) as y2022,
       SUM(CASE WHEN year = 2023 THEN salary ELSE 0 END) as y2023
FROM employees
GROUP BY department;
```

### Percentiles and Quantiles

```sql
-- Percentile
SELECT first_name, salary,
       PERCENT_RANK() OVER (ORDER BY salary) as percentile
FROM employees;

-- NTILE for quantiles
SELECT first_name, salary,
       NTILE(100) OVER (ORDER BY salary) as percentile
FROM employees;
```

---

## 💻 Python Code Examples

```python
import sqlite3
import pandas as pd

# === Example 1: Basic Database Operations ===

def basic_sql_example():
    """Demonstrate basic SQL operations"""
    
    # Connect to database (in-memory)
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            salary REAL,
            department TEXT
        )
    ''')
    
    # Insert data
    employees = [
        (1, 'John', 'Doe', 75000, 'Engineering'),
        (2, 'Jane', 'Smith', 80000, 'Engineering'),
        (3, 'Bob', 'Johnson', 65000, 'Sales'),
        (4, 'Alice', 'Williams', 90000, 'Engineering'),
        (5, 'Charlie', 'Brown', 70000, 'Sales')
    ]
    
    cursor.executemany(
        'INSERT INTO employees VALUES (?, ?, ?, ?, ?)',
        employees
    )
    conn.commit()
    
    # Query data
    cursor.execute('SELECT * FROM employees')
    print("All employees:")
    for row in cursor.fetchall():
        print(row)
    
    # Query with WHERE
    cursor.execute('''
        SELECT first_name, last_name, salary 
        FROM employees 
        WHERE salary > 70000
    ''')
    print("\nHigh earners:")
    for row in cursor.fetchall():
        print(row)
    
    # Aggregation
    cursor.execute('''
        SELECT department, AVG(salary) as avg_salary, COUNT(*) as emp_count
        FROM employees
        GROUP BY department
    ''')
    print("\nDepartment stats:")
    for row in cursor.fetchall():
        print(f"{row[0]}: Avg={row[1]:.2f}, Count={row[2]}")
    
    conn.close()

# === Example 2: Using Pandas with SQL ===

def pandas_sql_example():
    """Demonstrate pandas SQL integration"""
    
    # Create sample database
    conn = sqlite3.connect(':memory:')
    
    # Create DataFrames
    employees_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'department_id': [1, 1, 2, 1, 2]
    })
    
    departments_df = pd.DataFrame({
        'id': [1, 2],
        'name': ['Engineering', 'Sales'],
        'location': ['NYC', 'LA']
    })
    
    # Save to SQL
    employees_df.to_sql('employees', conn, index=False)
    departments_df.to_sql('departments', conn, index=False)
    
    # Read from SQL
    query = '''
        SELECT e.first_name, d.name as department, d.location
        FROM employees e
        JOIN departments d ON e.department_id = d.id
    '''
    
    result = pd.read_sql_query(query, conn)
    print(result)
    
    conn.close()

# === Example 3: Complex Query ===

def complex_query_example():
    """Demonstrate complex SQL query"""
    
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            amount REAL,
            order_date TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT
        )
    ''')
    
    # Insert sample data
    cursor.executemany('INSERT INTO customers VALUES (?, ?, ?)', [
        (1, 'Alice', 'NYC'),
        (2, 'Bob', 'LA'),
        (3, 'Charlie', 'NYC')
    ])
    
    cursor.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?)', [
        (1, 1, 100, 50.00, '2024-01-01'),
        (2, 1, 101, 75.00, '2024-01-15'),
        (3, 2, 100, 50.00, '2024-01-10'),
        (4, 3, 102, 100.00, '2024-01-20')
    ])
    
    conn.commit()
    
    # Complex query with window function
    query = '''
        SELECT 
            c.name,
            c.city,
            o.amount,
            SUM(o.amount) OVER (PARTITION BY c.id) as total_spent,
            ROW_NUMBER() OVER (PARTITION BY c.id ORDER BY o.amount DESC) as order_rank
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        ORDER BY c.name, o.order_date
    '''
    
    result = pd.read_sql_query(query, conn)
    print(result)
    
    conn.close()

# Run examples
basic_sql_example()
pandas_sql_example()
complex_query_example()
```

---

## 📊 Summary Tables

### SQL Commands

| Category | Commands |
|----------|----------|
| DDL | CREATE, ALTER, DROP, TRUNCATE |
| DML | INSERT, UPDATE, DELETE |
| DQL | SELECT |
| DCL | GRANT, REVOKE |
| TCL | COMMIT, ROLLBACK, SAVEPOINT |

### JOIN Types

| JOIN Type | Description |
|-----------|-------------|
| INNER JOIN | Matching rows only |
| LEFT JOIN | All from left, matching from right |
| RIGHT JOIN | All from right, matching from left |
| FULL OUTER | All from both tables |
| CROSS JOIN | Cartesian product |
| SELF JOIN | Join table to itself |

### Window Functions

| Function | Purpose |
|----------|---------|
| ROW_NUMBER | Unique row number |
| RANK | Rank with gaps |
| DENSE_RANK | Rank without gaps |
| NTILE | Divide into buckets |
| LEAD/LAG | Access next/previous row |
| FIRST_VALUE/LAST_VALUE | First/last value in window |

---

## 🎯 ML Applications

| SQL Feature | ML Application |
|-------------|----------------|
| SELECT/JOIN | Feature extraction |
| Aggregation | Feature engineering |
| Window functions | Time series features |
| Subqueries | Complex feature logic |
| CTEs | Query organization |

---

**Status:** ✅ Complete
**Next:** NoSQL Basics
