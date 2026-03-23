# 4.2-4.5 Data Engineering - Complete Reference

## 🎯 Quick Overview
- **NoSQL**: Non-relational databases for specific use cases
- **Data Preprocessing**: Clean and transform data for ML
- **ETL**: Extract, Transform, Load pipelines
- **Data Pipelines**: Automated data workflows
- **Foundation for**: Data infrastructure, ML pipelines

---

## Part 1: NoSQL Basics (4.2)

### SQL vs NoSQL

```
SQL Databases:
- Table-based
- Schema-based
- ACID compliant
- Vertical scaling
- Examples: MySQL, PostgreSQL, Oracle

NoSQL Databases:
- Document/Key-Value/Graph/Column-based
- Schema-less
- BASE model (Basically Available, Soft state, Eventual consistency)
- Horizontal scaling
- Examples: MongoDB, Redis, Cassandra, Neo4j
```

### CAP Theorem

```
Choose 2 of 3:
- Consistency: All nodes see same data
- Availability: Every request gets response
- Partition Tolerance: System works despite network failures

SQL: Typically CP
NoSQL: Typically AP
```

### Document Databases (MongoDB)

```javascript
// Document structure
{
    _id: ObjectId("..."),
    name: "Alice",
    age: 25,
    skills: ["Python", "ML"],
    address: {
        city: "NYC",
        zip: "10001"
    }
}

// CRUD Operations
// Create
db.users.insertOne({name: "Bob", age: 30})

// Read
db.users.find({age: {$gt: 25}})
db.users.findOne({name: "Alice"})

// Update
db.users.updateOne(
    {name: "Bob"},
    {$set: {age: 31}}
)

// Delete
db.users.deleteOne({name: "Bob"})

// Aggregation Pipeline
db.orders.aggregate([
    {$match: {status: "completed"}},
    {$group: {_id: "$customer", total: {$sum: "$amount"}}},
    {$sort: {total: -1}},
    {$limit: 10}
])
```

### Key-Value Stores (Redis)

```python
import redis

r = redis.Redis(host='localhost', port=6379)

# String operations
r.set('name', 'Alice')
r.get('name')  # b'Alice'

# List operations
r.lpush('mylist', 'a', 'b', 'c')
r.lrange('mylist', 0, -1)

# Hash operations
r.hset('user:1', mapping={'name': 'Alice', 'age': 25})
r.hgetall('user:1')

# Set operations
r.sadd('tags', 'python', 'ml', 'ai')
r.smembers('tags')

# Sorted Set
r.zadd('leaderboard', {'Alice': 100, 'Bob': 85})
r.zrevrange('leaderboard', 0, 9, withscores=True)

# Expiration
r.setex('cache:key', 3600, 'value')  # Expires in 1 hour
```

### Column-Family Stores (Cassandra)

```sql
-- CQL (Cassandra Query Language)
CREATE KEYSPACE mydb 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};

USE mydb;

CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    email TEXT,
    created_at TIMESTAMP
);

INSERT INTO users (user_id, name, email) 
VALUES (uuid(), 'Alice', 'alice@example.com');

SELECT * FROM users WHERE user_id = ?;
```

### Graph Databases (Neo4j)

```cypher
// Create nodes
CREATE (p:Person {name: 'Alice', age: 25})
CREATE (f:Person {name: 'Bob', age: 30})

// Create relationship
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
CREATE (a)-[:FRIENDS_WITH {since: 2020}]->(b)

// Query
MATCH (p:Person)-[:FRIENDS_WITH]->(friend)
WHERE p.name = 'Alice'
RETURN friend.name

// Find shortest path
MATCH path = shortestPath(
    (start:Person {name: 'Alice'})-[*]-(end:Person {name: 'Charlie'})
)
RETURN path
```

---

## Part 2: Data Preprocessing (4.3)

### Handling Missing Data

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Types of missing data
# MCAR: Missing Completely At Random
# MAR: Missing At Random
# MNAR: Missing Not At Random

# Detection
df.isnull().sum()
df.isnull().mean()  # Percentage missing

# Deletion
df.dropna()  # Drop rows with any NA
df.dropna(axis=1)  # Drop columns
df.dropna(thresh=5)  # Keep rows with at least 5 non-NA

# Imputation
# Mean/Median/Mode
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)

# Forward/Backward fill (time series)
df.fillna(method='ffill')
df.fillna(method='bfill')

# Missing indicator
df['col_missing'] = df['col'].isnull().astype(int)
```

### Handling Outliers

```python
from scipy import stats

# Detection methods
# Z-score method
z_scores = np.abs(stats.zscore(df['column']))
outliers = np.where(z_scores > 3)[0]

# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df['column'] < (Q1 - 1.5 * IQR)) | 
            (df['column'] > (Q3 + 1.5 * IQR)))

# Isolation Forest
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.1)
outliers = clf.fit_predict(df)

# Treatment methods
# Capping/Winsorizing
df['column'] = df['column'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)

# Transformation
df['column_log'] = np.log1p(df['column'])
df['column_sqrt'] = np.sqrt(df['column'])

# Binning
df['column_binned'] = pd.cut(df['column'], bins=5, labels=False)
```

### Data Transformation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# Scaling
# StandardScaler (Z-score normalization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
# Result: mean=0, std=1

# MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)
# Result: range [0, 1]

# RobustScaler (robust to outliers)
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)

# Power Transformation
# Log transform
df['col_log'] = np.log1p(df['col'])

# Box-Cox (positive values only)
pt = PowerTransformer(method='box-cox')
df['col_transformed'] = pt.fit_transform(df[['col']])

# Yeo-Johnson (handles negative values)
pt = PowerTransformer(method='yeo-johnson')
df['col_transformed'] = pt.fit_transform(df[['col']])
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Label Encoding (ordinal data)
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# One-Hot Encoding (nominal data)
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Or using sklearn
ohe = OneHotEncoder(sparse=False)
df_encoded = ohe.fit_transform(df[['category']])

# Ordinal Encoding (ordered categories)
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
df['level_encoded'] = oe.fit_transform(df[['level']])

# Target Encoding (mean encoding)
df['category_target'] = df.groupby('category')['target'].transform('mean')

# Binary Encoding
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['category'])
df_encoded = encoder.fit_transform(df)

# Hashing Trick
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=10, input_type='string')
df_hashed = fh.transform(df['category'].astype(str))
```

### Feature Scaling

```python
# When to scale:
# - Distance-based algorithms (KNN, K-Means, SVM)
# - Neural Networks
# - PCA
# - Gradient Descent optimization

# When NOT to scale:
# - Tree-based models (Decision Trees, Random Forests, XGBoost)

# Scaling pipeline
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

## Part 3: ETL Concepts (4.4)

### ETL Fundamentals

```
ETL = Extract, Transform, Load

Extract: Get data from source systems
Transform: Clean, validate, enrich data
Load: Store in target system (data warehouse)

ETL vs ELT:
- ETL: Transform before loading (traditional)
- ELT: Load then transform (modern, cloud-based)
```

### Extraction

```python
# Database extraction
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="user",
    password="password"
)

query = "SELECT * FROM users WHERE created_at > %s"
df = pd.read_sql_query(query, conn, params=['2024-01-01'])

# API extraction
import requests

response = requests.get('https://api.example.com/data')
data = response.json()

# File extraction
df_csv = pd.read_csv('data.csv')
df_excel = pd.read_excel('data.xlsx')
df_json = pd.read_json('data.json')

# Web scraping
from bs4 import BeautifulSoup

response = requests.get('https://example.com')
soup = BeautifulSoup(response.text, 'html.parser')
```

### Transformation

```python
def transform_data(df):
    """Data transformation pipeline"""
    
    # Data cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=['id', 'date'])
    
    # Data validation
    df = df[df['amount'] > 0]
    df = df[df['date'] >= '2024-01-01']
    
    # Data enrichment
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    
    # Aggregation
    summary = df.groupby('category').agg({
        'amount': ['sum', 'mean', 'count']
    }).reset_index()
    
    return summary
```

### Loading

```python
# Load to database
df.to_sql('target_table', conn, if_exists='append', index=False)

# Load to file
df.to_csv('output.csv', index=False)
df.to_parquet('output.parquet', index=False)

# Load to cloud storage
import boto3

s3 = boto3.client('s3')
df.to_csv('output.csv', index=False)
s3.upload_file('output.csv', 'bucket-name', 'path/output.csv')

# Bulk loading
# PostgreSQL COPY
cursor.copy_expert(
    "COPY target_table FROM STDIN WITH CSV HEADER",
    open('data.csv', 'r')
)
```

### ETL Tools

```
Apache Airflow: Workflow orchestration
Apache NiFi: Data flow automation
Talend: Enterprise ETL
Informatica: Enterprise data integration
dbt: Data transformation (ELT)
AWS Glue: Serverless ETL
Azure Data Factory: Cloud ETL
Google Cloud Dataflow: Stream/batch processing
```

---

## Part 4: Data Pipelines (4.5)

### Pipeline Architecture

```
Components:
1. Source: Data origin (databases, APIs, files)
2. Ingestion: Data collection
3. Processing: Transformation, validation
4. Storage: Data warehouse, data lake
5. Consumption: Analytics, ML, reporting

Patterns:
- Batch processing: Process data in batches
- Stream processing: Process data in real-time
- Lambda architecture: Batch + Stream
- Kappa architecture: Stream only
```

### Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'etl_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

def extract():
    # Extract data
    pass

def transform():
    # Transform data
    pass

def load():
    # Load data
    pass

extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform,
    dag=dag
)

load_task = PythonOperator(
    task_id='load',
    python_callable=load,
    dag=dag
)

extract_task >> transform_task >> load_task
```

### Pipeline Design Principles

```
1. Idempotency: Running multiple times produces same result
2. Fault Tolerance: Handle failures gracefully
3. Monitoring: Track pipeline health
4. Logging: Record all operations
5. Versioning: Track pipeline changes
6. Testing: Validate pipeline output

Error Handling:
- Retry logic
- Dead letter queues
- Alerting
- Rollback mechanisms
```

### Modern Data Stack

```
Data Lake: Raw data storage (S3, ADLS, GCS)
Data Warehouse: Structured data (Snowflake, BigQuery, Redshift)
Data Lakehouse: Combined approach (Delta Lake, Iceberg)

Table Formats:
- Delta Lake: ACID transactions on data lakes
- Apache Iceberg: Open table format
- Apache Hudi: Incremental processing

Streaming:
- Apache Kafka: Event streaming
- Apache Spark Streaming: Stream processing
- AWS Kinesis: Real-time streaming
```

---

## 💻 Python Code Examples

```python
# === Complete ETL Pipeline Example ===

import pandas as pd
import sqlite3
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ETLPipeline:
    """Complete ETL pipeline implementation"""
    
    def __init__(self, source_conn, target_conn):
        self.source_conn = source_conn
        self.target_conn = target_conn
    
    def extract(self, query):
        """Extract data from source"""
        logger.info(f"Extracting data with query: {query}")
        df = pd.read_sql_query(query, self.source_conn)
        logger.info(f"Extracted {len(df)} rows")
        return df
    
    def transform(self, df):
        """Transform data"""
        logger.info("Transforming data")
        
        # Data cleaning
        df = df.drop_duplicates()
        df = df.dropna(subset=['id', 'date'])
        
        # Data validation
        df = df[df['amount'] > 0]
        
        # Data transformation
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Feature engineering
        df['amount_log'] = np.log1p(df['amount'])
        
        logger.info(f"Transformed {len(df)} rows")
        return df
    
    def load(self, df, table_name):
        """Load data to target"""
        logger.info(f"Loading data to {table_name}")
        df.to_sql(table_name, self.target_conn, 
                  if_exists='replace', index=False)
        logger.info("Load complete")
    
    def run(self, query, target_table):
        """Run complete pipeline"""
        try:
            df = self.extract(query)
            df = self.transform(df)
            self.load(df, target_table)
            logger.info("Pipeline completed successfully")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

# Usage
if __name__ == "__main__":
    source_conn = sqlite3.connect('source.db')
    target_conn = sqlite3.connect('target.db')
    
    pipeline = ETLPipeline(source_conn, target_conn)
    pipeline.run(
        query="SELECT * FROM raw_data WHERE date >= '2024-01-01'",
        target_table="processed_data"
    )
```

---

## 📊 Summary Tables

### NoSQL Database Types

| Type | Example | Use Case |
|------|---------|----------|
| Document | MongoDB | Content management |
| Key-Value | Redis | Caching, sessions |
| Column-Family | Cassandra | Time series, IoT |
| Graph | Neo4j | Social networks, recommendations |

### Preprocessing Techniques

| Technique | Method | When to Use |
|-----------|--------|-------------|
| Missing Data | Imputation, Deletion | Any dataset |
| Outliers | Capping, Transformation | Skewed distributions |
| Scaling | Standard, MinMax | Distance-based algorithms |
| Encoding | One-Hot, Label | Categorical variables |

### ETL Components

| Component | Tools | Purpose |
|-----------|-------|---------|
| Extract | SQL, APIs, Scraping | Get data from source |
| Transform | Pandas, Spark | Clean and transform |
| Load | SQL COPY, Bulk insert | Store in target |
| Orchestrate | Airflow, Prefect | Schedule and monitor |

---

## 🎯 ML Applications

| Data Engineering Concept | ML Application |
|-------------------------|----------------|
| Data Preprocessing | Feature engineering |
| ETL Pipelines | Training data preparation |
| Data Quality | Model reliability |
| Feature Stores | Reusable features |
| Streaming | Real-time predictions |

---

**Status:** ✅ Complete
**Next:** DSA Foundations
