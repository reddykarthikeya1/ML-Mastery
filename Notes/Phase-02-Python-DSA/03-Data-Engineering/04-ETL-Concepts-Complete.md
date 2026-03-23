# 4.4 ETL Concepts

## 🎯 Quick Overview
- **ETL**: Extract, Transform, Load
- **Data Integration**: Combine data from multiple sources
- **Data Quality**: Ensure clean, validated data
- **Foundation for**: Data warehouses, analytics, ML pipelines

---

## 1. ETL Fundamentals

### What is ETL?

```
ETL = Extract, Transform, Load

Extract: Get data from source systems
Transform: Clean, validate, enrich data
Load: Store in target system (data warehouse)

ETL vs ELT:
- ETL: Transform before loading (traditional)
- ELT: Load then transform (modern, cloud-based)
```

### ETL Architecture Patterns

```
Batch ETL:
- Process data in scheduled batches
- Example: Nightly data warehouse updates

Streaming ETL:
- Process data in real-time
- Example: Real-time analytics

Lambda Architecture:
- Combines batch and streaming
- Batch layer for accuracy
- Speed layer for real-time

Kappa Architecture:
- Streaming only
- Simpler than Lambda
```

### ETL Tools

```
Open Source:
- Apache Airflow (orchestration)
- Apache NiFi (data flow)
- Talend Open Studio
- dbt (transformation)

Commercial:
- Informatica
- Microsoft SSIS
- Oracle Data Integrator

Cloud:
- AWS Glue
- Azure Data Factory
- Google Cloud Dataflow
```

---

## 2. Extraction

### Data Sources

```python
# Database extraction
import psycopg2
import pandas as pd

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

response = requests.get('https://api.example.com/data', 
                       params={'page': 1, 'limit': 100})
data = response.json()

# File extraction
df_csv = pd.read_csv('data.csv')
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df_json = pd.read_json('data.json')
df_parquet = pd.read_parquet('data.parquet')

# Web scraping
from bs4 import BeautifulSoup

response = requests.get('https://example.com')
soup = BeautifulSoup(response.text, 'html.parser')
```

### Full vs Incremental Extraction

```python
# Full extraction (all data)
def full_extract():
    df = pd.read_sql_query("SELECT * FROM users", conn)
    return df

# Incremental extraction (only new/changed data)
def incremental_extract(last_extract_date):
    query = """
        SELECT * FROM users 
        WHERE updated_at > %s
    """
    df = pd.read_sql_query(query, conn, params=[last_extract_date])
    return df

# Change Data Capture (CDC)
def cdc_extract():
    # Track changes using timestamps, version numbers, or database logs
    query = """
        SELECT * FROM users 
        WHERE version > %s
    """
    df = pd.read_sql_query(query, conn, params=[last_version])
    return df
```

---

## 3. Transformation

### Data Cleaning

```python
def clean_data(df):
    """Data cleaning transformations"""
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna({'column1': 0, 'column2': 'Unknown'})
    
    # Standardize formats
    df['email'] = df['email'].str.lower().str.strip()
    df['phone'] = df['phone'].str.replace(r'\D', '', regex=True)
    
    # Fix data types
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Validate data
    df = df[df['age'] >= 18]
    df = df[df['amount'] > 0]
    
    return df
```

### Data Enrichment

```python
def enrich_data(df):
    """Enrich data with additional information"""
    
    # Add calculated columns
    df['full_name'] = df['first_name'] + ' ' + df['last_name']
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 65, 100], 
                             labels=['Young', 'Adult', 'Middle', 'Senior'])
    
    # Add date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Geocoding (example)
    def get_city_from_zip(zip_code):
        # API call or lookup table
        return city
    
    df['city'] = df['zip_code'].apply(get_city_from_zip)
    
    return df
```

### Aggregation

```python
def aggregate_data(df):
    """Aggregate data to summary level"""
    
    # Group by and aggregate
    summary = df.groupby(['category', 'region']).agg({
        'amount': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'customer_id': 'nunique'
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['category', 'region', 'total_amount', 'avg_amount', 
                       'order_count', 'total_quantity', 'unique_customers']
    
    # Add calculated metrics
    summary['amount_per_customer'] = summary['total_amount'] / summary['unique_customers']
    
    return summary
```

---

## 4. Loading

### Load Strategies

```python
# Full load (overwrite)
def full_load(df, table_name):
    df.to_sql(table_name, conn, if_exists='replace', index=False)

# Incremental load (append)
def incremental_load(df, table_name):
    df.to_sql(table_name, conn, if_exists='append', index=False)

# Upsert (update or insert)
def upsert_load(df, table_name, key_columns):
    # Using PostgreSQL ON CONFLICT
    df.to_sql(table_name + '_temp', conn, if_exists='replace', index=False)
    
    merge_query = f"""
        INSERT INTO {table_name}
        SELECT * FROM {table_name}_temp
        ON CONFLICT ({', '.join(key_columns)}) 
        DO UPDATE SET
            {', '.join([f'{col} = EXCLUDED.{col}' for col in df.columns if col not in key_columns])}
    """
    
    cursor.execute(merge_query)
    cursor.execute(f"DROP TABLE {table_name}_temp")
    conn.commit()
```

### Bulk Loading

```python
# PostgreSQL COPY
def bulk_load_postgres(df, table_name):
    from io import StringIO
    
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    
    cursor.copy_expert(
        f"COPY {table_name} FROM STDIN WITH CSV",
        buffer
    )
    conn.commit()

# MySQL LOAD DATA
def bulk_load_mysql(df, table_name):
    df.to_csv('/tmp/data.csv', index=False, header=False)
    
    query = f"""
        LOAD DATA LOCAL INFILE '/tmp/data.csv'
        INTO TABLE {table_name}
        FIELDS TERMINATED BY ','
        LINES TERMINATED BY '\n'
    """
    cursor.execute(query)
    conn.commit()
```

---

## 5. ETL Tools and Technologies

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
    # Extract data from source
    pass

def transform():
    # Transform data
    pass

def load():
    # Load data to target
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

### dbt (Data Build Tool)

```sql
-- dbt model: models/customers.sql
{{
    config(
        materialized='table',
        unique_key='customer_id'
    )
}}

with source_data as (
    select * from {{ source('raw', 'customers') }}
),

cleaned as (
    select
        customer_id,
        lower(email) as email,
        trim(name) as name,
        created_at
    from source_data
    where customer_id is not null
),

enriched as (
    select
        *,
        extract(year from created_at) as signup_year,
        case
            when signup_year >= 2024 then 'New'
            else 'Existing'
        end as customer_type
    from cleaned
)

select * from enriched
```

---

## 💻 Python Code Examples

```python
# === Complete ETL Pipeline ===

import pandas as pd
import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETLPipeline:
    """Complete ETL pipeline implementation"""
    
    def __init__(self, source_conn, target_conn):
        self.source_conn = source_conn
        self.target_conn = target_conn
    
    def extract(self, query, params=None):
        """Extract data from source"""
        logger.info(f"Extracting data with query: {query}")
        df = pd.read_sql_query(query, self.source_conn, params=params)
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
        df = df[df['date'] >= '2024-01-01']
        
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
    
    def run(self, query, target_table, params=None):
        """Run complete pipeline"""
        try:
            df = self.extract(query, params)
            df = self.transform(df)
            self.load(df, target_table)
            logger.info("Pipeline completed successfully")
            return True
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

# Usage
if __name__ == "__main__":
    source_conn = sqlite3.connect('source.db')
    target_conn = sqlite3.connect('target.db')
    
    pipeline = ETLPipeline(source_conn, target_conn)
    pipeline.run(
        query="SELECT * FROM raw_data WHERE date >= ?",
        target_table="processed_data",
        params=['2024-01-01']
    )
```

---

## 📊 Summary Tables

### ETL Components

| Component | Tools | Purpose |
|-----------|-------|---------|
| Extract | SQL, APIs, Files | Get data from source |
| Transform | Pandas, Spark, SQL | Clean and transform |
| Load | SQL COPY, Bulk insert | Store in target |
| Orchestrate | Airflow, Prefect | Schedule and monitor |

### Load Strategies

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| Full Load | Small datasets, initial load | Simple | Slow, resource-intensive |
| Incremental | Large datasets, regular updates | Fast, efficient | Complex logic |
| Upsert | Need to update existing records | Accurate | More complex |

---

## 🎯 ML Applications

| ETL Concept | ML Application |
|-------------|----------------|
| Data Extraction | Training data collection |
| Data Cleaning | Data preprocessing |
| Feature Engineering | ML feature creation |
| Data Loading | Feature store population |
| Pipeline Orchestration | ML pipeline automation |

---

## ❓ Quick Check Questions

1. What is the fundamental difference between ETL and ELT?
2. Why is Incremental Extraction often preferred over Full Extraction in large data systems?
3. What is Change Data Capture (CDC)?
4. What happens during an "Upsert" load strategy?
5. Name two popular open-source tools for ETL orchestration and data transformation.

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **ETL (Extract, Transform, Load)** transforms data on a separate processing server before loading it into the data warehouse. **ELT (Extract, Load, Transform)** loads raw data directly into the target system (like a modern cloud data warehouse) and uses the target system's compute power to do the transformation.
2. **Incremental extraction** only pulls data that has changed or been added since the last run. For large datasets, this is vastly faster and less resource-intensive than pulling the entire dataset every time (Full Extraction).
3. **Change Data Capture (CDC)** is a set of software design patterns used to determine and track the data that has changed so that action can be taken using the changed data (often using database logs or version numbers).
4. An **Upsert** (Update or Insert) checks if a record already exists based on a key. If it exists, it updates the record; if it does not exist, it inserts a new record.
5. **Apache Airflow** is a popular tool for ETL orchestration, and **dbt (Data Build Tool)** is popular for data transformation.

</details>
---

**Status:** ✅ Complete
**Next:** Data Pipelines
