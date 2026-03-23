# 4.5 Data Pipelines

## 🎯 Quick Overview
- **Data Pipelines**: Automated data workflows
- **Orchestration**: Schedule and manage pipelines
- **Data Quality**: Ensure reliable data
- **Foundation for**: Analytics, ML, business intelligence

---

## 1. Pipeline Architecture

### Pipeline Components

```
1. Source: Data origin (databases, APIs, files)
2. Ingestion: Data collection
3. Processing: Transformation, validation
4. Storage: Data warehouse, data lake
5. Consumption: Analytics, ML, reporting
```

### Pipeline Patterns

```
Batch Processing:
- Process data in scheduled batches
- Example: Nightly ETL jobs
- Tools: Airflow, Luigi

Stream Processing:
- Process data in real-time
- Example: Real-time fraud detection
- Tools: Kafka, Spark Streaming, Flink

Lambda Architecture:
- Combines batch and stream
- Batch layer for accuracy
- Speed layer for real-time
- Complex but comprehensive

Kappa Architecture:
- Stream only
- Simpler than Lambda
- All data treated as stream
```

### DAGs (Directed Acyclic Graphs)

```
DAG = Collection of tasks with dependencies

Properties:
- Directed: Tasks have direction
- Acyclic: No circular dependencies
- Graph: Tasks and dependencies

Example:
Task A → Task B → Task C
           ↓
        Task D

B and C depend on A
D depends on B
```

---

## 2. Pipeline Design

### Key Principles

```
1. Idempotency: Running multiple times produces same result
2. Fault Tolerance: Handle failures gracefully
3. Monitoring: Track pipeline health
4. Logging: Record all operations
5. Versioning: Track pipeline changes
6. Testing: Validate pipeline output
```

### Error Handling

```python
# Retry logic
from airflow.utils.retry import retry

@retry(max_retries=3, retry_delay=timedelta(minutes=5))
def extract_data():
    # May fail due to network issues
    pass

# Dead letter queue
def process_record(record):
    try:
        # Process record
        pass
    except Exception as e:
        # Send to dead letter queue
        send_to_dlq(record, str(e))

# Alerting
def send_alert(message):
    # Send email/Slack alert
    pass

def run_pipeline():
    try:
        # Run pipeline
        pass
    except Exception as e:
        send_alert(f"Pipeline failed: {str(e)}")
        raise
```

### Data Lineage

```
Track data from source to consumption:

Source → Transform 1 → Transform 2 → Target

Benefits:
- Debugging: Trace errors to source
- Compliance: Know data origin
- Impact Analysis: Understand downstream effects
```

---

## 3. Pipeline Tools

### Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'data_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['etl', 'production']
)

# Python task
def extract():
    import pandas as pd
    df = pd.read_csv('data.csv')
    return df.to_json()

extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract,
    dag=dag
)

# Bash task
transform_task = BashOperator(
    task_id='transform',
    bash_command='python /scripts/transform.py',
    dag=dag
)

# Task with dependencies
load_task = BashOperator(
    task_id='load',
    bash_command='python /scripts/load.py',
    dag=dag
)

# Set dependencies
extract_task >> transform_task >> load_task
```

### Prefect

```python
from prefect import task, Flow, Parameter
from prefect.executors import LocalDaskExecutor

@task
def extract(source_path):
    import pandas as pd
    return pd.read_csv(source_path)

@task
def transform(df):
    df = df.dropna()
    df['processed'] = True
    return df

@task
def load(df, target_path):
    df.to_csv(target_path, index=False)

with Flow("ETL Pipeline", executor=LocalDaskExecutor()) as flow:
    source = Parameter("source_path", default="data.csv")
    target = Parameter("target_path", default="output.csv")
    
    data = extract(source)
    transformed = transform(data)
    load(transformed, target)

# Run
flow.run()
```

### Dagster

```python
from dagster import job, op, schedule

@op
def extract(context):
    context.log.info("Extracting data")
    return {"data": "extracted"}

@op
def transform(context, data):
    context.log.info("Transforming data")
    return {"data": "transformed"}

@op
def load(context, data):
    context.log.info("Loading data")
    return True

@job
def etl_job():
    load(transform(extract()))

@schedule(cron_schedule="0 0 * * *", job=etl_job)
def daily_etl_schedule():
    return {}
```

---

## 4. Data Quality in Pipelines

### Validation Frameworks

```python
# Great Expectations
import great_expectations as ge

def validate_data(df):
    gdf = ge.from_pandas(df)
    
    # Define expectations
    gdf.expect_column_to_exist('id')
    gdf.expect_column_values_to_not_be_null('id')
    gdf.expect_column_values_to_be_unique('id')
    gdf.expect_column_values_to_be_between('age', min_value=0, max_value=150)
    gdf.expect_column_values_to_match_regex('email', r'^[^@]+@[^@]+\.[^@]+$')
    
    # Validate
    results = gdf.validate()
    
    if not results.success:
        raise Exception("Data validation failed")
    
    return results

# Usage
validate_data(df)
```

### Data Testing

```python
# Unit tests for data
def test_data_quality(df):
    # Test row count
    assert len(df) > 0, "DataFrame is empty"
    
    # Test for duplicates
    assert df.duplicated().sum() == 0, "Duplicates found"
    
    # Test for nulls
    assert df.isnull().sum().sum() == 0, "Null values found"
    
    # Test data types
    assert df['id'].dtype == 'int64', "Wrong type for id"
    
    # Test value ranges
    assert df['age'].between(0, 150).all(), "Invalid age values"
    
    print("All tests passed!")

# Integration tests
def test_pipeline():
    # Run pipeline on test data
    result = run_pipeline('test_data.csv')
    
    # Validate output
    assert len(result) > 0
    assert 'expected_column' in result.columns
    
    print("Pipeline test passed!")
```

---

## 5. Modern Data Stack

### Data Lakes vs Warehouses

```
Data Lake:
- Raw data storage
- Schema-on-read
- Examples: S3, ADLS, GCS
- Use: Data science, exploration

Data Warehouse:
- Processed data
- Schema-on-write
- Examples: Snowflake, Redshift, BigQuery
- Use: Analytics, reporting

Data Lakehouse:
- Combines lake and warehouse
- ACID transactions on lake
- Examples: Delta Lake, Iceberg, Hudi
```

### Table Formats

```python
# Delta Lake
from delta.tables import DeltaTable

# Write to Delta Lake
df.write.format("delta").save("/data/delta_table")

# Read Delta Lake
delta_df = spark.read.format("delta").load("/data/delta_table")

# Time travel
delta_df = spark.read.format("delta").option("versionAsOf", 1).load("/data/delta_table")

# Apache Iceberg
df.write.format("iceberg").save("/data/iceberg_table")

# Apache Hudi
df.write.format("hudi").save("/data/hudi_table")
```

### Streaming Pipelines

```python
# Kafka consumer
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'topic_name',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    process_message(message.value)

# Spark Streaming
from pyspark.streaming import StreamingContext

ssc = StreamingContext(sc, batchDuration=1)  # 1 second batches

lines = ssc.socketTextStream("localhost", 9999)
words = lines.flatMap(lambda line: line.split(" "))

word_counts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
word_counts.pprint()

ssc.start()
ssc.awaitTermination()
```

---

## 💻 Python Code Examples

```python
# === Complete Pipeline with Airflow ===

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack import SlackAPIOperator
from datetime import datetime, timedelta
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_feature_pipeline',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False
)

def extract_data(**context):
    """Extract data from source"""
    logger.info("Extracting data")
    # Extract logic here
    return {'status': 'success'}

def validate_data(**context):
    """Validate extracted data"""
    logger.info("Validating data")
    # Validation logic here
    return {'status': 'valid'}

def transform_data(**context):
    """Transform data"""
    logger.info("Transforming data")
    # Transformation logic here
    return {'status': 'transformed'}

def load_data(**context):
    """Load data to feature store"""
    logger.info("Loading data")
    # Load logic here
    return {'status': 'loaded'}

def send_success_notification(**context):
    """Send Slack notification on success"""
    # Slack notification logic
    pass

extract = PythonOperator(
    task_id='extract',
    python_callable=extract_data,
    dag=dag
)

validate = PythonOperator(
    task_id='validate',
    python_callable=validate_data,
    dag=dag
)

transform = PythonOperator(
    task_id='transform',
    python_callable=transform_data,
    dag=dag
)

load = PythonOperator(
    task_id='load',
    python_callable=load_data,
    dag=dag
)

notify = PythonOperator(
    task_id='notify',
    python_callable=send_success_notification,
    trigger_rule='all_success',
    dag=dag
)

# Set dependencies
extract >> validate >> transform >> load >> notify

# === Data Quality Pipeline ===

class DataQualityPipeline:
    """Pipeline with data quality checks"""
    
    def __init__(self, df):
        self.df = df
        self.errors = []
    
    def check_completeness(self, column, threshold=0.95):
        """Check column completeness"""
        completeness = 1 - (self.df[column].isnull().sum() / len(self.df))
        if completeness < threshold:
            self.errors.append(f"Column {column} completeness {completeness:.2%} < {threshold:.2%}")
        return completeness >= threshold
    
    def check_uniqueness(self, column):
        """Check column uniqueness"""
        duplicates = self.df[column].duplicated().sum()
        if duplicates > 0:
            self.errors.append(f"Column {column} has {duplicates} duplicates")
        return duplicates == 0
    
    def check_validity(self, column, validation_func):
        """Check column validity"""
        invalid = ~self.df[column].apply(validation_func)
        invalid_count = invalid.sum()
        if invalid_count > 0:
            self.errors.append(f"Column {column} has {invalid_count} invalid values")
        return invalid_count == 0
    
    def run_all_checks(self):
        """Run all quality checks"""
        checks = [
            self.check_completeness('id', 1.0),
            self.check_uniqueness('id'),
            self.check_completeness('email', 0.95),
            self.check_validity('age', lambda x: 0 <= x <= 150),
            self.check_validity('email', lambda x: '@' in str(x))
        ]
        
        if all(checks):
            logger.info("All quality checks passed")
            return True
        else:
            logger.error(f"Quality checks failed: {self.errors}")
            return False

# Usage
pipeline = DataQualityPipeline(df)
if pipeline.run_all_checks():
    # Proceed with pipeline
    pass
else:
    # Handle errors
    raise Exception(f"Data quality checks failed: {pipeline.errors}")
```

---

## 📊 Summary Tables

### Pipeline Tools Comparison

| Tool | Type | Best For | Learning Curve |
|------|------|----------|---------------|
| Airflow | Orchestrator | Complex workflows | Medium |
| Prefect | Orchestrator | Modern Python pipelines | Low |
| Dagster | Orchestrator | Data-aware pipelines | Medium |
| Luigi | Orchestrator | Batch pipelines | Low |
| dbt | Transformation | SQL transformations | Low |

### Data Quality Checks

| Check Type | Method | Tool |
|------------|--------|------|
| Completeness | Null checks | Great Expectations |
| Uniqueness | Duplicate detection | Custom SQL |
| Validity | Range/format checks | Deequ |
| Consistency | Cross-table checks | Custom logic |
| Freshness | Timestamp checks | Airflow sensors |

---

## 🎯 ML Applications

| Pipeline Concept | ML Application |
|-----------------|----------------|
| Feature Pipeline | Feature engineering automation |
| Training Pipeline | Model training automation |
| Inference Pipeline | Real-time predictions |
| Monitoring Pipeline | Model drift detection |
| Data Quality | Training data validation |

---

**Status:** ✅ Complete
**Next:** DSA Foundations
