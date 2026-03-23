# 4.2 NoSQL Basics

## 🎯 Quick Overview
- **NoSQL**: Non-relational databases for specific use cases
- **Types**: Document, Key-Value, Column-Family, Graph
- **CAP Theorem**: Consistency, Availability, Partition Tolerance
- **Foundation for**: Modern data infrastructure, caching, real-time systems

---

## 1. NoSQL Overview

### SQL vs NoSQL

```
SQL Databases:
- Table-based structure
- Schema-based (rigid)
- ACID compliant
- Vertical scaling
- Examples: MySQL, PostgreSQL, Oracle

NoSQL Databases:
- Document/Key-Value/Graph/Column-based
- Schema-less (flexible)
- BASE model (Basically Available, Soft state, Eventual consistency)
- Horizontal scaling
- Examples: MongoDB, Redis, Cassandra, Neo4j
```

### CAP Theorem

```
Choose 2 of 3:
- Consistency: All nodes see same data at same time
- Availability: Every request gets response (success or failure)
- Partition Tolerance: System works despite network failures

SQL: Typically CP (Consistency + Partition Tolerance)
NoSQL: Typically AP (Availability + Partition Tolerance)
```

### Types of NoSQL Databases

| Type | Example | Use Case |
|------|---------|----------|
| Document | MongoDB | Content management, catalogs |
| Key-Value | Redis | Caching, sessions, real-time |
| Column-Family | Cassandra | Time series, IoT, big data |
| Graph | Neo4j | Social networks, recommendations |

---

## 2. Document Databases (MongoDB)

### Document Structure

```javascript
{
    _id: ObjectId("507f1f77bcf86cd799439011"),
    name: "Alice",
    age: 25,
    skills: ["Python", "ML", "Data Science"],
    address: {
        city: "NYC",
        zip: "10001"
    },
    created_at: ISODate("2024-01-01")
}
```

### CRUD Operations

```javascript
// Create (Insert)
db.users.insertOne({
    name: "Bob",
    age: 30,
    email: "bob@example.com"
})

db.users.insertMany([
    {name: "Charlie", age: 35},
    {name: "Diana", age: 28}
])

// Read (Query)
db.users.find()  // All documents
db.users.findOne({name: "Alice"})  // Single document
db.users.find({age: {$gt: 25}})  // age > 25
db.users.find({age: {$gte: 25, $lte: 35}})  // 25 <= age <= 35
db.users.find({name: {$in: ["Alice", "Bob"]}})  // name IN (...)
db.users.find({name: /A/i})  // Regex search

// Update
db.users.updateOne(
    {name: "Bob"},
    {$set: {age: 31}}
)

db.users.updateMany(
    {age: {$lt: 30}},
    {$set: {status: "young"}}
)

// Delete
db.users.deleteOne({name: "Bob"})
db.users.deleteMany({age: {$lt: 18}})
```

### Aggregation Pipeline

```javascript
db.orders.aggregate([
    // Stage 1: Match
    {$match: {status: "completed", date: {$gte: "2024-01-01"}}},
    
    // Stage 2: Group
    {$group: {
        _id: "$customer_id",
        total: {$sum: "$amount"},
        count: {$sum: 1},
        avg: {$avg: "$amount"}
    }},
    
    // Stage 3: Sort
    {$sort: {total: -1}},
    
    // Stage 4: Limit
    {$limit: 10},
    
    // Stage 5: Lookup (Join)
    {$lookup: {
        from: "customers",
        localField: "_id",
        foreignField: "_id",
        as: "customer_info"
    }}
])
```

### Indexing

```javascript
// Create index
db.users.createIndex({email: 1})  // Ascending
db.users.createIndex({name: 1, age: -1})  // Compound

// Create unique index
db.users.createIndex({email: 1}, {unique: true})

// Create TTL index (auto-expire)
db.sessions.createIndex({expires_at: 1}, {expireAfterSeconds: 3600})

// View indexes
db.users.getIndexes()

// Drop index
db.users.dropIndex("email_1")
```

---

## 3. Key-Value Stores (Redis)

### Basic Operations

```python
import redis

# Connect
r = redis.Redis(host='localhost', port=6379, db=0)

# String operations
r.set('name', 'Alice')
r.get('name')  # b'Alice'
r.mset({'name': 'Alice', 'age': '25'})
r.mget('name', 'age')

# Expiration
r.setex('cache:key', 3600, 'value')  # Expires in 1 hour
r.expire('key', 3600)  # Set expiration on existing key
r.ttl('key')  # Time to live in seconds

# Counter operations
r.set('counter', 0)
r.incr('counter')  # Increment by 1
r.incrby('counter', 5)  # Increment by 5
r.decr('counter')  # Decrement by 1
```

### List Operations

```python
# List (linked list)
r.lpush('mylist', 'a', 'b', 'c')  # Push to left
r.rpush('mylist', 'd', 'e')  # Push to right
r.lrange('mylist', 0, -1)  # Get all elements
r.lpop('mylist')  # Pop from left
r.rpop('mylist')  # Pop from right
r.llen('mylist')  # Length

# Queue pattern (FIFO)
r.rpush('queue', 'task1', 'task2')
task = r.lpop('queue')  # Get first task

# Stack pattern (LIFO)
r.push('stack', 'item1', 'item2')
item = r.pop('stack')  # Get last item
```

### Hash Operations

```python
# Hash (object)
r.hset('user:1', mapping={'name': 'Alice', 'age': '25', 'city': 'NYC'})
r.hget('user:1', 'name')  # b'Alice'
r.hgetall('user:1')  # {b'name': b'Alice', b'age': b'25', ...}
r.hincrby('user:1', 'age', 1)  # Increment age by 1
r.hkeys('user:1')  # All field names
r.hvals('user:1')  # All values
r.hlen('user:1')  # Number of fields
```

### Set Operations

```python
# Set (unordered, unique)
r.sadd('tags', 'python', 'ml', 'ai', 'python')  # 'python' added once
r.smembers('tags')  # {b'python', b'ml', b'ai'}
r.sismember('tags', 'python')  # 1 (True)
r.scard('tags')  # Cardinality (count)

# Set operations
r.sadd('set1', 'a', 'b', 'c')
r.sadd('set2', 'b', 'c', 'd')
r.sunion('set1', 'set2')  # Union: {a, b, c, d}
r.sinter('set1', 'set2')  # Intersection: {b, c}
r.sdiff('set1', 'set2')  # Difference: {a}
```

### Sorted Set Operations

```python
# Sorted Set (scored, ordered)
r.zadd('leaderboard', {'Alice': 100, 'Bob': 85, 'Charlie': 95})
r.zrange('leaderboard', 0, -1, withscores=True)  # All ranked
r.zrevrange('leaderboard', 0, 9, withscores=True)  # Top 10
r.zscore('leaderboard', 'Alice')  # 100.0
r.zrank('leaderboard', 'Alice')  # Rank (0-based)
r.zincrby('leaderboard', 10, 'Bob')  # Increment score
```

### Use Cases

```python
# Caching
def get_user_data(user_id):
    # Try cache first
    cached = r.get(f'user:{user_id}')
    if cached:
        return json.loads(cached)
    
    # Query database
    user_data = db.query('SELECT * FROM users WHERE id = ?', user_id)
    
    # Cache for 1 hour
    r.setex(f'user:{user_id}', 3600, json.dumps(user_data))
    
    return user_data

# Session storage
def create_session(user_id):
    session_id = generate_uuid()
    r.setex(f'session:{session_id}', 86400, user_id)  # 24 hours
    return session_id

# Rate limiting
def is_rate_limited(user_id, max_requests=100, window=3600):
    key = f'rate_limit:{user_id}'
    current = r.get(key)
    
    if current and int(current) >= max_requests:
        return True
    
    r.incr(key)
    r.expire(key, window)
    return False
```

---

## 4. Column-Family Stores (Cassandra)

### Data Model

```
Keyspace (Database)
  └── Table (Column Family)
      ├── Partition Key (for distribution)
      └── Clustering Columns (for sorting)
```

### CQL (Cassandra Query Language)

```sql
-- Create keyspace
CREATE KEYSPACE mydb 
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 3
};

USE mydb;

-- Create table
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    email TEXT,
    created_at TIMESTAMP
);

-- Create table with composite key
CREATE TABLE sensor_data (
    sensor_id UUID,
    timestamp TIMESTAMP,
    value DOUBLE,
    PRIMARY KEY (sensor_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Insert data
INSERT INTO users (user_id, name, email) 
VALUES (uuid(), 'Alice', 'alice@example.com');

-- Query
SELECT * FROM users WHERE user_id = ?;

-- Query with clustering
SELECT * FROM sensor_data 
WHERE sensor_id = ? 
AND timestamp > '2024-01-01'
ORDER BY timestamp DESC
LIMIT 100;
```

---

## 5. Graph Databases (Neo4j)

### Cypher Query Language

```cypher
// Create nodes
CREATE (p:Person {name: 'Alice', age: 25})
CREATE (f:Person {name: 'Bob', age: 30})

// Create relationship
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
CREATE (a)-[:FRIENDS_WITH {since: 2020}]->(b)

// Query - Find friends
MATCH (p:Person {name: 'Alice'})-[:FRIENDS_WITH]->(friend)
RETURN friend.name, friend.age

// Query - Find path
MATCH path = shortestPath(
    (start:Person {name: 'Alice'})-[*]-(end:Person {name: 'Charlie'})
)
RETURN path

// Query - Friends of friends
MATCH (p:Person {name: 'Alice'})-[:FRIENDS_WITH]->()-[:FRIENDS_WITH]->(fof)
WHERE fof.name <> 'Alice'
RETURN DISTINCT fof.name

// Update
MATCH (p:Person {name: 'Alice'})
SET p.age = 26

// Delete
MATCH (p:Person {name: 'Bob'})
DETACH DELETE p
```

### Use Cases

```
Social Networks:
- Friend recommendations
- Community detection

Recommendation Systems:
- "Users who bought this also bought..."
- Content-based recommendations

Fraud Detection:
- Detect suspicious patterns
- Network analysis

Knowledge Graphs:
- Entity relationships
- Semantic search
```

---

## 💻 Python Code Examples

```python
# === MongoDB Example ===

from pymongo import MongoClient
from datetime import datetime

# Connect
client = MongoClient('mongodb://localhost:27017/')
db = client['mydb']
collection = db['users']

# Insert
collection.insert_one({
    'name': 'Alice',
    'age': 25,
    'created_at': datetime.now()
})

# Query
users = collection.find({'age': {'$gt': 25}})
for user in users:
    print(user)

# Aggregation
pipeline = [
    {'$match': {'age': {'$gte': 18}}},
    {'$group': {
        '_id': '$city',
        'count': {'$sum': 1},
        'avg_age': {'$avg': '$age'}
    }},
    {'$sort': {'count': -1}}
]
results = collection.aggregate(pipeline)

# === Redis Example ===

import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Caching pattern
def get_user(user_id):
    # Try cache
    cached = r.get(f'user:{user_id}')
    if cached:
        return json.loads(cached)
    
    # Query database (simulated)
    user = {'id': user_id, 'name': 'Alice', 'email': 'alice@example.com'}
    
    # Cache for 1 hour
    r.setex(f'user:{user_id}', 3600, json.dumps(user))
    
    return user

# Rate limiting
def check_rate_limit(user_id, limit=100, window=3600):
    key = f'rate:{user_id}'
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, window)
    results = pipe.execute()
    
    return results[0] <= limit

# === Neo4j Example ===

from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.run(query, parameters)

# Usage
conn = Neo4jConnection('bolt://localhost:7687', 'neo4j', 'password')

# Find friends
result = conn.query("""
    MATCH (p:Person {name: $name})-[:FRIENDS_WITH]->(friend)
    RETURN friend.name as name, friend.age as age
""", {'name': 'Alice'})

for record in result:
    print(f"Friend: {record['name']}, Age: {record['age']}")

conn.close()
```

---

## 📊 Summary Tables

### NoSQL Database Comparison

| Database | Type | Best For | Scaling |
|----------|------|----------|---------|
| MongoDB | Document | Content, catalogs | Horizontal |
| Redis | Key-Value | Caching, real-time | Vertical/Horizontal |
| Cassandra | Column-Family | Time series, IoT | Horizontal |
| Neo4j | Graph | Relationships, networks | Vertical |

### When to Use NoSQL

| Use Case | Recommended Type |
|----------|-----------------|
| Caching | Redis (Key-Value) |
| Content Management | MongoDB (Document) |
| Real-time Analytics | Redis/Cassandra |
| Social Networks | Neo4j (Graph) |
| IoT/Time Series | Cassandra (Column) |
| Product Catalogs | MongoDB (Document) |

---

## 🎯 ML Applications

| NoSQL Type | ML Application |
|------------|----------------|
| Document | Feature storage, model metadata |
| Key-Value | Model caching, session storage |
| Column-Family | Time series features, IoT data |
| Graph | Knowledge graphs, GNNs |

---

**Status:** ✅ Complete
**Next:** Data Preprocessing and Cleaning
