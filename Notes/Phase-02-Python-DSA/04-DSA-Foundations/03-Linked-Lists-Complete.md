# 5.3 Linked Lists

## 🎯 Quick Overview
- **Linked List**: Nodes connected by pointers
- **Singly**: One direction
- **Doubly**: Both directions
- **Foundation for**: Stacks, queues, hash table collision handling

---

## 1. Singly Linked List

### Node Definition

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### Basic Operations

```python
class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        """Add to end"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, val):
        """Add to beginning"""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, val):
        """Delete first occurrence"""
        if not self.head:
            return
        
        if self.head.val == val:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next
    
    def search(self, val):
        """Search for value"""
        current = self.head
        while current:
            if current.val == val:
                return True
            current = current.next
        return False
    
    def to_list(self):
        """Convert to Python list"""
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
```

---

## 2. Common Linked List Problems

### Reversal

```python
def reverse_list(head):
    """Reverse linked list iteratively"""
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev

def reverse_list_recursive(head):
    """Reverse linked list recursively"""
    if not head or not head.next:
        return head
    
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

### Cycle Detection

```python
def has_cycle(head):
    """Floyd's cycle detection (tortoise and hare)"""
    if not head:
        return False
    
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def detect_cycle_start(head):
    """Find start of cycle"""
    slow = fast = head
    
    # Find meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

### Merge and Split

```python
def merge_two_lists(l1, l2):
    """Merge two sorted linked lists"""
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 if l1 else l2
    return dummy.next

def split_in_half(head):
    """Split list in half using fast/slow pointers"""
    if not head:
        return None, None
    
    slow = fast = head
    prev = None
    
    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
    
    if prev:
        prev.next = None
    
    return head, slow
```

### Middle and Kth Elements

```python
def find_middle(head):
    """Find middle element"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

def find_kth_from_end(head, k):
    """Find kth element from end"""
    slow = fast = head
    
    # Move fast k steps ahead
    for _ in range(k):
        if not fast:
            return None
        fast = fast.next
    
    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

### Remove Elements

```python
def remove_elements(head, val):
    """Remove all elements with value val"""
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
        else:
            current = current.next
    
    return dummy.next

def remove_duplicates(head):
    """Remove duplicates from sorted list"""
    current = head
    while current and current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next
    return head
```

---

## 3. Doubly Linked List

### Node Definition

```python
class DoublyListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
```

### Basic Operations

```python
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def append(self, val):
        """Add to end"""
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
    
    def prepend(self, val):
        """Add to beginning"""
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
    
    def delete(self, node):
        """Delete node"""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
```

---

## 4. Advanced Problems

### LRU Cache

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = DoublyListNode(0)  # Dummy head
        self.tail = DoublyListNode(0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove node"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node):
        """Move node to head"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove and return tail node"""
        node = self.tail.prev
        self._remove_node(node)
        return node
    
    def get(self, key):
        node = self.cache.get(key)
        if not node:
            return -1
        self._move_to_head(node)
        return node.val
    
    def put(self, key, value):
        node = self.cache.get(key)
        if node:
            node.val = value
            self._move_to_head(node)
        else:
            new_node = DoublyListNode(value)
            self.cache[key] = new_node
            self._add_node(new_node)
            
            if len(self.cache) > self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.val]
```

### Copy List with Random Pointer

```python
class Node:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copy_random_list(head):
    """Copy list with random pointers"""
    if not head:
        return None
    
    # Step 1: Create interleaved copy
    current = head
    while current:
        new_node = Node(current.val)
        new_node.next = current.next
        current.next = new_node
        current = new_node.next
    
    # Step 2: Set random pointers
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate lists
    old_head = head
    new_head = head.next
    old_current = old_head
    new_current = new_head
    
    while old_current:
        old_current.next = old_current.next.next
        if new_current.next:
            new_current.next = new_current.next.next
        old_current = old_current.next
        new_current = new_current.next
    
    return new_head
```

---

## 💻 Python Code Examples

```python
# === Add Two Numbers ===

def add_two_numbers(l1, l2):
    """Add two numbers represented by linked lists"""
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        
        current.next = ListNode(total % 10)
        current = current.next
        
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    
    return dummy.next

# === Reorder List ===

def reorder_list(head):
    """Reorder list: L0→Ln→L1→Ln-1→L2→..."""
    if not head or not head.next:
        return
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    prev = None
    current = slow.next
    slow.next = None
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    # Merge two halves
    first, second = head, prev
    while second:
        temp1, temp2 = first.next, second.next
        first.next = second
        second.next = temp1
        first, second = temp1, temp2

# === Flatten Multilevel Doubly Linked List ===

def flatten(head):
    """Flatten multilevel doubly linked list"""
    if not head:
        return None
    
    dummy = DoublyListNode(0)
    dummy.next = head
    stack = [head]
    
    while stack:
        curr = stack.pop()
        
        if curr.next:
            stack.append(curr.next)
        if curr.child:
            stack.append(curr.child)
            curr.next = curr.child
            curr.child.prev = curr
            curr.child = None
        
        if stack:
            curr.next.prev = curr
    
    return dummy.next
```

---

## 📊 Summary Tables

### Linked List Operations

| Operation | Time | Space |
|-----------|------|-------|
| Access | O(n) | O(1) |
| Search | O(n) | O(1) |
| Insert at head | O(1) | O(1) |
| Insert at tail | O(n) | O(1) |
| Delete | O(n) | O(1) |

### Two-Pointer Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| Fast/Slow | Cycle detection | Floyd's algorithm |
| Fast/Slow | Find middle | Middle of list |
| Two pointers | Reverse | In-place reversal |
| Two pointers | Nth from end | Remove Nth node |

---

## 🎯 ML Applications

| Linked List Concept | ML Application |
|---------------------|----------------|
| LRU Cache | Model caching, batch processing |
| Hash table chaining | Collision resolution |
| Adjacency lists | Graph neural networks |

---

## ❓ Quick Check Questions

1. What is the time complexity of accessing the $n$-th element in a Linked List compared to an Array?
2. How does the "Fast and Slow Pointer" (Tortoise and Hare) algorithm detect a cycle in a linked list?
3. Why might you choose a Doubly Linked List over a Singly Linked List?
4. What is a "Dummy Node" and why is it useful in linked list operations?
5. How is an LRU Cache typically implemented using linked lists?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. Accessing the $n$-th element in a Linked List is **O(n)** because you must traverse from the head. In an Array, it is **O(1)** because of contiguous memory allocation.
2. The **Fast and Slow Pointer** algorithm uses two pointers: one moves one step at a time (slow), and the other moves two steps (fast). If there is a cycle, the fast pointer will eventually "lap" and meet the slow pointer.
3. A **Doubly Linked List** allows traversal in both directions (forward and backward) and allows $O(1)$ deletion of a node if you already have a pointer to it (since you can easily access its predecessor).
4. A **Dummy Node** is a fake node placed before the actual head of the list. It helps simplify edge cases, such as inserting or deleting the very first node, by ensuring there is always a "previous" node to reference.
5. An **LRU (Least Recently Used) Cache** is implemented using a combination of a Hash Map (for $O(1)$ lookups) and a Doubly Linked List (for $O(1)$ removal and insertion). When an item is accessed, it is moved to the head of the list; when the cache is full, the item at the tail (least recently used) is evicted.

</details>
---

**Status:** ✅ Complete
**Next:** Stacks and Queues
