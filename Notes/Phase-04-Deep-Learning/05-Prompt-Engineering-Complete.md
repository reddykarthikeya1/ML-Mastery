# 10.5 Advanced Prompt Engineering: From Chat to Programming

## 🎯 Quick Overview
- **Reasoning Frameworks**: Chain-of-Thought (CoT) and Self-Consistency
- **Advanced Planning**: Tree-of-Thoughts (ToT) and Graph-of-Thoughts
- **Reliability Techniques**: Delimiters, Few-shot selection, and Self-Correction
- **Programmatic Prompting**: Moving beyond "strings" to **DSPy** and Structured Outputs
- **Foundation for**: Building autonomous agents and production-grade LLM pipelines

---

## 1. The Reasoning Hierarchy

Prompting has evolved from "magic spells" to structured reasoning frameworks.

### 1.1 Chain-of-Thought (CoT) & Zero-shot CoT
Forces the model to generate intermediate steps.
- **Zero-shot CoT**: Adding "Let's think step by step" triggers emergent reasoning in models >7B parameters.

### 1.2 Self-Consistency (Majority Voting)
Instead of taking one answer, sample multiple reasoning paths (at `temperature > 0`) and take the **majority vote** of the final answers.
- **Benefit**: Significantly reduces "flukey" logic errors in math and coding.

---

## 2. Advanced Planning: Tree-of-Thoughts (ToT)

For complex tasks where a linear path fails, ToT allows the model to:
1.  **Generate** multiple potential "thoughts" (steps).
2.  **Evaluate** each step (e.g., "Is this path likely to lead to a solution?").
3.  **Search** using algorithms like BFS (Breadth-First Search) or DFS to find the optimal path.

---

## 3. The Programmatic Shift: DSPy

**DSPy** (Declarative Self-improving Language Programs) is the future of prompt engineering.
- **The Concept**: Instead of manually "tuning" prompts, you define a **Signature** (Input/Output behavior) and a **Module** (Pipeline).
- **The Optimizer**: DSPy automatically "compiles" the best prompts and few-shot examples based on a tiny metric-driven dataset.
- **Analogy**: DSPy is to Prompting what **PyTorch** is to manual Neural Network weight tuning.

---

## 💻 Professional Implementation

### 1. Self-Consistency Implementation (Logic)
```python
import collections

def self_consistency_query(prompt, n=5):
    answers = []
    for _ in range(n):
        # Sample with higher temperature for diversity
        resp = llm.generate(prompt, temperature=0.7)
        # Extract the final answer (e.g., the number after 'The answer is')
        answers.append(extract_answer(resp))
    
    # Majority Vote
    prediction = collections.Counter(answers).most_common(1)[0][0]
    return prediction
```

### 2. DSPy Signature Example
```python
import dspy

class MultiHopQA(dspy.Signature):
    """Answer questions by searching for multiple facts."""
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="A concise 1-2 sentence answer.")

# This module can now be 'compiled' and optimized automatically
generate_answer = dspy.Predict(MultiHopQA)
```

---

## 📊 Summary Comparison

| Technique | Logic | Complexity | Impact on Accuracy |
| :--- | :--- | :--- | :--- |
| **Zero-shot** | Direct | Low | Baseline |
| **Few-shot** | Pattern Match | Low | High (Format) |
| **CoT** | Step-by-step | Medium | **High (Reasoning)** |
| **Self-Consistency**| Voting | Medium | **Very High (Reliability)**|
| **ToT** | Search/Tree | High | Transformative (Planning)|

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Iterative Refinement**| Asking the LLM to find bugs in its own generated code and fix them. |
| **Persona Prompting** | Simulating a "Red Team" to find security vulnerabilities in a system. |
| **Metaprompting** | Using a strong model (GPT-4) to write the prompts for a weaker model (Llama-3-8B). |
| **Dynamic Few-shot** | Using a Vector DB to retrieve the most similar training examples for each user query. |

---

## ❓ Quick Check Questions

1. How does Self-Consistency differ from standard Chain-of-Thought?
2. Why is "Prompt Sensitivity" a problem, and how does DSPy solve it?
3. What is the "Step-level reward" in Tree-of-Thoughts?
4. When should you use a "System Role" vs. putting instructions in the "User Role"?
5. Explain the "Chain of Hindsight" (CoH) concept.

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. **Chain-of-Thought** focuses on a single reasoning path. **Self-Consistency** generates multiple independent CoT paths and uses a majority vote to decide the final answer, making it more robust against random errors in any one path.
2. **Prompt Sensitivity** means that changing one word in a prompt can drastically change the LLM's output. **DSPy** solves this by abstracting the prompt into a "program." It uses an optimizer to mathematically find the most reliable instructions and examples for your specific data.
3. In ToT, the model acts as a "judge" for its own intermediate steps. A **Step-level reward** is a score (e.g., "Sure", "Maybe", "Impossible") that the model gives to a current path, allowing the search algorithm to prune bad paths early.
4. Use the **System Role** for permanent constraints (e.g., "Never use emojis", "Only output JSON"). Use the **User Role** for the specific, transient task data. Models are trained to follow System instructions with higher priority.
5. **Chain of Hindsight** is a technique where the model is shown both its past mistakes and the corrections, learning to identify and avoid bad reasoning patterns by reflecting on "hindsight" data.

</details>

---

## 📚 Recommended Resources
- **Paper**: [Tree of Thoughts: Deliberative Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- **Library**: [DSPy Documentation (Stanford NLP)](https://dspy-docs.vercel.app/)
- **Guide**: [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/) - *Comprehensive community resource*.

---

**Status:** ✅ Expanded Standard (10/10)
**Next:** RAG Architectures (HNSW, Re-ranking, GraphRAG)
