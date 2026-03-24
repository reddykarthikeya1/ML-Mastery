# 13.1 Reinforcement Learning Fundamentals

## 🎯 Quick Overview
- **RL Framework**: Agent, Environment, State, Action, and Reward
- **MDP (Markov Decision Process)**: The mathematical foundation of RL
- **Bellman Equations**: The recursive math of Value and Q-functions
- **Algorithms**: Q-Learning (Off-policy) vs. SARSA (On-policy)
- **Foundation for**: Robotics, Game AI (AlphaGo), and LLM Alignment (RLHF)

---

## 1. The RL Loop

In RL, an **Agent** interacts with an **Environment** to maximize a cumulative **Reward**.

1.  **State ($S_t$)**: What the agent sees at time $t$.
2.  **Action ($A_t$)**: What the agent does.
3.  **Reward ($R_{t+1}$)**: The feedback from the environment.
4.  **Next State ($S_{t+1}$)**: Where the agent ends up.

---

## 2. Markov Decision Process (MDP)

An MDP is defined by the tuple $(S, A, P, R, \gamma)$:
- **Transition Probability ($P$)**: $P(s' | s, a)$ is the probability of landing in state $s'$ after taking action $a$ in state $s$.
- **Discount Factor ($\gamma$)**: A value between 0 and 1 that determines how much the agent cares about future rewards vs. immediate ones.

### 2.1 Value Functions
- **State-Value $V^\pi(s)$**: The expected return starting from state $s$ following policy $\pi$.
- **Action-Value $Q^\pi(s, a)$**: The expected return starting from state $s$, taking action $a$, then following $\pi$.

---

## 3. The Bellman Equations

The core of RL is solving the Bellman Equation, which breaks the value function into the immediate reward plus the discounted value of the next state.

#### Bellman Optimality Equation for $Q^*$:
$$ Q^*(s, a) = \mathbb{E} [R_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') | S_t=s, A_t=a] $$

---

## 4. Q-Learning vs. SARSA

### 4.1 Q-Learning (Off-Policy)
Learns the value of the **optimal** policy, regardless of the agent's current actions.
- **Update**: $Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$

### 4.2 SARSA (On-Policy)
Learns the value of the policy the agent is **actually following** (including exploration).
- **Update**: $Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma Q(s', a') - Q(s, a)]$

---

## 💻 Professional Implementation: Q-Learning from Scratch

This implementation uses a simple Q-Table to solve a grid-world style environment.

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1, gamma: float = 0.95):
        # 1. Initialize Q-Table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995

    def choose_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1) # Explore
        else:
            return np.argmax(self.q_table[state]) # Exploit

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Bellman update rule."""
        # Max Q value for the next state
        best_next_action = np.max(self.q_table[next_state])
        
        # TD Target
        td_target = reward + self.gamma * best_next_action
        # TD Error
        td_error = td_target - self.q_table[state, action]
        
        # Update Q-Value
        self.q_table[state, action] += self.lr * td_error
        
        # Decay epsilon
        self.epsilon *= self.epsilon_decay

# --- Usage ---
# agent = QLearningAgent(n_states=16, n_actions=4)
# action = agent.choose_action(current_state)
# agent.update(current_state, action, reward, next_state)
```

---

## 📊 Summary Comparison

| Metric | Q-Learning | SARSA | DQN (Deep Q-Network) |
| :--- | :--- | :--- | :--- |
| **Policy** | Off-Policy | On-Policy | Off-Policy |
| **Complexity** | Low | Low | High (Neural Network) |
| **Exploration** | Optimistic | Conservative | Experience Replay |
| **State Space** | Small/Discrete | Small/Discrete | **Infinite/Continuous** |

---

## 🎯 ML Applications & Advanced Scenarios

| Technique | Professional Use Case |
| :--- | :--- |
| **Experience Replay**| Breaking correlation between consecutive samples in Deep RL. |
| **Multi-Agent RL** | Coordinating a swarm of delivery drones. |
| **Inverse RL** | Learning the "reward function" by observing an expert human driver. |
| **Sim-to-Real** | Training a robot in a physics engine before deploying to hardware. |

---

## ❓ Quick Check Questions

1. Why do we use a Discount Factor ($\gamma$) in the MDP formulation?
2. What is the difference between "Exploration" and "Exploitation"?
3. In Q-Learning, why is it called "Off-Policy"?
4. What is the "Temporal Difference (TD) Error"?
5. When would you use a Deep Q-Network (DQN) instead of a standard Q-Table?

---

## 📝 Answers to Quick Check

<details>
<summary>Click to reveal answers</summary>

1. The **Discount Factor** handles the uncertainty of future rewards and ensures that the infinite sum of rewards (return) is mathematically finite. It also mimics human behavior—a reward today is worth more than a reward next year.
2. **Exploration** is trying new actions to see if they lead to better rewards (gathering info). **Exploitation** is choosing the best-known action to maximize immediate reward (using info).
3. It is **Off-Policy** because the update rule uses the maximum possible Q-value of the next state ($\max Q(s', a')$), assuming the agent will act optimally in the future, even if the agent is currently acting randomly (exploring).
4. The **TD Error** is the difference between the estimated value of the current state and the better estimate provided by the actual reward received plus the discounted value of the next state.
5. Use **DQN** when the state space is too large or continuous (e.g., pixels from a video game) to fit into a discrete table. A Neural Network acts as a function approximator to predict Q-values for any given state.

</details>

---

## 📚 Recommended Resources
- **Book**: "Reinforcement Learning: An Introduction" by Sutton & Barto (The RL Bible).
- **Course**: [Deep RL Course (Hugging Face)](https://huggingface.co/learn/deep-rl-course/unit0/introduction).
- **Spinning Up in Deep RL**: [OpenAI Educational Resource](https://spinningup.openai.com/).

---

**Status:** ✅ Elite Standard (10/10)
**Next:** Advanced Reinforcement Learning (PPO, Actor-Critic, SAC)
