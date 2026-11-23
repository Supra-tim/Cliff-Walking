# Cliff Walking
## SARSA Algorithm – Cliff Walking Environment

### Overview
This project implements the **SARSA (State-Action-Reward-State-Action)** algorithm, an On-Policy Temporal Difference Reinforcement Learning technique, using the `CliffWalking-v1` environment from Gymnasium.

The objective is to train an agent to navigate from the starting position to the goal while avoiding the cliff and maximizing its cumulative reward.

### Technologies Used
- Python
- NumPy
- Gymnasium
- Random

### Implementation Process

1. **Environment Setup:** Initialize the CliffWalking-v1 environment, which consists of 48 states and 4 possible actions (Up, Right, Down, Left).

2. **Q-Table Initialization:** Create a Q-table of dimensions `(48, 4)` with all values initialized to zero.

3. **Hyperparameter Configuration:** Define the learning rate (`alpha = 0.5`), discount factor (`gamma = 0.99`), exploration rate (`epsilon = 0.1`), and training episodes (`500`).

4. **Epsilon-Greedy Policy:** Implement an action-selection strategy that balances exploration (random actions) and exploitation (choosing the action with the highest Q-value).

5. **SARSA Training:** Train the agent through repeated interactions with the environment and update Q-values using the SARSA update equation:

   \[
   Q(S,A) \leftarrow Q(S,A)+\alpha[R+\gamma Q(S',A')-Q(S,A)]
   \]

6. **Performance Monitoring:** Track the total reward and episode length to observe the agent's learning progress.

7. **Agent Evaluation:** Test the trained agent using the learned Q-table and `np.argmax()` to select the best-known action at each state.

### Expected Results
- The agent gradually learns to navigate the environment.
- Learns to avoid the cliff region.
- Improves its decision-making through repeated interactions.
- Attempts to reach the goal using fewer steps and higher cumulative rewards.

### Key Concepts
- On-Policy Reinforcement Learning
- Temporal Difference Learning
- Q-Table
- Epsilon-Greedy Policy
- Exploration vs. Exploitation
- Cumulative Reward

### Conclusion
This project demonstrates how SARSA enables an agent to learn an effective navigation strategy through trial and error. By continuously updating its Q-values based on the current policy, the agent gradually improves its ability to reach the goal while avoiding dangerous states.
