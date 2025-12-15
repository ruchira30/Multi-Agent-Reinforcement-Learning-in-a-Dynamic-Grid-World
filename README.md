# Multi-Agent Reinforcement Learning in a Dynamic Grid World

This project demonstrates multi-agent reinforcement learning (MARL) in a dynamic 10×10 grid world where multiple agents collaboratively navigate toward a shared goal while avoiding collisions and moving obstacles. Agents learn cooperative behavior using Proximal Policy Optimization (PPO) with independent actor networks and centralized training via a shared critic. Sparse and dynamic rewards encourage goal-reaching, collision avoidance, and efficient navigation.

# Key Features
## Grid Environment
- 10×10 grid world with a shared goal.
- Randomly moving obstacles introduce stochasticity and non-stationarity.
- Local observations capture nearby agents, obstacles within a fixed radius, and relative goal position.
## Learning Architecture
- Independent actor networks for decentralized execution.
- Centralized critic evaluating joint observations to stabilize cooperative learning.
- PPO with a clipped surrogate objective for stable policy updates.
## Training and Evaluation
- Trained for 900 episodes with 20 steps per episode.
- Evaluated over 100 episodes using metrics including success rate, collision frequency, average distance to goal, and steps to goal.
- Generates visualizations of agent trajectories and dynamic obstacles.

## Results
Agents learn coordinated navigation strategies, reaching the goal efficiently while sparsely receiving rewards, and minimizing collisions.
The dynamic and stochastic environment encourages adaptive behavior and robust learning.
