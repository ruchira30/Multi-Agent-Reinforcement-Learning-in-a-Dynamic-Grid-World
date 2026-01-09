## Multi-Agent Reinforcement Learning in a Dynamic Gridworld

This project implements a cooperative Multi-Agent Reinforcement Learning (MARL) system in a dynamic gridworld using Proximal Policy Optimization (PPO). Multiple agents learn to coordinate their movements toward a shared goal while avoiding collisions and stochastically moving obstacles.The learning setup follows a centralized critic with decentralized actors, enabling stable cooperative learning under partial observability and environmental uncertainty.

# Environment
- 11×11 grid world with a fixed shared goal at the top-right corner.
- Two cooperative agents initialized at random positions.
- Dynamic obstacles that move stochastically at each timestep.

# Partial observability:
- Each agent observes a local 3×3 neighborhood encoding nearby agents and obstacles.
- Includes a normalized relative vector to the global goal.

# Discrete action space (5 actions):

Up, Down, Left, Right, Stay.

# Learning Architecture

# A.Decentralized Actors:
- Each agent is controlled by an independent neural network policy.
- Policies operate only on local observations at execution time.

# B.Centralized Critic:
- A shared value function receives the concatenated observations of all agents.
- Provides a global estimate of team value to stabilize learning.

# Proximal Policy Optimization (PPO):
- Clipped surrogate objective for robust policy updates.
- Multiple PPO epochs per rollout.
- Entropy regularization encourages exploration.

# Reward Design
A hybrid cooperative reward structure is used:
- Dense shaping reward based on average agent distance to the goal.
- Sparse cooperative bonus when all agents reach the goal region.
- Collision penalty when agents occupy the same grid cell.
- Small step penalty to promote efficient navigation.

This reward formulation balances learning speed with coordinated behavior.

# Training
- 900 training episodes
- 20 steps per episode
- Discount factor: γ = 0.995
- Optimizer: Adam (actors and critic)
- Framework: PyTorch
- Hardware support: CPU / CUDA GPU

# Evaluation & Visualization
  Quantitative Metrics
  During evaluation, the following metrics are reported:
  - Average Episode Reward
  - Success Rate (all agents reach the goal)
  - Average Distance to Goal
  - Average Collisions per Episode
  - Average Steps per Episode

# Qualitative Visualization
Generates animated GIFs showing:
- Agent trajectories
- Dynamic obstacle movements
- Goal location

Useful for inspecting coordination and emergent behaviors.

## Key Highlights
- Centralized training with decentralized execution (CTDE)
- Cooperative PPO in a stochastic, partially observable environment
- Dynamic obstacles introduce non-stationarity
- Clear visualization of learned multi-agent coordination
