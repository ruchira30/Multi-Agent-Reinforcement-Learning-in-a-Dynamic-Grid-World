# Multi-Agent Reinforcement Learning in a Dynamic Gridworld

This project demonstrates multi-agent reinforcement learning (MARL) in a dynamic 10×10 grid world, where multiple agents collaboratively navigate toward a shared goal while avoiding collisions and moving obstacles. Agents learn cooperative strategies using Proximal Policy Optimization (PPO) with independent actor networks and a centralized critic, enabling stable learning in partially observable, stochastic environments.

# Features
## Environment
- 10×10 grid world with a shared goal.
- Randomly moving obstacles introduce stochasticity and dynamic challenges.
- Local observations capture nearby agents, obstacles, and relative goal positions.
## Learning Architecture
- Independent actor networks for decentralized execution.
- Centralized critic evaluating joint states to stabilize cooperative learning.
- PPO with clipped surrogate objective for stable policy updates.

## Rewards
- Sparse & dynamic rewards: agents receive rewards for reaching the goal and penalties for collisions.
- Optional step penalty encourages efficient navigation.

## Training
- Trained for 900 episodes with 20 steps per episode.
- Supports GPU acceleration via PyTorch.

## Evaluation & Visualization

Metrics:
- Average Reward
- Success Rate
- Average Distance to Goal
- Average Collisions per Episode

Average Steps to Goal

Generates trajectory GIFs showing agent paths and dynamic obstacles.
