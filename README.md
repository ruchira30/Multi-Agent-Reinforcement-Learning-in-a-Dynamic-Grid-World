# Multi-Agent-Reinforcement-Learning-in-a-Dynamic-Grid-World
This project demonstrates multi-agent reinforcement learning (MARL) in a dynamic 10x10 grid world where multiple agents collaboratively navigate to a shared goal while avoiding collisions and moving obstacles. Agents learn cooperative behavior using Proximal Policy Optimization (PPO) with independent actor networks and a shared central critic. Reward shaping encourages goal-reaching, collision avoidance, and efficient navigation.

# Key Features
# Grid Environment
- 10x10 grid world with a shared goal.
- Randomly moving obstacles introduce stochasticity and learning challenges.
- Local observations capture nearby agents, obstacles within a radius, and relative goal position.

# Learning Architecture
- Each agent has its own actor network (policy).
- Centralized critic evaluates joint states to stabilize learning.
- PPO with a clipped surrogate objective ensures stable policy updates.

# Training and Evaluation
- Trained for 900 episodes with 20 steps per episode.
- Reward structure includes goal bonuses, collision penalties, and step penalties to promote efficient cooperative behavior.
- Evaluated over 100 episodes measuring success rate, collisions, average distance to goal, and steps to goal.
- Generates an image visualizing agent trajectories and dynamic obstacles.

# Results
- Agents learn to reach the goal efficiently while minimizing collisions.

# Metrics include:
- Average Reward
- Success Rate
- Average Distance to Goal
- Average Collisions per Episode
- Average Steps to Goal

<img width="450" height="181" alt="image" src="https://github.com/user-attachments/assets/405ebbdd-f07b-4f28-acd9-775add91f989" />


# Visualization:

