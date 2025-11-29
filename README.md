# Multi-Agent-Reinforcement-Learning-in-a-Dynamic-Grid-World
This project implements a multi-agent reinforcement learning (MARL) environment in a simple grid world where multiple agents must navigate to a shared goal while avoiding collisions and dynamic obstacles. The agents learn cooperative behavior using Proximal Policy Optimization (PPO) with independent actor networks and a shared central critic.

# Key Features:
# Grid Environment:
- Agents operate in a 10x10 grid with a fixed goal position.
- Randomly moving obstacles introduce stochasticity and increase the learning challenge.
- Local observations include surrounding agents and obstacles within a radius, plus the relative position to the goal.

# Learning Architecture:
- Each agent has its own actor network (policy).
- A central critic evaluates joint states to stabilize learning.
- PPO is used to update policies with clipped surrogate objective for stable training.

# Training and Evaluation:
- Trained for 900 episodes with 15 steps per episode.
- Agents learn to optimize average reward, minimize collisions, and efficiently reach the goal.
- Evaluated over 100 episodes measuring success rate, collisions, steps to goal, and distance to goal.
- Generates a GIF visualizing agent trajectories and dynamic obstacles during evaluation.

# Results:
- Trained agents learn to efficiently navigate to the goal while avoiding collisions.
    # Metrics reported include:
      - Average Reward
      - Success Rate
      - Average Distance to Goal
      - Average Collisions per Episode
      - Average Steps to Goal
  
<img width="450" height="181" alt="image" src="https://github.com/user-attachments/assets/405ebbdd-f07b-4f28-acd9-775add91f989" />


# Visualization:
![MARL Demo](marl_simple_outputs/marl_demo.gif)

