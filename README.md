# Multi-Agent Reinforcement Learning with PPO

A **Multi-Agent Reinforcement Learning (MARL)** project that trains multiple agents to cooperatively navigate a dynamic grid environment using **Proximal Policy Optimization (PPO)** with decentralized actors and a centralized critic.

## Overview

This project implements a cooperative multi-agent navigation environment in which two agents learn to reach a common goal while navigating obstacles.

The environment includes:

* An `11 × 11` grid world
* 2 cooperative agents
* Stochastically moving obstacles
* Local observations for each agent
* A shared goal location
* Collision penalties
* Dense distance-based reward shaping
* Cooperative terminal reward

The agents are trained using **PPO** with separate actor networks and a centralized critic.

## Environment

Each agent receives a local observation consisting of:

* A local `3 × 3` grid around the agent
* Relative position of the goal

Agents can perform five actions:

* Move right
* Move left
* Move up
* Move down
* Stay

Obstacles can move stochastically during the episode, creating a dynamic environment.

## Reward Function

The environment uses a hybrid reward consisting of:

* **Dense reward** based on distance to the goal
* **+5.0 cooperative reward** when both agents reach the goal
* **−1.0 collision penalty**
* **−0.005 step penalty**

This encourages the agents to reach the shared goal efficiently while avoiding collisions.

## MARL Architecture

The project follows a **centralized-critic, decentralized-actor** setup.

### Actor Networks

Each agent has its own policy network:

```text
Local Observation
       ↓
   Linear (64)
       ↓
      ReLU
       ↓
   Linear (32)
       ↓
      ReLU
       ↓
  Action Logits
```

Each actor selects from the five available actions.

### Centralized Critic

The critic receives the observations of both agents:

```text
Combined Agent Observations
          ↓
      Linear (128)
          ↓
         ReLU
          ↓
       Linear (64)
          ↓
         ReLU
          ↓
        Value
```

The centralized critic estimates the value of the joint multi-agent state.

## Training

The agents are trained using **Proximal Policy Optimization (PPO)**.

Key training configuration:

| Parameter            |    Value |
| -------------------- | -------: |
| Grid Size            |  11 × 11 |
| Number of Agents     |        2 |
| Episodes             |      900 |
| Steps / Episode      |       20 |
| Discount Factor (γ)  |    0.995 |
| PPO Epochs           |        4 |
| PPO Clip             |      0.2 |
| Actor Learning Rate  | 3 × 10⁻⁴ |
| Critic Learning Rate | 3 × 10⁻⁴ |
| Entropy Coefficient  |     0.01 |

Advantage estimates are normalized before the actor updates, and the critic is trained using mean squared error against the computed returns.

## Evaluation

The trained policies are evaluated over **100 episodes**, with up to 30 steps per episode.

The evaluation tracks:

* Average reward
* Success rate
* Average distance to goal
* Average collisions per episode
* Average steps per episode

### Evaluation Results

| Metric                       | Result |
| ---------------------------- | -----: |
| Average Reward               |  3.589 |
| Success Rate                 |    21% |
| Average Distance to Goal     |  1.402 |
| Average Collisions / Episode |  0.830 |
| Average Steps / Episode      | 30.000 |

A trajectory visualization is also generated as a GIF to illustrate agent movement, obstacle positions, and the shared goal.

## Visualization

The evaluation generates a grid-world animation showing:

* Agent trajectories
* Agent positions
* Moving obstacles
* Goal location

The resulting demonstration is saved as:

`marl_outputs/marl_demo.gif`

## Tech Stack

**Language:** Python

**Framework:** PyTorch

**Libraries:** NumPy, Matplotlib, ImageIO, tqdm

**Methods:** Multi-Agent Reinforcement Learning, PPO, Actor-Critic, Policy Gradient, Advantage Estimation

## Future Work

* Improve cooperative success rate through additional training and hyperparameter tuning.
* Experiment with centralized training and decentralized execution more explicitly.
* Explore alternative reward designs for stronger collision avoidance.
* Compare PPO with other multi-agent reinforcement learning algorithms.
* Introduce more agents and larger environments.
* Evaluate performance under different obstacle dynamics.
* Add systematic training curves and comparative experiments.

## Author

**Ruchira Purohit**
