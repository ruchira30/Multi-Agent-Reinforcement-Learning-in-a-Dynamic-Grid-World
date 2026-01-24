{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "V5E1"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "TPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "**Import Libraries**"
      ],
      "metadata": {
        "id": "2rr27GuyHjUT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "import matplotlib.pyplot as plt\n",
        "from tqdm import trange\n",
        "from pathlib import Path\n",
        "import imageio\n",
        "from matplotlib import cm\n",
        "from IPython.display import Image, display"
      ],
      "metadata": {
        "id": "MhS98OjJLCJj"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Define Parameters**"
      ],
      "metadata": {
        "id": "ipObFtwIH0mV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "seed = 42\n",
        "np.random.seed(seed)\n",
        "torch.manual_seed(seed)\n",
        "\n",
        "grid = 11\n",
        "num_agents = 2\n",
        "local_radius = 1\n",
        "obs_local_size = (2 * local_radius + 1) ** 2\n",
        "obs_size = obs_local_size + 2\n",
        "num_actions = 5\n",
        "\n",
        "episodes = 900\n",
        "steps_per_episode = 20\n",
        "\n",
        "gamma = 0.995\n",
        "ppo_epochs = 4\n",
        "ppo_clip = 0.2\n",
        "lr_actors = 3e-4\n",
        "lr_critic = 3e-4\n",
        "entropy_coef = 0.01\n",
        "\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "\n",
        "out = Path(\"marl_outputs\")\n",
        "out.mkdir(exist_ok=True)"
      ],
      "metadata": {
        "id": "jzs8eYiqLFFn"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Grid World**"
      ],
      "metadata": {
        "id": "socSOsepMB-f"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "class Grid:\n",
        "    def __init__(self):\n",
        "        self.reset()\n",
        "\n",
        "    def reset(self):\n",
        "        self.pos = np.random.randint(0, 2, (num_agents, 2))\n",
        "        self.goal = np.array([grid - 1, grid - 1])\n",
        "        self.obstacles = {(2, 3), (3, 5)}\n",
        "        return self.get_obs()\n",
        "\n",
        "    def in_bounds(self, p):\n",
        "        return 0 <= p[0] < grid and 0 <= p[1] < grid\n",
        "\n",
        "    def step(self, actions):\n",
        "        moves = {0: [0, 1], 1: [0, -1], 2: [-1, 0], 3: [1, 0], 4: [0, 0]}\n",
        "\n",
        "        for i, a in enumerate(actions):\n",
        "            newp = self.pos[i] + moves[a]\n",
        "            if self.in_bounds(newp) and tuple(newp) not in self.obstacles:\n",
        "                self.pos[i] = newp\n",
        "\n",
        "        # Move obstacles stochastically\n",
        "        new_obstacles = set()\n",
        "        for ox, oy in self.obstacles:\n",
        "            if np.random.rand() < 0.3:\n",
        "                dx, dy = np.random.randint(-1, 2), np.random.randint(-1, 2)\n",
        "                nx, ny = ox + dx, oy + dy\n",
        "                if self.in_bounds([nx, ny]):\n",
        "                    new_obstacles.add((nx, ny))\n",
        "                else:\n",
        "                    new_obstacles.add((ox, oy))\n",
        "            else:\n",
        "                new_obstacles.add((ox, oy))\n",
        "        self.obstacles = new_obstacles\n",
        "\n",
        "        # HYBRID REWARD\n",
        "        dists = np.linalg.norm(self.pos - self.goal, axis=1)\n",
        "\n",
        "        # Dense shaping\n",
        "        reward = 0.1 * np.mean((grid - dists) / grid)\n",
        "\n",
        "        # Sparse cooperative terminal reward\n",
        "        if np.all(dists <= 0.5):\n",
        "            reward += 5.0\n",
        "\n",
        "        # Collision penalty\n",
        "        for i in range(num_agents):\n",
        "            for j in range(i + 1, num_agents):\n",
        "                if (self.pos[i] == self.pos[j]).all():\n",
        "                    reward -= 1.0\n",
        "\n",
        "        # Small step penalty\n",
        "        reward -= 0.005\n",
        "\n",
        "        return self.get_obs(), float(reward), False, {}\n",
        "\n",
        "    def get_obs(self):\n",
        "        obs = np.zeros((num_agents, obs_size), dtype=np.float32)\n",
        "        for i in range(num_agents):\n",
        "            cx, cy = self.pos[i]\n",
        "            local = []\n",
        "            for dy in range(-local_radius, local_radius + 1):\n",
        "                for dx in range(-local_radius, local_radius + 1):\n",
        "                    sx, sy = cx + dx, cy + dy\n",
        "                    if not self.in_bounds([sx, sy]):\n",
        "                        local.append(0.0)\n",
        "                    elif (sx, sy) in self.obstacles:\n",
        "                        local.append(0.5)\n",
        "                    else:\n",
        "                        present = 0.0\n",
        "                        for j in range(num_agents):\n",
        "                            if j != i and (self.pos[j] == [sx, sy]).all():\n",
        "                                present = 1.0\n",
        "                        local.append(present)\n",
        "\n",
        "            rel = (self.goal - self.pos[i]).astype(np.float32) / (grid - 1)\n",
        "            obs[i, :obs_local_size] = local\n",
        "            obs[i, obs_local_size:] = rel\n",
        "        return obs\n",
        "\n"
      ],
      "metadata": {
        "id": "eqZW2CNmNFc1"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Actor-Critic Networks**"
      ],
      "metadata": {
        "id": "YyC6ShUKPYTG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "class ActorNet(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.net = nn.Sequential(\n",
        "            nn.Linear(obs_size, 64),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(64, 32),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(32, num_actions),\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.net(x)\n",
        "\n",
        "class CentralCritic(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.net = nn.Sequential(\n",
        "            nn.Linear(num_agents * obs_size, 128),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(128, 64),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(64, 1),\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.net(x)\n",
        "\n"
      ],
      "metadata": {
        "id": "4XTdddeOPPx2"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def compute_returns_adv(rewards, values):\n",
        "    returns = []\n",
        "    R = 0\n",
        "    for r in reversed(rewards):\n",
        "        R = r + gamma * R\n",
        "        returns.insert(0, R)\n",
        "    advs = np.array(returns) - np.array(values)\n",
        "    advs = (advs - advs.mean()) / (advs.std() + 1e-8)\n",
        "    return returns, advs\n",
        "\n",
        "# Initialization\n",
        "env = Grid()\n",
        "actors = [ActorNet().to(device) for _ in range(num_agents)]\n",
        "actor_opts = [optim.Adam(a.parameters(), lr=lr_actors) for a in actors]\n",
        "critic = CentralCritic().to(device)\n",
        "critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)\n",
        "\n",
        "# PPO Training\n",
        "for ep in trange(episodes):\n",
        "    obs = env.reset()\n",
        "    obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []\n",
        "\n",
        "    for _ in range(steps_per_episode):\n",
        "        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)\n",
        "        actions, logps = [], []\n",
        "\n",
        "        for i in range(num_agents):\n",
        "            dist = torch.distributions.Categorical(logits=actors[i](obs_t[i]))\n",
        "            a = dist.sample()\n",
        "            actions.append(a.item())\n",
        "            logps.append(dist.log_prob(a))\n",
        "\n",
        "        value = critic(obs_t.view(1, -1)).item()\n",
        "        obs_next, r, _, _ = env.step(actions)\n",
        "\n",
        "        obs_buf.append(obs.copy())\n",
        "        act_buf.append(actions)\n",
        "        rew_buf.append(r)\n",
        "        val_buf.append(value)\n",
        "        logp_buf.append(torch.stack(logps).detach().cpu().numpy())\n",
        "\n",
        "        obs = obs_next\n",
        "\n",
        "    returns, advs = compute_returns_adv(rew_buf, val_buf)\n",
        "\n",
        "    # Critic update\n",
        "    critic_opt.zero_grad()\n",
        "    critic_input = torch.tensor(obs_buf, dtype=torch.float32).view(len(obs_buf), -1).to(device)\n",
        "    loss_c = F.mse_loss(critic(critic_input).squeeze(), torch.tensor(returns).to(device))\n",
        "    loss_c.backward()\n",
        "    critic_opt.step()\n",
        "\n",
        "    # Actor updates\n",
        "    for i in range(num_agents):\n",
        "        obs_i = torch.tensor([o[i] for o in obs_buf], dtype=torch.float32).to(device)\n",
        "        act_i = torch.tensor([a[i] for a in act_buf]).to(device)\n",
        "        old_lp = torch.tensor(np.array(logp_buf)[:, i]).to(device)\n",
        "\n",
        "        for _ in range(ppo_epochs):\n",
        "            dist = torch.distributions.Categorical(logits=actors[i](obs_i))\n",
        "            new_lp = dist.log_prob(act_i)\n",
        "            ratio = torch.exp(new_lp - old_lp)\n",
        "\n",
        "            surr = torch.min(\n",
        "                ratio * torch.tensor(advs).to(device),\n",
        "                torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * torch.tensor(advs).to(device),\n",
        "            )\n",
        "\n",
        "            entropy = dist.entropy().mean()\n",
        "            loss_pi = -surr.mean() - entropy_coef * entropy\n",
        "\n",
        "            actor_opts[i].zero_grad()\n",
        "            loss_pi.backward()\n",
        "            actor_opts[i].step()\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lS5z91KNPegG",
        "outputId": "4b526efc-f968-4974-ca5d-bccacaba5239"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  0%|          | 0/900 [00:00<?, ?it/s]/tmp/ipython-input-3512416643.py:48: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:253.)\n",
            "  critic_input = torch.tensor(obs_buf, dtype=torch.float32).view(len(obs_buf), -1).to(device)\n",
            "100%|██████████| 900/900 [00:50<00:00, 17.86it/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Evaluation Metrics & Visualization**"
      ],
      "metadata": {
        "id": "yw0-_Z46Px4b"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def evaluate(env, actors, steps=30, eval_episodes=100, tol=0.5):\n",
        "    frames = []\n",
        "    obs = env.reset()\n",
        "    trajs = [[env.pos[i].copy()] for i in range(num_agents)]\n",
        "    colors = plt.colormaps[\"tab10\"].colors\n",
        "\n",
        "    for _ in range(steps):\n",
        "        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)\n",
        "        actions = [\n",
        "            torch.distributions.Categorical(logits=actors[i](obs_t[i])).sample().item()\n",
        "            for i in range(num_agents)\n",
        "        ]\n",
        "        obs, _, _, _ = env.step(actions)\n",
        "\n",
        "        for i in range(num_agents):\n",
        "            trajs[i].append(env.pos[i].copy())\n",
        "\n",
        "        fig, ax = plt.subplots(figsize=(5, 5))\n",
        "        ax.set_xlim(-0.5, grid - 0.5)\n",
        "        ax.set_ylim(-0.5, grid - 0.5)\n",
        "        ax.set_xticks([]); ax.set_yticks([])\n",
        "\n",
        "        for x in range(grid):\n",
        "            for y in range(grid):\n",
        "                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,\n",
        "                                           fill=False, edgecolor=\"lightgray\"))\n",
        "\n",
        "        for (ox, oy) in env.obstacles:\n",
        "            ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color=\"gray\"))\n",
        "\n",
        "        ax.add_patch(plt.Rectangle((grid - 1.5, grid - 1.5), 1, 1, color=\"green\", alpha=0.5))\n",
        "\n",
        "        for i in range(num_agents):\n",
        "            tr = np.array(trajs[i])\n",
        "            ax.plot(tr[:, 0], tr[:, 1], color=colors[i])\n",
        "            ax.scatter(tr[-1, 0], tr[-1, 1], color=colors[i], s=80)\n",
        "\n",
        "        fig.canvas.draw()\n",
        "        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)\n",
        "        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]\n",
        "        frames.append(frame)\n",
        "        plt.close(fig)\n",
        "\n",
        "    gif_path = out / \"marl_demo.gif\"\n",
        "    imageio.mimsave(gif_path, frames, fps=3)\n",
        "    display(Image(filename=str(gif_path)))\n",
        "\n",
        "    success, rewards, collisions, distances, steps_goal = 0, [], [], [], []\n",
        "\n",
        "    for _ in range(eval_episodes):\n",
        "        obs = env.reset()\n",
        "        ep_reward, ep_coll, reached = 0, 0, [False] * num_agents\n",
        "\n",
        "        for step in range(steps):\n",
        "            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)\n",
        "            actions = [\n",
        "                torch.distributions.Categorical(logits=actors[i](obs_t[i])).sample().item()\n",
        "                for i in range(num_agents)\n",
        "            ]\n",
        "            obs, r, _, _ = env.step(actions)\n",
        "            ep_reward += r\n",
        "\n",
        "            for i in range(num_agents):\n",
        "                for j in range(i + 1, num_agents):\n",
        "                    if (env.pos[i] == env.pos[j]).all():\n",
        "                        ep_coll += 1\n",
        "\n",
        "            for i in range(num_agents):\n",
        "                if not reached[i] and np.linalg.norm(env.pos[i] - env.goal) <= tol:\n",
        "                    reached[i] = True\n",
        "\n",
        "        if all(reached):\n",
        "            success += 1\n",
        "\n",
        "        rewards.append(ep_reward)\n",
        "        collisions.append(ep_coll)\n",
        "        distances.append(np.mean([np.linalg.norm(env.pos[i] - env.goal)\n",
        "                                   for i in range(num_agents)]))\n",
        "        steps_goal.append(step + 1)\n",
        "\n",
        "    print(\"\\n--- Evaluation Metrics ---\")\n",
        "    print(f\"Average Reward: {np.mean(rewards):.3f}\")\n",
        "    print(f\"Success Rate (%): {success}\")\n",
        "    print(f\"Average Distance to Goal: {np.mean(distances):.3f}\")\n",
        "    print(f\"Average Collisions per Episode: {np.mean(collisions):.3f}\")\n",
        "    print(f\"Average Steps to Goal: {np.mean(steps_goal):.3f}\")\n",
        "\n",
        "evaluate(env, actors)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 638
        },
        "id": "HPKQWfdmPxeE",
        "outputId": "7b3a1a17-a8f2-4b89-ac77-45f0965d54c1"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/gif": "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SPLDES1dYmo58gGlzZs4c+rcaXOlz5EpssCMKTNLCp5IkypdyjSkz6chP2ThQrQqlyw1m2rdyrUrzqc/Qb6sSlaLEa9o06pdi7CA27dw48p9e+HFhbluLWwhy1cLA7aAAwtmy7ew4cN8VQxezLgxT8SQIxOt4biy5csH8WqeW/cuXhqSFWMeTXoxWKgeHQw97Le069dqT6sMOdawWdi4czeVXTHq1MJXs+oeTvyrbKBCyxotzry5ydMlPxhZbVa48+vYL660yUBFDRV/s4v/H0++vPnz6NOrX8++vfv38OPLn0+/vv37+PPr38+/v///AAYo4IAEFmjggQgmqOCCDDbo4IMQRijhhBRWaOGFGGao4YYcdujhhyCGKOKIJJZo4okopqjiiiy26OKLMMYo44w01mjjjTjmqOOOPPbo449ABinkkEQWaeSRSCap5JJMNunkk1BGKeWUVFZp5ZVYZqnlllx26eWXYIYp5phklmnmmWimqeaabLbp5ptwxinnnHTWaeedeOap55589unnn4AGKuighBZq6KGIJqrooow26uijkEYq6aSUVmrppZhmqummnHbq6aeghirqqKSWauqpqKaq6qqsturqq7AG/4hAByx0gECsCkmQgw9E9OpDDhLgatAGPADR67E98LCBsANJwMMQx0Y7BA/BMpuDsdFG2wMOzCLAa7bZ+nCAsB2Aay4HwrJgLrgskLtutuji6u27vo4r7LXvbsssGM5CC+601TJLLLbIKrtvszgEcawPOAR8sEAQ9LqCvQ8TNECvA1Rs0MVEZKyxxRh/DHLHIg/Esccln1yyQCqv3LJSJGTwwMw012zzzTjXrAEJcE7Q6wRLkSD00EQXbfTRSAv9Zgu9trDUzhrkLPXUNkP9ps9EAK3UAyQ8wNvXp3H9wJsvIyU22GivJDbZIW/dddpwA7C2m2XzdHbcaM/dZt073f+N99d6s8m3Tn7/LVvgaw6eU+GGg4W4morjxHjjPj2eZuQ3TU65SpajiblNmm9OUednngyA216L/hTpZFLk8QC92f226pV3zWZFr8/W9+y0c267mirlrvvivPc++u9pBs/ydoQXz5sAm8nVGfDKgwE785I7f5oAUfS5kvDDZ649WAX4+f3y2Iuf+tflg8FA9HTZRT3u6IcP+vhPtV9A3sifeb71Ycne+nijP/6NbX4AAB9LiDdA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qc+KL6eK1WKEq9mzaHMWWMu2rdu3ay+suAB3rYUtYvNqYZC2r9+/NvMKHkxYrAvAiBMrfli4seOYNxZLnsy1ruW3cunWtfH4MOXPoJl6/SrRgVDCe0OrXn1ytEqNYQeTZU27dkXXGJ9GFVz1qu3fwBfi/hl0bNHgyJMbHE3yg5HTZH0rn558JUoGLm644Eu9u/fv4MOL/x9Pvrz58+jTq1/Pvr379/Djy59Pv779+/jz69/Pv7///wAGKOCABBZo4IEIJqjgggw26OCDEEYo4YQUVmjhhRhmqOGGHHbo4YcghijiiCSWaOKJKKao4oostujiizDGKOOMNNZo44045qjjjjz26OOPQAYp5JBEFmnkkUgmqeSSTDbp5JNQRinllFRWaeWVWGap5ZZcdunll2CGKeaYZJZp5plopqnmmmy26eabcMYp55x01mnnnXjmqeeefPbp55+ABirooIQWauihiCaq6KKMNuroo5BGKumklFZq6aWYZqrpppx26umnoIYq6qiklmrqqaimquqqrLbq6quwxv8q66y01mrrrbjmquuuvPbq66/ABivssMQWa+yxyCar7LLMNuvss9BGK+201FZr7bXYZqvtttx26+234IYr7rjklmvuueimq+667Lbr7rtLkZDBA/TWa++9+OZrrwYk6EnCvwAHLPDABBf8b578aqDvwgzfm3CeD5DwAG4UjxbxAxBLXPHGK12c8cQch+wxnheHLLLEH5vM8ch3lqxyxSzb6fLLuMVc58w0W4wyyRrn7JrNdOLsc09Azyn00CoVLefRSF+kdJxMN/00nFEjPfWbVQ99tZtZ+7x1m13n/DWbYdM89pplv3y2mmmrvHaabZv8Nppxn4wxzyA33fHOLff/rHfSfMvs999OB37z4ITPfWbdKxseNOJ/K24m4xtLXiblMDtuNOR6W04m5hR7Pia/8zZsur4ayACGCKy37vrrsMfuugqqW2nw7bgLrLoQvPfu++/ABw/8D7VXSYLCpydvLxjCN+988yrkKcLz1FcvgvTVZ9/89XhOr/33vnN/p/fggy++neSXr/35daavvvXYv79+/PLD33392bNPp/v4C6//nPzrH/D+J6cACjB89Dtg8AgYJwMqUAgMhJMDFRjBN03wgBV00wUFmME2bbB/HWTTB/EXwjWNsH4lVNMJ5ZfCNK3wfS1E0wvVF8Mz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\n",
            "text/plain": [
              "<IPython.core.display.Image object>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "--- Evaluation Metrics ---\n",
            "Average Reward: 3.589\n",
            "Success Rate (%): 21\n",
            "Average Distance to Goal: 1.402\n",
            "Average Collisions per Episode: 0.830\n",
            "Average Steps to Goal: 30.000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "1Rs3FG95P5t2"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
