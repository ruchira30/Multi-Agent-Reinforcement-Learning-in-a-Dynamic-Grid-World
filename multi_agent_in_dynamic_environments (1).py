
**Import Libraries**
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path
from IPython.display import Image
import matplotlib.cm as cm

"""**Define Parameters**"""

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

grid = 10
num_agents = 2
local_radius = 1
obs_local_size = (2*local_radius+1)**2
obs_size = obs_local_size + 2
num_actions = 5
episodes = 900
steps_per_episode = 20
gamma = 0.99
ppo_epochs = 10
ppo_clip = 0.2
lr_actors = 3e-4
lr_critic = 3e-4

out = Path("marl_simple_outputs")
out.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""**Grid World**"""

class Grid:
    def __init__(self):
        self.grid = grid
        self.n = num_agents
        self.reset()

    def reset(self):
        self.pos = np.random.randint(0, 2, (self.n,2))
        self.goal = np.array([grid-1, grid-1])
        self.obstacles = {(2,3), (3,5)}
        return self.get_obs()

    def in_bounds(self, p):
        return 0 <= p[0] < grid and 0 <= p[1] < grid

    def step(self, actions):
        moves = {0:[0,1], 1:[0,-1], 2:[-1,0], 3:[1,0], 4:[0,0]}
        for i, a in enumerate(actions):
            newp = self.pos[i] + moves[int(a)]
            if self.in_bounds(newp) and tuple(newp) not in self.obstacles:
                self.pos[i] = newp

        # Move obstacles randomly
        new_obstacles = set()
        for ox, oy in self.obstacles:
            if np.random.rand() < 0.3:
                dx, dy = np.random.randint(-1,2), np.random.randint(-1,2)
                nx, ny = ox+dx, oy+dy
                if self.in_bounds([nx, ny]) and (nx, ny) not in self.obstacles and not any((self.pos==[nx,ny]).all(1)):
                    new_obstacles.add((nx, ny))
                else:
                    new_obstacles.add((ox, oy))
            else:
                new_obstacles.add((ox, oy))
        self.obstacles = new_obstacles

        # Compute reward
        dists = np.linalg.norm(self.pos - self.goal, axis=1)
        reward = np.mean(1 - dists/grid)
        for i in range(self.n):
            for j in range(i+1, self.n):
                if (self.pos[i] == self.pos[j]).all():
                    reward = -1.0

        return self.get_obs(), float(reward), False, {}

    def get_obs(self):
        obs = np.zeros((self.n, obs_size), dtype=np.float32)
        for i in range(self.n):
            cx, cy = self.pos[i]
            local = []
            for dy in range(-local_radius, local_radius+1):
                for dx in range(-local_radius, local_radius+1):
                    sx, sy = cx+dx, cy+dy
                    if not self.in_bounds([sx,sy]):
                        local.append(0.0)
                    elif (sx,sy) in self.obstacles:
                        local.append(0.5)
                    else:
                        present = 0.0
                        for j in range(self.n):
                            if j==i: continue
                            if (self.pos[j] == [sx,sy]).all():
                                present = 1.0
                        local.append(present)
            rel = (self.goal - self.pos[i]).astype(np.float32)/ (grid-1)
            obs[i,:obs_local_size] = local
            obs[i,obs_local_size:obs_local_size+2] = rel
        return obs

"""**PPO with centralized critic**"""

class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,num_actions)
        )

    def forward(self,x):
        return self.fc(x)

class CentralCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_agents*obs_size,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.fc(x)

def compute_gae(rewards, values, gamma=gamma):
    returns = []
    R = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        R = r + gamma*R
        returns.insert(0,R)
    advs = np.array(returns) - np.array(values)
    return returns, advs

"""**Initialization**"""

env = Grid()
actors = [ActorNet().to(device) for _ in range(num_agents)]
actor_opts = [optim.Adam(a.parameters(),lr=lr_actors) for a in actors]
critic = CentralCritic().to(device)
critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)

"""**Training**"""

ep_rewards = []

for ep in trange(episodes):
    obs = env.reset()
    obs_list, actions_list, rewards_list, values_list, logps_list = [], [], [], [], []

    for step in range(steps_per_episode):
        obs_tensor = torch.tensor(obs,dtype=torch.float32).to(device)
        actions, logps = [], []
        for i in range(num_agents):
            logits = actors[i](obs_tensor[i])
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            actions.append(int(a.item()))
            logps.append(dist.log_prob(a))

        flat_state = obs_tensor.view(1,-1)
        value = critic(flat_state).item()

        obs_next, r, _, _ = env.step(actions)

        obs_list.append(obs.copy())
        actions_list.append(actions)
        rewards_list.append(r)
        values_list.append(value)
        logps_list.append(torch.stack(logps).cpu().detach().numpy())
        obs = obs_next

    returns, advs = compute_gae(rewards_list, values_list)
    returns_t = torch.tensor(returns,dtype=torch.float32).to(device)
    advs_t = torch.tensor(advs,dtype=torch.float32).to(device)
    advs_t = (advs_t - advs_t.mean())/(advs_t.std()+1e-8)

    # CRITIC UPDATE
    critic_opt.zero_grad()
    crit_inp = torch.tensor(np.array(obs_list),dtype=torch.float32).view(len(obs_list),-1).to(device)
    loss_critic = F.mse_loss(critic(crit_inp).squeeze(), returns_t)
    loss_critic.backward()
    critic_opt.step()

    # ACTORS UPDATE
    for i in range(num_agents):
        obs_batch = torch.tensor(np.array([o[i] for o in obs_list]),dtype=torch.float32).to(device)
        actions_idx = torch.tensor([a[i] for a in actions_list],dtype=torch.long).to(device)
        old_lp = torch.tensor(np.array(logps_list)[:,i],dtype=torch.float32).to(device)

        for _ in range(ppo_epochs):
            logits = actors[i](obs_batch)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(actions_idx)
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advs_t
            surr2 = torch.clamp(ratio, 1-ppo_clip, 1+ppo_clip)*advs_t
            loss_pi = -torch.min(surr1,surr2).mean()
            actor_opts[i].zero_grad()
            loss_pi.backward()
            actor_opts[i].step()

    ep_rewards.append(np.mean(rewards_list))


plt.figure(figsize=(8,4))
plt.plot(ep_rewards)
plt.xlabel("Episode")
plt.ylabel("Avg reward")
plt.title("Training Reward Curve")
plt.show()

"""**Evaluation & Visualization**"""

def evaluation(env, actors, steps=30, out_gif=out/"marl_demo.gif", tol=0.5):
    import matplotlib
    matplotlib.use("Agg")
    frames = []
    obs = env.reset()
    trajs = [[env.pos[i].copy()] for i in range(num_agents)]

    for t in range(steps):
        obs_tensor = torch.tensor(obs,dtype=torch.float32).to(device)
        actions = [torch.distributions.Categorical(logits=actors[i](obs_tensor[i])).sample().item() for i in range(num_agents)]
        obs, _, _, _ = env.step(actions)
        for i in range(num_agents):
            trajs[i].append(env.pos[i].copy())

        # Draw frame
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlim(-0.5, grid-0.5)
        ax.set_ylim(-0.5, grid-0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for x in range(grid):
            for y in range(grid):
                ax.add_patch(plt.Rectangle((x-0.5,y-0.5),1,1,fill=False,edgecolor="lightgray"))
        for (ox,oy) in env.obstacles:
            ax.add_patch(plt.Rectangle((ox-0.5,oy-0.5),1,1,color="gray"))
        gx, gy = env.goal
        ax.add_patch(plt.Rectangle((gx-0.5,gy-0.5),1,1,color="green",alpha=0.5))
        colors = cm.get_cmap('tab10',num_agents).colors
        for i in range(num_agents):
            traj = np.array(trajs[i])
            ax.plot(traj[:,0],traj[:,1],color=colors[i],alpha=0.6)
            ax.scatter(traj[-1,0],traj[-1,1],color=colors[i],s=80,edgecolor='k')
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1]+(4,))
        frames.append(frame[:,:,:3])
        plt.close(fig)

    imageio.mimsave(out_gif, frames, fps=3)
    print("Image saved to", out_gif)

    # Compute metrics
    success = 0
    total_rewards = []
    avg_distances = []
    total_collisions = []
    time_to_goal = []

    for ep in range(100):
        obs = env.reset()
        ep_reward = 0
        ep_collisions = 0
        ep_steps_to_goal = 0
        reached_goal = [False]*num_agents
        for step in range(steps):
            obs_tensor = torch.tensor(obs,dtype=torch.float32).to(device)
            actions = [torch.distributions.Categorical(logits=actors[i](obs_tensor[i])).sample().item() for i in range(num_agents)]
            obs, r, _, _ = env.step(actions)
            ep_reward += r
            for i in range(num_agents):
                for j in range(i+1, num_agents):
                    if (env.pos[i] == env.pos[j]).all():
                        ep_collisions += 1
            for i in range(num_agents):
                if not reached_goal[i] and np.linalg.norm(env.pos[i]-env.goal) <= tol:
                    reached_goal[i] = True
                    ep_steps_to_goal += step+1
        total_rewards.append(ep_reward)
        avg_distances.append(np.mean([np.linalg.norm(env.pos[i]-env.goal) for i in range(num_agents)]))
        total_collisions.append(ep_collisions)
        time_to_goal.append(ep_steps_to_goal/num_agents if ep_steps_to_goal>0 else steps)
        dists = [np.linalg.norm(env.pos[i]-env.goal) for i in range(num_agents)]
        if np.all(np.array(dists) <= tol):
            success += 1

    metrics = {
        "Average Reward": np.mean(total_rewards),
        "Success Rate (%)": success,
        "Average Distance to Goal": np.mean(avg_distances),
        "Average Collisions per Episode": np.mean(total_collisions),
        "Average Steps to Goal": np.mean(time_to_goal)
    }
    print("\n--- Evaluation Metrics ---")
    for k,v in metrics.items():
        print(f"{k}: {v:.3f}" if isinstance(v,float) else f"{k}: {v}")
    return metrics

metrics = evaluation(env, actors)
Image(str(out/"marl_demo.gif"))

