# train the critic alone
# with random actor

from grid_world import UavConfig, ClientConfig
import numpy as np
from loguru import logger
from gymnasium.envs.registration import register
import gymnasium as gym
from wrapper import RelativePosition, FlattenDict, SerializeAction
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.linear1 = nn.Linear(state_dim + action_dim, 1024)
		self.layer_norm1 = nn.LayerNorm(1024)

		self.linear2 = nn.Linear(1024, 512)
		self.layer_norm2 = nn.LayerNorm(512)

		self.linear3 = nn.Linear(512, 256)
		self.layer_norm3 = nn.LayerNorm(256)

		self.linear4 = nn.Linear(256, 1)


	def forward(self, state, action):
		q = self.linear1(torch.cat([state, action], 1))
		q = self.layer_norm1(q)
		q = F.relu(q)

		q = self.linear2(q)
		q = self.layer_norm2(q)
		q = F.relu(q)

		q = self.linear3(q)
		q = self.layer_norm3(q)
		q = F.relu(q)

		q = self.linear4(q)

		return q

# 
size = 1000
uav_config = UavConfig(num=10, speed=10.)
client_config = ClientConfig(num=100, speed=5.)
# register the environment
register(
    id='GridWorld-v0',
    entry_point='grid_world:GridWorldEnv',
    max_episode_steps=500,
    kwargs={
        "size": 1000,
        "uav_config": uav_config,
        "client_config": client_config,
        "is_plot": False,
        "is_show": False,
    }
)

# create the environment
origin_env = gym.make('GridWorld-v0')
relative_env = RelativePosition(origin_env)
flatten_env = FlattenDict(relative_env)
env = SerializeAction(flatten_env)

replay_buffer = ReplayBuffer(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
critic = Critic(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]).to(device)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(f"runs/critic_{current_time}")

max_timesteps = 1e5
start_timesteps = 1e3
seed = 0
done = False
episode_reward = 0
episode_timesteps = 0
episode_num = 0
batch_size = 256
discount = 0.99

	
def train(replay_buffer: ReplayBuffer, critic: Critic, env: gym.Env):
    # Sample replay buffer 
    state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

    next_action = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=(batch_size, env.action_space.shape[0])).astype(np.float32)
    next_action = torch.FloatTensor(next_action).to(device)
    target_Q = critic(next_state, next_action)
    target_Q = reward + (discount * target_Q).detach()
    # Get current Q estimate
    current_Q = critic(state, action)

    # Compute critic loss
    critic_loss = F.mse_loss(current_Q, target_Q)   

    # Optimize the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return critic_loss

state, _ = env.reset(seed=seed)

for t in range(int(max_timesteps)):
    episode_timesteps += 1

    if t >= start_timesteps:
        # train the network
        loss = train(replay_buffer=replay_buffer, critic=critic, env=env)
        writer.add_scalar("Loss/Timestep", loss, t)
    
    
    action = env.action_space.sample()
    next_state, reward, terminated, done, info = env.step(action=action) 
    replay_buffer.add(state, action, next_state, reward, done)
	
    writer.add_scalar("Reward/Timestep", reward, t)

    state = next_state
    episode_reward += reward
	
    if done: 
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        writer.add_scalar("Reward/episode", episode_reward, episode_num+1)
        # Reset environment
        state, _ = env.reset(seed=t)
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
    