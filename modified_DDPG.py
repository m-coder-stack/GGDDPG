import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import ReplayBuffer
from typing import Optional, Union


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, max_action: float, output_range: Optional[Union[list, dict]] = None):
		super(Actor, self).__init__()

		self.linear1 = nn.Linear(state_dim, 1024)
		self.layer_norm1 = nn.LayerNorm(1024)

		self.linear2 = nn.Linear(1024, 512)
		self.layer_norm2 = nn.LayerNorm(512)

		self.linear3 = nn.Linear(512, 256)
		self.layer_norm3 = nn.LayerNorm(256)

		self.linear4 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		self.output_range = output_range

	
	def forward(self, state):
		a = self.linear1(state)
		a = self.layer_norm1(a)
		a = F.relu(a)

		a = self.linear2(a)
		a = self.layer_norm2(a)
		a = F.relu(a)

		a = self.linear3(a)
		a = self.layer_norm3(a)
		a = F.relu(a)

		a = self.linear4(a)

		if self.output_range is None:
			return torch.tanh(a) * self.max_action
		elif type(self.output_range) == list:
			return self.output_range[0] + (F.tanh(a) + 1) / 2 * (self.output_range[1] - self.output_range[0])
		elif type(self.output_range) == dict:
			output: torch.Tensor = F.tanh(a)
			output = output.clone()
			# expect the shape of output is (batch_size, data_dim)
			output = output.reshape([output.shape[0], -1, 3])
			position_range = self.output_range["position"]
			height_range = self.output_range["height"]
			output[:, :, :2] = position_range[0] + (output[:, :, :2] + 1) / 2 * (position_range[1] - position_range[0])
			output[:, :, 2] = height_range[0] + (output[:, :, 2] + 1) / 2 * (height_range[1] - height_range[0])
			output = output.reshape([output.shape[0], -1])
			return output
		else:
			raise ValueError("The output range should be a list or a dictionary")


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim ,output_range: list = None):
		super(Critic, self).__init__()

		self.output_range = output_range

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

		if self.output_range is not None:
			q = self.output_range[0] + (F.tanh(q) + 1) / 2 * (self.output_range[1] - self.output_range[0])

		return q


class DDPG(object):
	def __init__(self, state_dim: int, action_dim: int, max_action: float, discount=0.99, tau=0.005,\
			  position_range: Optional[Union[list, dict]] = None, relay_dim:int = 30, client_dim: int = 200, speed: float = 10.0):
		
		assert state_dim == relay_dim + client_dim, f"The state dimension should be {relay_dim + client_dim}, but got {state_dim}"
		
		# the output range of the actor is (-1, 1)
		self.actor = Actor(relay_dim + state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-3)

		# the output range of the critic is (0, discount / (1 - discount)
		self.critic = Critic(state_dim, action_dim, [0, discount / (1 - discount)]).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

		# the output range of the position actor is position_range
		self.position_actor = Actor(client_dim, relay_dim, max_action=1.0, output_range=position_range).to(device)
		self.position_actor_optimizer = torch.optim.Adam(self.position_actor.parameters(), lr=1e-4, weight_decay=1e-3)

		self.discount = discount
		self.tau = tau
		self.position_range = position_range
		self.relay_dim = relay_dim
		self.client_dim = client_dim
		self.speed = speed


	def select_action(self, state: np.ndarray) -> np.ndarray:
		"""
		Selects an action based on the given state.

		Parameters:
		- state (np.ndarray): The current state of the environment.

		Returns:
		- np.ndarray: The selected action.

		"""
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		state = torch.cat([state, self.position_actor(state[:, self.relay_dim:])], dim=1)
		return self.actor(state).cpu().data.numpy().flatten()
	
	def select_position(self, state: np.ndarray) -> np.ndarray:
		# the shape of state should be (1, client_dim)
		# the composition of the state is (relay_dim, client_dim)
		# the output should be the (1, relay_dim)
		state = torch.FloatTensor(state[self.relay_dim:].reshape(1, -1)).to(device)
		return self.position_actor(state).cpu().data.numpy().flatten()
	
	def calculate_Q(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
		"""
		Calculate the Q-value for a given state-action pair.

		Parameters:
		- state (np.ndarray): The state array.
		- action (np.ndarray): The action array.

		Returns:
		- np.ndarray: The Q-value for the given state-action pair.
		"""
		state = torch.FloatTensor(state).to(device).reshape(1, -1)
		action = torch.FloatTensor(action).to(device).reshape(1, -1)
		return self.critic(state, action).cpu().data.numpy().flatten()


	def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		next_state_input = torch.cat([next_state, self.position_actor(next_state[:, self.relay_dim:])], dim=1)
		target_Q = self.critic_target(next_state, self.actor_target(next_state_input))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		state_input = torch.cat([state, self.position_actor(state[:, self.relay_dim:])], dim=1)
		expect_action = self.actor(state_input)

		expect_relay_state = copy.deepcopy(state[:, :self.relay_dim]) + expect_action * self.speed
		best_position = self.position_actor(state[:, self.relay_dim:])
		position_diff = F.mse_loss(expect_relay_state, best_position)
		actor_loss = -self.critic(state, expect_action).mean() + 1e-5 * position_diff
		
		# Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Compute the best position for the situation
		position_action = self.position_actor(state[:, self.relay_dim:])
		expect_state = torch.zeros_like(state)
		expect_state[:, :self.relay_dim] = position_action
		expect_state[:, self.relay_dim:] = state[:, self.relay_dim:]
		position_reward = self.critic(expect_state, torch.zeros_like(action)).mean()
		position_loss = -position_reward

		# optimize the position actor
		self.position_actor_optimizer.zero_grad()
		position_loss.backward()
		self.position_actor_optimizer.step()


		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return {
			"critic_loss": critic_loss,
			"actor_loss": actor_loss,
			"position_loss": position_loss,
			"position_diff": position_diff,
			"target_Q": target_Q,
			"current_Q": current_Q
		}

	def save(self, filename: str):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		torch.save(self.position_actor.state_dict(), filename + "_position_actor")
		torch.save(self.position_actor_optimizer.state_dict(), filename + "_position_actor_optimizer")


	def load(self, filename: str):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

		self.position_actor.load_state_dict(torch.load(filename + "_position_actor"))
		self.position_actor_optimizer.load_state_dict(torch.load(filename + "_position_actor_optimizer"))
		