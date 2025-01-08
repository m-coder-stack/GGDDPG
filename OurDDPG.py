import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, max_action: float):
		"""
		Initializes the Actor network.
		Args:
			state_dim (int): Dimension of the state space.
			action_dim (int): Dimension of the action space.
			max_action (float): Maximum action value. The action space is assumed to be symmetric.
		"""
		super(Actor, self).__init__()

		self.linear1 = nn.Linear(state_dim, 1024)
		self.layer_norm1 = nn.LayerNorm(1024)

		self.linear2 = nn.Linear(1024, 512)
		self.layer_norm2 = nn.LayerNorm(512)

		self.linear3 = nn.Linear(512, 256)
		self.layer_norm3 = nn.LayerNorm(256)

		self.linear4 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

	
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
		
		return self.max_action * torch.tanh(a)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		"""
		Initializes the Critic network.

		Args:
			state_dim (int): Dimension of the state space.
			action_dim (int): Dimension of the action space.
		"""
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


class DDPG(object):
	def __init__(self, state_dim: int, action_dim:int , max_action: float, discount=0.99, tau=0.005):
		"""
		Initializes the DDPG agent with the given parameters.
		Args:
			state_dim (int): Dimension of the state space.
			action_dim (int): Dimension of the action space.
			max_action (float): Maximum value for an action. The action space is assumed to be symmetric.
			discount (float, optional): Discount factor for future rewards. Default is 0.99.
			tau (float, optional): Target network update rate. Default is 0.005.
		"""
		
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

		self.discount = discount
		self.tau = tau


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()
	
	def calculate_Q(self, state, action):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		action = torch.FloatTensor(action.reshape(1, -1)).to(device)
		return self.critic(state, action).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=4096):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
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
		# we hope that the action be small, so we add a L2 regularization term
		pred_action = self.actor(state)
		actor_loss = -self.critic(state, pred_action).mean()

		# pred_action_l2 = torch.linalg.vector_norm(pred_action, dim=1)
		# actor_loss += pred_action_l2.mean() * 1e-2 
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return {
			"critic_loss": critic_loss,
			"actor_loss": actor_loss,
			"target_Q": target_Q,
			"current_Q": current_Q
		}

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		