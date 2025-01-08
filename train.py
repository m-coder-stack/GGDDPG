import numpy as np
import torch
import gymnasium as gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import modified_DDPG

from gymnasium.envs.registration import register
from grid_world import RelayConfig, ClientConfig, InitConfig
from wrapper import RelativePosition, FlattenDict, SerializeAction
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from datetime import datetime
from loguru import logger

# import the environment
# init the environment
size = 1000
relay_config = RelayConfig(num=10, speed=10.0, limit_position=True, limit_height=True, max_height=100.0, min_height=50.0)
client_config = ClientConfig(num=100, speed=5.0, is_move=False, link_establish=200)
init_config = InitConfig(center_type="center", relay_type="follow", client_type="random")
is_polar = False
# register the environment
register(
    id='GridWorld-v0',
    entry_point='grid_world:GridWorldEnv',
    max_episode_steps=500,
    kwargs={
        "size": size,
        "relay_config": relay_config,
        "client_config": client_config,
		"init_config": init_config,
		"is_polar": is_polar,
		# not suggested to plot the environment
		# it will slow down the training process
		"is_plot": False,
		"is_show": False
    }
)

def get_env():
	origin_env = gym.make("GridWorld-v0")
	relative_env = RelativePosition(origin_env)
	flatten_env = FlattenDict(relative_env)
	env = SerializeAction(flatten_env, is_polar=is_polar)
	return env

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=1, writer: SummaryWriter = None):
	eval_env = get_env()

	# eval_env.seed(seed + 100)
	
	if not hasattr(eval_policy, "timestep"):
		eval_policy.timestep = 0

	avg_reward = 0.0
	for _ in range(eval_episodes):
		state, info= eval_env.reset(seed=seed +100)
		done = False
		while not done:
			action = policy.select_action(np.array(state))
			next_state, reward, _, done, next_info = eval_env.step(action)
			avg_reward += reward

			if writer is not None:
				writer.add_scalar("Reward/Timestep/Eval", reward, eval_policy.timestep)
				if "image" in info:
					writer.add_image("Environment/Image/Eval", info["image"], eval_policy.timestep)
			eval_policy.timestep += 1
			state = next_state
			info = next_info

	avg_reward /= eval_episodes

	logger.info("---------------------------------------")
	logger.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	logger.info("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="modified_DDPG")        # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="GridWorld-v0")          	# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e3, type=int)	# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.5, type=float)     	# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	logger.info("---------------------------------------")
	logger.info(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	logger.info("---------------------------------------")

	# get the current file path
	current_file_path = Path(__file__).resolve()
	# get the current directory path
	current_dir_path = current_file_path.parent
	# Set the tensorboard writer
	current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	writer = SummaryWriter(current_dir_path / f'runs/{file_name}_time_{current_time}')

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = get_env()

	# Set seeds
	# env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)
	elif args.policy == "modified_DDPG":
		kwargs["position_range"] = {
			"position": [-size / 2, size / 2],
			"height": [0.0, relay_config.max_height]
		}
		kwargs["relay_dim"] = relay_config.num * 3
		kwargs["client_dim"] = client_config.num * 2
		kwargs["speed"] = relay_config.speed
		policy = modified_DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, _ = env.reset(seed=args.seed)
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

			# Add the best position to tensorboard
			selected_position = policy.select_position(np.array(state)).reshape([-1, 3])
			writer.add_histogram("State/Relay/Best/Position/Timestep", selected_position[:, :2], t)
			writer.add_histogram("State/Relay/Best/Height/Timestep", selected_position[:, 2:], t)

			actual_position = state[:relay_config.num * 3].reshape([-1, 3])
			writer.add_histogram("State/Relay/Actual/Position/Timestep", actual_position[:, :2], t)
			writer.add_histogram("State/Relay/Actual/Height/Timestep", actual_position[:, 2:], t)

		# Add the action to tensorboard
		action_record = action.reshape([-1, 3])
		writer.add_histogram("Action/Position/Timestep", action_record[:, :2], t)
		writer.add_scalar("Action//Position/Mean", np.abs(action_record[:, :2]).mean(), t)
		writer.add_histogram("Action/Height/Timestep", action_record[:, 2:], t)
		writer.add_scalar("Action/Height/Mean", np.abs(action_record[:, 2:]).mean(), t)
		

		# Perform action
		next_state, reward, _, done, info = env.step(action) 
		done_bool = float(done) if True else 0

		# Add the data to tensorboard
		writer.add_scalar("Reward/timestep", reward, t)
		if "image" in info:
			writer.add_image("Environment/image", np.array(info["image"]).transpose(2, 0, 1), t)
		if "reach_rate" in info:
			writer.add_scalar("Reward/Reach_rate/timestep", info["reach_rate"], t)

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			info_dict = policy.train(replay_buffer, args.batch_size)

			# Get some information from the training process
			# To see the process of the training
			if "critic_loss" in info_dict:
				writer.add_scalar("Loss/Critic_loss/Timestep", info_dict["critic_loss"], t)
			if "actor_loss" in info_dict:
				writer.add_scalar("Loss/Actor_loss/Timestep", info_dict["actor_loss"], t)
			if "position_loss" in info_dict:
				writer.add_scalar("Loss/Position_loss/Timestep", info_dict["position_loss"], t)

			if "target_Q" in info_dict:
				writer.add_histogram("Q/Target_Q/Timestep", info_dict["target_Q"], t)
				writer.add_scalar("Q/Target_Q/Timestep/Mean", info_dict["target_Q"].mean(), t)
				writer.add_scalar("Q/Target_Q/Timestep/Max", info_dict["target_Q"].max(), t)
				writer.add_scalar("Q/Target_Q/Timestep/Min", info_dict["target_Q"].min(), t)
			if "current_Q" in info_dict:
				writer.add_histogram("Q/Current_Q/Timestep", info_dict["current_Q"], t)
				writer.add_scalar("Q/Current_Q/Timestep/Mean", info_dict["current_Q"].mean(), t)
				writer.add_scalar("Q/Current_Q/Timestep/Max", info_dict["current_Q"].max(), t)
				writer.add_scalar("Q/Current_Q/Timestep/Min", info_dict["current_Q"].min(), t)

			if "position_diff" in info_dict:
				writer.add_scalar("Loss/Position/Diff/Timestep", info_dict["position_diff"], t)

			# to see the situation of the network
			# writer.add_histogram("Actor/Linear1/Weights", policy.actor.linear1.weight, t)
			# writer.add_histogram("Actor/Linear1/Bias", policy.actor.linear1.bias, t)
			# writer.add_histogram("Actor/Linear2/Weights", policy.actor.linear2.weight, t)
			# writer.add_histogram("Actor/Linear2/Bias", policy.actor.linear2.bias, t)
			# writer.add_histogram("Actor/Linear3/Weights", policy.actor.linear3.weight, t)
			# writer.add_histogram("Actor/Linear3/Bias", policy.actor.linear3.bias, t)
			# writer.add_histogram("Actor/Linear4/Weights", policy.actor.linear4.weight, t)
			# writer.add_histogram("Actor/Linear4/Bias", policy.actor.linear4.bias, t)

			# writer.add_histogram("Critic/Linear1/Weights", policy.critic.linear1.weight, t)
			# writer.add_histogram("Critic/Linear1/Bias", policy.critic.linear1.bias, t)
			# writer.add_histogram("Critic/Linear2/Weights", policy.critic.linear2.weight, t)
			# writer.add_histogram("Critic/Linear2/Bias", policy.critic.linear2.bias, t)
			# writer.add_histogram("Critic/Linear3/Weights", policy.critic.linear3.weight, t)
			# writer.add_histogram("Critic/Linear3/Bias", policy.critic.linear3.bias, t)
			# writer.add_histogram("Critic/Linear4/Weights", policy.critic.linear4.weight, t)
			# writer.add_histogram("Critic/Linear4/Bias", policy.critic.linear4.bias, t)
			

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			logger.info(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			writer.add_scalar("Reward/Episode", episode_reward, episode_num+1)
			# Reset environment
			state, _ = env.reset(seed=t)
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
