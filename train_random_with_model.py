import numpy as np
import torch
import gymnasium as gym
import argparse
import os

from gymnasium.envs.registration import register
from grid_world import RelayConfig, ClientConfig, InitConfig
from wrapper import RelativePosition, FlattenDict, SerializeAction
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from datetime import datetime
from loguru import logger
import train_params_with_model as params

# import the environment
# init the environment
size = params.size
relay_config = params.relay_config
client_config = params.client_config
init_config = params.init_config
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
		"is_show": False,
		"use_model": True,
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
def eval_policy(env_name, seed, eval_episodes=10, writer: SummaryWriter = None):
	eval_env = get_env()

	# eval_env.seed(seed + 100)
	
	if not hasattr(eval_policy, "timestep"):
		eval_policy.timestep = 0
	if not hasattr(eval_policy, "episode_num"):
		eval_policy.episode_num = 0
	else:
		eval_policy.episode_num += 1

	avg_reward = 0.0
	for eval_index in range(eval_episodes):
		state, info= eval_env.reset(seed=seed+100+eval_index)
		done = False
		while not done:
			action = eval_env.action_space.sample()
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
	if writer is not None:
		writer.add_scalar("Reward/Episode/Eval", avg_reward, eval_policy.episode_num)
		
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
	parser.add_argument("--batch_size", default=512, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.5, type=float)     	# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	file_name = f"{args.env}_random_action_with_model_{args.seed}_{current_time}"
	logger.info("---------------------------------------")
	logger.info(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	logger.info("---------------------------------------")

	# get the current file path
	current_file_path = Path(__file__).resolve()
	# get the current directory path
	current_dir_path = current_file_path.parent
	# Set the tensorboard writer
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
	
	# Evaluate untrained policy
	evaluations = [eval_policy(args.env, args.seed, writer=writer)]

	state, _ = env.reset(seed=args.seed)
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action randomly or according to policy
		action = env.action_space.sample()
		
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


		state = next_state
		episode_reward += reward

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			logger.info(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			writer.add_scalar("Reward/Episode", episode_reward, episode_num+1)
			# Reset environment
			state, _ = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(args.env, args.seed, writer=writer))
			np.save(f"./results/{file_name}", evaluations)
