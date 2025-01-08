from wrapper import RelativePosition, FlattenDict, SerializeAction
from gymnasium.envs.registration import register
import gymnasium as gym
import train_params_with_model as params
import modified_DDPG
import OurDDPG
import numpy as np
from loguru import logger

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
        "is_plot": False,
        "is_log": False,
        "use_model": True,
    }
)

def get_env():
    origin_env = gym.make("GridWorld-v0")
    relative_env = RelativePosition(origin_env)
    flatten_env = FlattenDict(relative_env)
    env = SerializeAction(flatten_env, is_polar=is_polar)
    return env

# create the environment
env = get_env()


def load_modified_DDPG():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": 0.5,
            "tau": 0.005,
        }

    kwargs["position_range"] = {
                "position": [-size / 2, size / 2],
                "height": [relay_config.min_height, relay_config.max_height]
            }
    kwargs["relay_dim"] = relay_config.num * 3
    kwargs["client_dim"] = client_config.num * 2
    kwargs["speed"] = relay_config.speed
    policy = modified_DDPG.DDPG(**kwargs)
    policy.load("models/modified_DDPG_GridWorld-v0_with_model_0_2024-10-13_22-50-42")
    return policy

def load_OurDDPG():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.5,
        "tau": 0.005,
    }
    policy = OurDDPG.DDPG(**kwargs)
    policy.load("models/OurDDPG_GridWorld-v0_with_model_0_2024-11-06_20-35-13")
    return policy

def set_seed(seed):
    env.action_space.seed(seed)
    np.random.seed(seed)
    state, info = env.reset(seed=seed)
    return state, info

seed = 0
eval_episodes = 100

def eval(policy_name, policy):
    reward_list = []
    step_reward_list = []
    for eval_index in range(eval_episodes):
        state, info = set_seed(seed + 100 + eval_index)
        done = False
        current_reward = 0
        current_step_reward = []
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, _, done, next_info = env.step(action)
            current_reward += reward
            current_step_reward.append(reward)
            state = next_state
        step_reward_list.append(current_step_reward)
        reward_list.append(current_reward)
    logger.info(f"-- {policy_name} --")
    logger.info(f"reward: {reward_list}")
    logger.info(f"reward length: {len(reward_list)}")
    logger.info(f"reward sum: {sum(reward_list)}")
    logger.info(f"reward average: {sum(reward_list) / len(reward_list)}")
    np.savetxt(f"eval/{policy_name}_reward.txt", reward_list, fmt="%.3f")
    np.savetxt(f"eval/{policy_name}_step_reward.txt", step_reward_list, fmt="%.5f", delimiter=",")

# use modified_DDPG to see the performance
policy = load_modified_DDPG()
eval("modified_DDPG", policy)

# use OurDDPG to see the performance
policy = load_OurDDPG()
eval("OurDDPG", policy)

# use random to see the performance
class RandomPolicy:
    def select_action(self, state):
        return env.action_space.sample()
policy = RandomPolicy()
eval("Random", policy)