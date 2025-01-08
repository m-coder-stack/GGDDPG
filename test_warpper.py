from grid_world import UavConfig, ClientConfig
import numpy as np
from loguru import logger
from gymnasium.envs.registration import register
import gymnasium as gym
from wrapper import RelativePosition, FlattenDict, SerializeAction

# 
size = 1000
uav_config = UavConfig(num=10, speed=10.)
client_config = ClientConfig(num=100, speed=5.)
# register the environment
register(
    id='GridWorld-v0',
    entry_point='grid_world:GridWorldEnv',
    max_episode_steps=10,
    kwargs={
        "size": 1000,
        "uav_config": uav_config,
        "client_config": client_config,
        "is_plot": True,
        "is_show": False,
    }
)

# create the environment
origin_env = gym.make('GridWorld-v0')

relative_env = RelativePosition(origin_env)
flatten_env = FlattenDict(relative_env)
env = SerializeAction(flatten_env)


logger.info(f"observation space: {env.observation_space}")
logger.info(f"action space: {env.action_space}")



seed = 0
# Set seeds
env.action_space.seed(seed)
np.random.seed(seed)
state, info = env.reset(seed=seed)
logger.info(f"state shape: {state.shape[0]}, state dtype: {state.dtype}")



for i in range(10):
    state, info = env.reset(seed=i)
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action=action) 
    info["center_node"].check_tree()
    logger.info(f"reward:{reward:.2f}")
    # logger.info(f"terminated:{terminated}, truncated:{truncated}")
    # logger.info(f"action type: {type(action)}")

    if "image" in info:
        logger.info(f"image shape: {info['image'].save(f"{i}.png")}")

    state = next_state
    # logger.info(f"state shape: {state.shape[0]}, state dtype: {state.dtype}")

    logger.info(f"episode {i} is done")

    