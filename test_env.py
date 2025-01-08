from grid_world import RelayConfig, ClientConfig
import numpy as np
from loguru import logger
from gymnasium.envs.registration import register
import gymnasium as gym

# 
size = 1000
relay_config = RelayConfig(num=10, speed=10.)
client_config = ClientConfig(num=100, speed=5.)
# register the environment
register(
    id='GridWorld-v0',
    entry_point='grid_world:GridWorldEnv',
    max_episode_steps=10,
    kwargs={
        "size": 1000,
        "uav_config": relay_config,
        "client_config": client_config,
        "is_plot": True,
        "is_show": False,
    }
)

# create the environment
env = gym.make('GridWorld-v0')


logger.info(f"observation space: {isinstance(env.observation_space, gym.spaces.Dict)}")
logger.info(f"action space: {env.action_space}")

for value in env.action_space.values():
    logger.info(f"observation space: {value}")

seed = 0
# Set seeds
env.action_space.seed(seed)
np.random.seed(seed)
state, info = env.reset(seed=seed)
check_uav_location = True
check_client_location = True
check_center_location = True
check_state = True


for i in range(10):
    
    action = env.action_space.sample()

    if check_uav_location:
        # check whether the uav's location is changed
        old_uav_value = state["uav"]

    if check_client_location:
        # check whether the client's location is changed
        old_client_value = state["client"]

    if check_center_location:
        # check whether the client's location is changed
        old_center_value = state["center"]

    next_state, reward, terminated, truncated, info = env.step(action=action) 
    logger.info(f"reward:{reward}")
    logger.info(f"terminated:{terminated}, truncated:{truncated}")
    logger.info(f"action type: {type(action)}")

    state = next_state

    if check_uav_location:
        # check whether the uav's location is changed
        new_uav_value = state["uav"]
        
        if np.all(new_uav_value == old_uav_value):
            # the uav's location does not change    
            logger.debug("the uav's location does not change")
        elif np.all(np.abs(new_uav_value - old_uav_value) < relay_config.speed):
            # the uav's location changes normally
            logger.info("the uav's location changes normally")
        else:
            # the uav's location changes too much
            logger.warning("the uav's location changes too much")

    if check_client_location:
        # check whether the client's location is changed
        new_client_value = info["client"]

        if np.all(new_client_value < 0) or np.all(new_client_value > size):
            # the client's location is out of the grid
            logger.warning("the client's location is out of the grid")

        if np.all(old_client_value==new_client_value):
            # the client's location does not change
            logger.warning("the client's location does not change")
        elif np.all(np.abs(new_client_value - old_client_value) < client_config.speed):
            # the client's location changes normally
            logger.info("the client's location changes normally")
        else:
            # the client's location changes too much
            logger.warning("the client's location changes too much")

    if check_center_location:
        # check whether the client's location is changed
        new_center_value = info["center"]

        if np.all(new_center_value < 0) or np.all(new_center_value > size):
            # the center's location is out of the grid
            logger.warning("the center's location is out of the grid")

        if np.all(old_center_value==new_center_value):
            # the center's location does not change
            logger.info("the center's location stays the same")
        else:
            # the center's location changes
            logger.warning("the center's location changes")

    logger.info(f"episode {i} is done")

    