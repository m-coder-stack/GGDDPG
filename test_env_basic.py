from grid_world import RelayConfig, ClientConfig, InitConfig
import numpy as np
from loguru import logger
from gymnasium.envs.registration import register
import gymnasium as gym

# import the environment
# init the environment
size = 4000
relay_config = RelayConfig(num=10, speed=10.0, limit_position=False, limit_height=True, max_height=500.0, min_height=100.0)
client_config = ClientConfig(num=100, speed=5.0, traffic=20.0, is_move=False, link_establish=10.0)
init_config = InitConfig(center_type="center", relay_type="random", client_type="random")
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
        "use_model": True
    }
)


# create the environment
env = gym.make('GridWorld-v0')


logger.info(f"observation space: {isinstance(env.observation_space, gym.spaces.Dict)}")
logger.info(f"action space: {env.action_space}")

# for value in env.action_space.values():
#     logger.info(f"observation space: {value}")

seed = 1
# Set seeds
env.action_space.seed(seed)
np.random.seed(seed)
state, info = env.reset(seed=seed)
check_uav_location = True
check_client_location = True
check_center_location = True
check_state = True


for i in range(1000):
    
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action=action) 
    logger.info(f"reward:{reward}")
    logger.info(f"terminated:{terminated}, truncated:{truncated}")
    logger.info(f"action type: {type(action)}")

    state = next_state

    logger.info(f"episode {i} is done")

    