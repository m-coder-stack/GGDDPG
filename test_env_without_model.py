from grid_world import RelayConfig, ClientConfig, InitConfig
import numpy as np
from loguru import logger
from gymnasium.envs.registration import register
import gymnasium as gym

# import the environment
# init the environment
size = 1000
relay_config = RelayConfig(num=10, speed=10.0, limit_position=False, limit_height=True, max_height=100.0, min_height=50.0)
client_config = ClientConfig(num=100, speed=5.0, traffic=2.0, link_establish=2.0, is_move=False)
init_config = InitConfig(center_type="center", relay_type="follow_nearby", client_type="random")
is_polar = False
# register the environment
register(
    id='GridWorld-v0',
    entry_point='grid_world:GridWorldEnv',
    max_episode_steps=1000,
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
        "use_model": False
    }
)


# create the environment
env = gym.make('GridWorld-v0')


logger.info(f"observation space: {isinstance(env.observation_space, gym.spaces.Dict)}")
logger.info(f"action space: {env.action_space}")

# for value in env.action_space.values():
#     logger.info(f"observation space: {value}")

seed = 0
# Set seeds
env.action_space.seed(seed)
np.random.seed(seed)
state, info = env.reset(seed=seed)
check_uav_location = True
check_client_location = True
check_center_location = True
check_state = True


for i in range(100):
    
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action=action) 
    logger.info(f"reward:{reward}")
    logger.info(f"terminated:{terminated}, truncated:{truncated}")
    logger.info(f"action type: {type(action)}")

    state = next_state

    logger.info(f"episode {i} is done")

    