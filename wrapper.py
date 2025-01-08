import gymnasium as gym
from gymnasium import ObservationWrapper, ActionWrapper
from gymnasium import spaces
import numpy as np
import copy

# Function to calculate the dimensions of each subspace
def calculate_space_dimensions(space):
    if isinstance(space, gym.spaces.Dict):
        # Sum dimensions of all subspaces for Dict space
        return sum(calculate_space_dimensions(subspace) for subspace in space.spaces.values())
    elif isinstance(space, gym.spaces.Discrete):
        # Discrete space has a single dimension
        return 1
    elif isinstance(space, gym.spaces.Box):
        # Product of the Box shape's dimensions
        return np.prod(space.shape)
    elif isinstance(space, gym.spaces.MultiBinary):
        # MultiBinary space dimension is given by 'n'
        return space.n
    elif isinstance(space, gym.spaces.Tuple):
        # Sum dimensions of all subspaces for Tuple space
        return sum(calculate_space_dimensions(subspace) for subspace in space.spaces)
    # Additional space types can be added here as needed
    else:
        raise TypeError("Unsupported space type")
    

class RelativePosition(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # modify the observation space
        # if the observation space is a dictionary
        # remove the center position from the observation space
        if isinstance(env.observation_space, spaces.Dict):
            obs_relay_position = env.observation_space.spaces["relay"]["position"]
            obs_relay_height = env.observation_space.spaces["relay"]["height"]
            obs_client = env.observation_space.spaces["client"]
            obs_center = env.observation_space.spaces["center"]
            self.observation_space = spaces.Dict({
                "relay": spaces.Dict({
                    "position": spaces.Box(low=obs_relay_position.low.min() - obs_center.high.max(), \
                                           high=obs_relay_position.high.max() - obs_center.low.min(), \
                                              shape=obs_relay_position.shape, dtype=np.float32),
                    "height": obs_relay_height,
                }),
                "client": spaces.Box(low=obs_client.low.min() - obs_center.high.max(), \
                                     high=obs_client.high.max() - obs_center.low.min(), \
                                        shape=obs_client.shape, dtype=np.float32),
            })
            
        else:
            self.observation_space = env.observation_space

    def observation(self, observation):
        return {
            "relay": {
                "position": observation["relay"]["position"] - observation["center"],
                "height": observation["relay"]["height"],
            },
            "client": observation["client"] - observation["center"],
        }
    
class FlattenDict(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), "environment's observation space must be a Dict"
        
        # calculate the size of new observation space
        obs_dim = calculate_space_dimensions(env.observation_space)
        obs_client: spaces.Box = env.observation_space.spaces["client"]
        obs_relay_position: spaces.Box = env.observation_space.spaces["relay"]["position"]
        obs_relay_height: spaces.Box = env.observation_space.spaces["relay"]["height"]

        space_low = np.concatenate([obs_relay_position.low, obs_relay_height.low], axis=1)
        space_low = np.concatenate([space_low, obs_client.low], axis=None)

        space_high = np.concatenate([obs_relay_position.high, obs_relay_height.high], axis=1)
        space_high = np.concatenate([space_high, obs_client.high], axis=None)
        
        # create new observation space
        self.observation_space = spaces.Box(low=space_low, high=space_high, shape=(obs_dim,), dtype=np.float32)

    def observation(self, observation):
        # tranform the observation from dictionary to one-dimensional array
        obs_relay_position = observation["relay"]["position"]
        obs_relay_height = observation["relay"]["height"]
        obs_client = observation["client"]

        obs_relay = np.concatenate([obs_relay_position, obs_relay_height], axis=1)
        return np.concatenate([obs_relay, obs_client], axis=None)
        # return np.concatenate([obs.flatten() for obs in observation.values()])
    

class SerializeAction(ActionWrapper):
    def __init__(self, env: gym.Env, is_polar: bool = True):
        super().__init__(env)
        # assert isinstance(env.action_space, spaces.Dict), "environment's action space must be a Dict"

        self.is_polar = is_polar
        
        if is_polar == False:
            # calculate the size of new action space
            act_dim = calculate_space_dimensions(env.action_space)
            # create new action space
            self.action_space = spaces.Box(low=env.action_space.low.flatten(), high=env.action_space.high.flatten(), shape=(act_dim,), dtype=np.float32)
        else:
            # calculate the size of new action space
            act_dim = calculate_space_dimensions(env.action_space["displacement"]) + calculate_space_dimensions(env.action_space["direction"])
            # create new action space
            space_low = np.concatenate([env.action_space["displacement"].low, env.action_space["direction"].low], axis=1)
            space_low = space_low.flatten()
            space_high = np.concatenate([env.action_space["displacement"].high, env.action_space["direction"].high], axis=1)
            space_high = space_high.flatten()
            self.action_space = spaces.Box(low=space_low, high=space_high, shape=(act_dim,), dtype=np.float32)
        
    def action(self, action):
        # action is a numpy array
        # transform the action from one-dimensional array to dictionary
        if self.is_polar == False:
            return action.reshape([-1,3])
        else:
            action  = action.reshape([-1,3])
            return {
                "displacement": action[:, :1],
                "direction": action[:, 1:],
            }
