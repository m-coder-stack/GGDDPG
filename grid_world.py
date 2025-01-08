import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from loguru import logger

from functions import calculate_displacement, calculate_link_speed, calculate_link_speed_with_model, \
calculate_link_speed_client_relay, calculate_link_speed_client_relay_with_model, \
calculate_link_speed_center, calculate_link_speed_center_with_model

from link_tree import build_tree, draw_map, calculate_reward_from_tree
from enum import Enum


class RelayConfig:
    def __init__(self, num: int = 100, speed: float = 1.0, limit_position: bool = True, limit_height: bool = True, max_height: float = 100.0, min_height: float = 50.0) -> None:
        """
        Initialize the grid world environment.

        Args:
            num (int, optional): The number of elements in the grid. Defaults to 100.
            speed (float, optional): The speed of movement within the grid. Defaults to 1.0.
            limit_position (bool, optional): Flag to limit the position within the grid. Defaults to True.
            limit_height (bool, optional): Flag to limit the height within the grid. Defaults to True.
            max_height (float, optional): The maximum height allowed in the grid. Defaults to 100.0.
            min_height (float, optional): The minimum height allowed in the grid. Defaults to 50.0.
        """
        self.num = num
        self.speed = speed
        self.limit_position = limit_position
        self.limit_height = limit_height
        self.max_height = max_height
        self.min_height = min_height
        

class ClientConfig:
    def __init__(self, num: int = 10, speed: float = 1.0, traffic: float = 2.0, link_establish: float = 200.0, is_move: bool = True) -> None:
        """
        Initialize the grid world environment.

        Args:
            num (int, optional): Number of elements in the grid. Defaults to 10.
            speed (float, optional): Speed of movement within the grid. Defaults to 1.0.
            traffic (float, optional): Traffic level within the grid. Defaults to 2.0.
            link_establish (float, optional): Time to establish a link. Defaults to 200.0.
            is_move (bool, optional): Flag indicating if movement is allowed. Defaults to True.
        """
        self.num = num
        self.speed = speed
        self.traffic = traffic
        self.link_establish = link_establish
        self.is_move = is_move

class InitConfig:

    class CenterType(Enum):
        RANDOM = 1
        CENTER = 2

    class RelayType(Enum):
        RANDOM = 1
        CENTER = 2
        FOLLOW = 3
        FOLLOW_NEARBY = 4

    class ClientType(Enum):
        RANDOM = 1
        REGULAR = 2

    def __init__(self, center_type: str = "random", relay_type: str = "random", client_type: str = "random") -> None:
        """
        Initialize the GridWorld environment with specified types for center, relay, and client.
        Args:
            center_type (str): The type of center to initialize. Must be one of the members of InitConfig.CenterType.
                               Default is "random".
            relay_type (str): The type of relay to initialize. Must be one of the members of InitConfig.RelayType.
                              Default is "random".
            client_type (str): The type of client to initialize. Must be one of the members of InitConfig.ClientType.
                               Default is "random".
        Raises:
            ValueError: If any of the provided types (center_type, relay_type, client_type) are not valid members of their
                        respective InitConfig enums.
        """
        if center_type.upper() in InitConfig.CenterType.__members__:
            self.center_type = InitConfig.CenterType.__members__[center_type.upper()]
        else:
            raise ValueError(f"center_type should be in {InitConfig.CenterType.__members__.keys()}")
        

        if relay_type.upper() in InitConfig.RelayType.__members__:
            self.relay_type = InitConfig.RelayType.__members__[relay_type.upper()]
        else:
            raise ValueError(f"relay_type should be in {InitConfig.RelayType.__members__.keys()}")

        if client_type.upper() in InitConfig.ClientType.__members__:
            self.client_type = InitConfig.ClientType.__members__[client_type.upper()]
        else:
            raise ValueError(f"client_type should be in {InitConfig.ClientType.__members__.keys()}")
        


class GridWorldEnv(gym.Env):
    """
    3d grid world communication environment
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: float = 1000.0, init_config: InitConfig = InitConfig(),\
                 relay_config: RelayConfig = RelayConfig(), client_config: ClientConfig = ClientConfig(),\
                 is_polar: bool = True, is_plot: bool = False, is_show: bool = True, is_log: bool = False, use_model: bool = False, keep_plot_data: bool = False):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Save the parameters of the environment
        self.is_polar = is_polar
        self.is_plot = is_plot
        self.is_show = is_show
        self.is_log = is_log
        self.use_model = use_model
        self.keep_plot_data = keep_plot_data

        self.init_config = init_config
        self.relay_config = relay_config
        self.client_config = client_config

        # for now we assume that the link_establish is equal to the traffic
        if self.client_config.link_establish != self.client_config.traffic:
            raise ValueError("link_establish should be equal to the traffic")

        # info
        self._info = {}

        # plot data
        self.plot_data = {}


        # Observations are dictionaries with the relay's and the client's location.
        self.observation_space = spaces.Dict(
            {
                "relay": spaces.Dict(
                    {
                        "position": spaces.Box(low=0.0, high=size, shape=(self.relay_config.num, 2), dtype=np.float32),
                        "height": spaces.Box(low=relay_config.min_height, high=relay_config.max_height, shape=(self.relay_config.num, 1), dtype=np.float32),
                    },
                ),
                "client": spaces.Box(low=0.0, high=size, shape=(self.client_config.num, 2), dtype=np.float32),
                "center": spaces.Box(low=0.0, high=size, shape=(1, 2), dtype=np.float32),
            }
        )

        if self.is_polar:
            # there are two params, one is the speed and the other is the direction
            self.action_space = spaces.Dict(
                {
                    "displacement": spaces.Box(low=-relay_config.speed, high=relay_config.speed, shape=(self.relay_config.num, 1), dtype=np.float32),
                    "direction": spaces.Box(low=np.tile(np.array([[0, -np.pi]], dtype=np.float32), reps=(self.relay_config.num, 1)), \
                                            high=np.pi, shape=(self.relay_config.num, 2), dtype=np.float32),
                    
                }
            )
        else:
            # just use the axis to control the relay
            self.action_space = spaces.Box(low=-relay_config.speed, high=relay_config.speed, shape=(self.relay_config.num, 3), dtype=np.float32)
        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_state(self):
        """
        Returns the state of the environment
        all value are copied so that the original value will not be changed
        can be used freely
        """
        return {
            "relay": {
                "position": self._relay_location.copy(),
                "height": self._relay_height.copy(),
            },
            "client": self._client_location.copy(),
            "center": self._center_location.copy(),
        }
    
    # def get_state_dim(self):
    #     state_dim = 0
    #     for value in self.observation_space.values():
    #         state_dim += value.size
    #     return state_dim
    
    # def get_action_dim(self):
    #     action_dim = 0
    #     for value in self.action_space.values():
    #         action_dim += value.size
    #     return action_dim


    def _get_info(self):
        """
        TODO: need to think the return value
        """
        return self._info
    
    def _get_reward(self):
        """
        calculate the reward of the environment
        """

        # copy the value so that the original value will not be changed
        relay_location = self._relay_location.copy()
        relay_height = self._relay_height.copy()
        client_location = self._client_location.copy()
        center_location = self._center_location.copy()

        # first we need to calculate the distance between the relay and the client
        # to find out which relay can client communicate with
        client_link = - np.ones(self.client_config.num, dtype=np.int32) # record the link between the client and the relay
        
        # calculate the link speed between the client and the relay
        # find the max link speed and the index of the relay
        if not self.use_model:
            client_relay_link_speed_matrix = calculate_link_speed_client_relay(client_location_list=client_location, \
                                                relay_location_list=relay_location, relay_height_list=relay_height)
            client_relay_max_index = np.argmax(client_relay_link_speed_matrix, axis=1)
            client_relay_max_speed = client_relay_link_speed_matrix[np.arange(self.client_config.num), client_relay_max_index]
        else:
            client_relay_link_speed_matrix = calculate_link_speed_client_relay_with_model(client_location_list=client_location, \
                                                relay_location_list=relay_location, relay_height_list=relay_height)
            client_relay_max_index = np.argmax(client_relay_link_speed_matrix, axis=1)
            client_relay_max_speed = client_relay_link_speed_matrix[np.arange(self.client_config.num), client_relay_max_index]
        client_link = np.where(client_relay_max_speed > self.client_config.link_establish, client_relay_max_index, -1)
           

        # then we need to calculate the traffic which the relay need to handle
        relay_client_num = np.zeros(self.relay_config.num, dtype=np.int32)
        for i in range(self.relay_config.num):
            relay_client_num[i] = np.sum(client_link == i)
        relay_data_amount = relay_client_num * self.client_config.traffic

        # then build the link martrix between the relay 
        if not self.use_model:
            relay_link_speed_matrix = calculate_link_speed(location_list=relay_location, height_list=relay_height)
        else:
            relay_link_speed_matrix = calculate_link_speed_with_model(location_list=relay_location, height_list=relay_height)

        # then we need to calculate the link speed between the relay and the center
        if not self.use_model:
            relay_center_link_speed_list = calculate_link_speed_center(location_list=relay_location, height_list=relay_height, center_location=center_location)
        else:
            relay_center_link_speed_list = calculate_link_speed_center_with_model(location_list=relay_location, height_list=relay_height, center_location=center_location)
        
        # calculate the order of relay to handle
        # cause the uncertainty between the height and the link speed, just use the location
        relay_center_distance = np.linalg.norm(relay_location - center_location, axis=1)
        relay_index = np.argsort(relay_center_distance)


        # calculate the reward
        # get the traffic load of the center node
        # then calculate the percentage of the traffic transmitted by the center node

        center_node = build_tree(center_node_speed_list=relay_center_link_speed_list, relay_node_speed_matrix=relay_link_speed_matrix,\
                                 relay_node_data_amount=relay_data_amount, relay_node_index=relay_index)
        
        # calculate the relay traffic to show
        relay_traffic = np.zeros(self.relay_config.num, dtype=np.float32)
        relay_traffic_dict = center_node.get_traffic_load()
        for i in range(self.relay_config.num):
            relay_traffic[i] = relay_traffic_dict[i]
        
        # plot the map
        if self.keep_plot_data:
            self.plot_data["tree"] = center_node.build_tree()
            self.plot_data["center_position"] = center_location.copy()
            self.plot_data["relay_position"] = relay_location.copy()
            self.plot_data["client_position"] = client_location.copy()
            self.plot_data["relay_height"] = relay_height.copy()
            self.plot_data["relay_traffic"] = relay_traffic.copy()
            self.plot_data["client_link"] = client_link.copy()
            self.plot_data["size"] = [[0, self.size], [0, self.size]]
            self.plot_data["is_show"] = self.is_show
            self.plot_data["center_node"] = center_node
            self._info["plot_data"] = self.plot_data

        if self.is_plot:
            image = draw_map(tree=center_node.build_tree(), center_position=center_location, relay_position=relay_location, client_position=client_location, \
                             relay_height=relay_height, relay_traffic=relay_traffic, client_link=client_link, \
                             size=[[0, self.size], [0, self.size]], is_show=self.is_show)
            if not self.is_show and image is not None:
                self._info["image"] = image
        
        if self.is_log:
            # debug the tree
            logger.debug(f"center_node.traffic_load: {center_node.traffic_load}")
            logger.debug(f"client_num * client_traffic: {self.client_config.num * self.client_config.traffic}")
            logger.debug("\n" + center_node.build_tree().show(stdout=False))
        

            # ergodic the tree and print the information of the node
            node_stack = [center_node]
            while node_stack:
                node = node_stack.pop()
                logger.debug(f"node id: {node.node_id}, node_parent: {node.parent.node_id if node.parent else "None"}, "
                        f"node data amount: {node.data_amount:.2f}, node traffic load: {node.traffic_load:.2f}, "
                        f"node parent link speed: {node.link_speed:.2f}, node center link speed: {relay_center_link_speed_list[node.node_id]:.2f}")
                
                node_stack.extend(node.children)

        # collect data for debug
        self._info["relay_link_speed_matrix"] = relay_link_speed_matrix
        self._info["relay_center_link_speed_list"] = relay_center_link_speed_list
        self._info["relay_data_amount"] = relay_data_amount
        self._info["relay_traffic"] = relay_traffic
        self._info["center_node"] = center_node

        # calculate distance
        relay_distance_matrix = np.zeros((self.relay_config.num, self.relay_config.num))
        for i in range(relay_location.shape[0]):
        # calculate the link speed between the i-th location and the j-th location
        # construct the upper triangular matrix
            distance_list = np.linalg.norm(relay_location[i+1:] - relay_location[i], axis=1)
            relay_distance_matrix[i][i+1:] = distance_list
        # construct the full matrix
        relay_distance_matrix = relay_distance_matrix + relay_distance_matrix.T
        self._info["relay_distance_matrix"] = relay_distance_matrix

        # calculate the distance between the relay and the center
        relay_center_distance = np.linalg.norm(relay_location - center_location, axis=1)
        self._info["relay_center_distance"] = relay_center_distance

        reach_rate = center_node.traffic_load / (self.client_config.num * self.client_config.traffic)
        self._info["reach_rate"] = reach_rate
        
        reward = reach_rate
        # reward = calculate_reward_from_tree(center_node)

        # assert reward >= 0.0 and reward <= 1.0, f"reward: {reward}"
        
        return reward
    

    
    def _update_client_location(self):
        """
        update the client's location
        """
        # update the client's location
        # the location of client is 2d, so we need to calculate the displacement and the direction
        client_displacement = self.np_random.uniform(low=-1.0, high=1.0, size=(self.client_config.num, 1))
        client_direction = self.np_random.uniform(low=-np.pi, high=np.pi, size=(self.client_config.num, 1))
        self._client_location += calculate_displacement(displacement=client_displacement, direction=client_direction, speed=self.client_config.speed)
        # make sure that all the clients are in the grid
        self._client_location = np.clip(self._client_location, 0.0, self.size)


    def reset(self, seed=None, options=None):
        """
        reset the environment
        Args:
            seed: int, the seed of the environment
            options: dict, the options of the environment
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the relay's, the center's and the client's location

        # init the center's location
        if self.init_config.center_type == InitConfig.CenterType.RANDOM:
            self._center_location = self.np_random.uniform(low=0., high=self.size, size=(1, 2)).astype(np.float32)
        elif self.init_config.center_type == InitConfig.CenterType.CENTER:
            self._center_location = np.array([[self.size / 2, self.size / 2]], dtype=np.float32)
        else:
            raise ValueError(f"center_type should be in {InitConfig.CenterType.__members__.keys()}")
        
        # init the relay's location
        if self.init_config.relay_type == InitConfig.RelayType.RANDOM:
            self._relay_location = self.np_random.uniform(low=0.0, high=self.size, size=(self.relay_config.num, 2)).astype(np.float32)
        elif self.init_config.relay_type == InitConfig.RelayType.CENTER:
            self._relay_location = np.tile(np.array([[self.size / 2, self.size / 2]], dtype=np.float32), reps=(self.relay_config.num, 1))
        elif self.init_config.relay_type == InitConfig.RelayType.FOLLOW:
            self._relay_location = np.tile(self._center_location, (self.relay_config.num, 1))
        elif self.init_config.relay_type == InitConfig.RelayType.FOLLOW_NEARBY:
            self._relay_location = np.tile(self._center_location, (self.relay_config.num, 1))
            self._relay_location += self.np_random.uniform(low=-100.0, high=100.0, size=(self.relay_config.num, 2)).astype(np.float32)
        else:
            raise ValueError(f"relay_type should be in {InitConfig.RelayType.__members__.keys()}")
        # init the relay's height
        self._relay_height = np.ones((self.relay_config.num, 1), dtype=np.float32) * self.relay_config.min_height
        
        # init the client's location
        self._client_location = self.np_random.uniform(low=0.0, high=self.size, size=(self.client_config.num, 2)).astype(np.float32)

        
        state = self._get_state()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return state, info

    def step(self, action):
        """
        take a step in the environment
        Args:
            action: dict, the action of the relay

        Returns:
            state: dict, the state of the relay can observe
            reward: float, the reward of the relay
            done: bool, whether the episode is done
            info: dict, the information of the environment
        """
        # calculate the reward
        reward = self._get_reward()


        # update the relay's location
        if self.is_polar:
            update_movement = calculate_displacement(displacement=action["displacement"], direction=action["direction"])
            self._relay_location += update_movement[:, :2]
            self._relay_height += update_movement[:, 2].reshape(-1, 1)
        else:
            self._relay_location += action[:, :2]
            self._relay_height += action[:, 2].reshape(-1, 1)
        if self.relay_config.limit_position:
            self._relay_location = np.clip(self._relay_location, 0.0, self.size)
        if self.relay_config.limit_height:
            self._relay_height = np.clip(self._relay_height, self.relay_config.min_height, self.relay_config.max_height)
        # update the client's location
        if self.client_config.is_move:
            self._update_client_location()
        
        state = self._get_state()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return state, reward, False, False, info
    

    """
    The following methods are used to render the environment
    """
    def render(self):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    data = np.random.uniform(low=0., high=1000., size=(10, 2))
    logger.info(calculate_link_speed(location_list=data))