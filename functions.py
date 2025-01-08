import numpy as np
from solver import solve_param
from typing import Optional, Union, List
from scipy.interpolate import RegularGridInterpolator
from speed_model import get_data, expand_data, get_model

# the default bandwidth is 20 MHz
param_B = 20.0

# when the distance is larger than distance_max, the link speed is 0
# when the distance is smaller than distance_min, the link speed is rate_reference
# when the distance is between distance_min and distance_max, the link speed is calculated by the formula

distance_min = 50
rate_reference = 50
param_alpha = solve_param(distance=distance_min,rate=rate_reference, bandwidth=param_B)
distance_max = 400

# the default model
x_train_air, y_train_air, z_train_air = get_data("./speed_modeling_data/air_air_data.csv")
model_air = get_model(x_train=x_train_air, y_train=y_train_air, z_train=z_train_air, method="linear")
distance_max_air = 4000


x_train_ground, y_train_ground, z_train_ground = get_data("./speed_modeling_data/air_ground_data.csv")
model_ground = get_model(x_train=x_train_ground, y_train=y_train_ground, z_train=z_train_ground, method="linear")
distance_max_ground = 3000


def calculate_displacement(displacement: np.ndarray, direction: np.ndarray, speed:Optional[float] = None) -> np.ndarray:
    """
    Calculates the displacement vector based on the given displacement, direction, and speed.
    Supports 2d and 3d environments.

    Args:
        displacement (np.ndarray): The displacement vector.
        direction (np.ndarray): The direction vector.
        speed (Optional[float], optional): The speed scalar. Defaults to None.

    Returns:
        np.ndarray: The calculated displacement vector.
    """
    
    assert type(displacement) is np.ndarray, "The displacement should be a numpy array"
    assert type(direction) is np.ndarray, "The direction should be a numpy array"
    assert displacement.shape[0] == direction.shape[0], "The length of the displacement and direction should be the same"
    assert displacement.shape[1] == 1, "The displacement should be a column vector"
    assert direction.shape[1] in [1, 2], "The direction should be one or two columns"
    
    
    if speed is not None:
        assert np.all(displacement >= -1) and np.all(displacement <= 1), "The displacement must be in the range of (-1, 1), the actual value is \n {}".format(displacement)
        assert np.all(speed >= 0), "The speed must be greater than or equal to 0"
    else:
        speed = 1.0

    if direction.shape[1] == 1:
        assert np.all(direction >= -np.pi) and np.all(direction <= np.pi), "The direction must be in the range of (-pi, pi), the actual value is \n {}".format(direction)
        return np.concatenate([
            displacement * speed * np.sin(direction),
            displacement * speed * np.cos(direction)
        ], axis=1)
    else:
        assert np.all(direction[:, 0] >= 0) and np.all(direction[:, 0] <= np.pi), "The direction 0 must be in the range of (0, pi), the actual value is \n {}".format(direction)
        assert np.all(direction[:, 1] >= -np.pi) and np.all(direction[:, 1] <= np.pi), "The direction 0 must be in the range of (-pi, pi), the actual value is \n {}".format(direction)
        return np.concatenate([
            displacement * speed * np.sin(direction[:, 0]) * np.cos(direction[:, 1]),
            displacement * speed * np.sin(direction[:, 0]) * np.sin(direction[:, 1]),
            displacement * speed * np.cos(direction[:, 0])
        ], axis=1)


def calculate_link_speed(location_list: np.ndarray, height_list: Optional[np.ndarray] = None, \
                         param_B: Optional[float] = None, param_alpha: Optional[float] = None) -> np.ndarray:
    """
    Calculate the link speed between the two locations.

    Parameters:
        location_list (np.ndarray): An array of locations.
        param_B (float): Parameter B.
        param_alpha (float): Parameter alpha.

    Returns:
        np.ndarray: The link speed matrix.
    """
    if param_B is None:
        param_B = globals()['param_B']
    if param_alpha is None:
        param_alpha = globals()['param_alpha']

    # ensure the length of the location_list and height_list is the same
    if height_list is not None:
        assert location_list.shape[0] == height_list.shape[0], "The length of the location_list and height_list should be the same"
    
    link_speed_matrix = np.zeros(shape=(location_list.shape[0], location_list.shape[0]), dtype=np.float32)
    for i in range(location_list.shape[0] - 1):
        # calculate the link speed between the i-th location and the j-th location
        # construct the upper triangular matrix
        if height_list is not None:
            distance_list = np.linalg.norm(np.concatenate([location_list[i+1:] - location_list[i], height_list[i+1:] - height_list[i]], axis = 1, dtype=np.float32), axis=1)
        else:
            distance_list = np.linalg.norm(location_list[i+1:] - location_list[i], axis=1)
        # bug fix: when the distance is 0, the link speed calculation will be wrong and throw error
        # so just make it bigger
        distance_list = np.where(distance_list < distance_min, distance_min, distance_list)
        link_speed_list = param_B * np.log2((1 + param_alpha / (distance_list ** 2)).astype(np.float32))
        link_speed_list = np.where(distance_list <= distance_min, rate_reference, link_speed_list)
        link_speed_list = np.where(distance_list >= distance_max, 0, link_speed_list)
        link_speed_matrix[i][i+1:] = link_speed_list

    # construct the full matrix
    link_speed_matrix = link_speed_matrix + link_speed_matrix.T

    # check the result
    # assert np.all(link_speed_matrix >= 0), "The link speed matrix should be greater than or equal to 0"
    # assert np.all(link_speed_matrix <= rate_reference), "The link speed matrix should be less than or equal to the reference rate"

    return link_speed_matrix


def calculate_link_speed_with_model(location_list: np.ndarray, height_list: Optional[np.ndarray] = None, \
                                    model: RegularGridInterpolator = None) -> np.ndarray:
    """
    Calculate the link speed using a given model based on location and optional height data.
    Args:
        location_list (np.ndarray): An array of location coordinates.
        height_list (Optional[np.ndarray], optional): An array of height values corresponding to the locations. Defaults to None.
        model (RectBivariateSpline, optional): A pre-trained RectBivariateSpline model to calculate the link speed. Defaults to None.
    Returns:
        np.ndarray: An array of calculated link speeds.
    """
    if model is None:
        model = globals()['model_air']

    if height_list is None:
        height_list = np.zeros((location_list.shape[0], 1), dtype=np.float32)

    # init the link speed matrix
    link_speed_matrix = np.zeros(shape=(location_list.shape[0], location_list.shape[0]), dtype=np.float32)

    # calculate the distance and height matrix
    for i in range(location_list.shape[0] - 1):
        distance_list = np.linalg.norm(location_list[i+1:] - location_list[i], axis=1)
        height_list_diff = np.abs(height_list[i+1:] - height_list[i])
        distance_list = np.where(distance_list > distance_max_air, distance_max_air, distance_list)
        points_list = np.concatenate([distance_list.reshape(-1, 1), height_list_diff], axis=1)
        link_speed_matrix[i][i+1:] = model(points_list)

    # construct the full matrix
    link_speed_matrix = link_speed_matrix + link_speed_matrix.T

    return link_speed_matrix

def calculate_link_speed_client_relay(client_location_list: np.ndarray, relay_location_list: np.ndarray, relay_height_list: np.ndarray, \
                                        param_B: Optional[float] = None, param_alpha: Optional[float] = None) -> np.ndarray:
    """
    Calculate the link speed between clients and relays.
    Args:
        client_location_list (np.ndarray): An array of client locations.
        relay_location_list (np.ndarray): An array of relay locations.
        relay_height_list (np.ndarray): An array of relay heights.
        param_B (Optional[float]): An optional parameter B. Default is None.
        param_alpha (Optional[float]): An optional parameter alpha. Default is None.
    Returns:
        np.ndarray: An array of calculated link speeds.
    """
    if param_B is None:
        param_B = globals()['param_B']
    if param_alpha is None:
        param_alpha = globals()['param_alpha']

    # init the link speed matrix
    link_speed_matrix = np.zeros(shape=(client_location_list.shape[0], relay_location_list.shape[0]), dtype=np.float32)

    # calculate the link speed between the i-th client and the j-th relay
    for i in range(client_location_list.shape[0]):
        if relay_height_list is not None:
            distance_list = np.linalg.norm(np.concatenate([relay_location_list - client_location_list[i], relay_height_list], axis=1, dtype=np.float32), axis=1)
        else:
            distance_list = np.linalg.norm(relay_location_list - client_location_list[i], axis=1)
        distance_list = np.where(distance_list < distance_min, distance_min, distance_list)
        link_speed_list = param_B * np.log2((1 + param_alpha / (distance_list ** 2)).astype(np.float32))
        link_speed_list = np.where(distance_list <= distance_min, rate_reference, link_speed_list)
        link_speed_list = np.where(distance_list >= distance_max, 0, link_speed_list)
        link_speed_matrix[i] = link_speed_list
    return link_speed_matrix

def calculate_link_speed_client_relay_with_model(client_location_list: np.ndarray, relay_location_list: np.ndarray, relay_height_list: np.ndarray, \
                                                    model: RegularGridInterpolator = None) -> np.ndarray:
    """
    Calculate the link speed between clients and relays using a given model.
    Args:
        client_location_list (np.ndarray): An array of client locations.
        relay_location_list (np.ndarray): An array of relay locations.
        relay_height_list (np.ndarray): An array of relay heights.
        model (RegularGridInterpolator): A pre-trained RegularGridInterpolator model to calculate the link speed.
    Returns:
        np.ndarray: An array of calculated link speeds.
    """
    if model is None:
        model = globals()['model_ground']

    # init the link speed matrix
    link_speed_matrix = np.zeros(shape=(client_location_list.shape[0], relay_location_list.shape[0]), dtype=np.float32)

    # calculate the link speed between the i-th client and the j-th relay
    for i in range(client_location_list.shape[0]):
        distance_list = np.linalg.norm(relay_location_list - client_location_list[i], axis=1)
        distance_list = np.where(distance_list > distance_max_ground, distance_max_ground, distance_list)
        points_list = np.concatenate([distance_list.reshape(-1, 1), relay_height_list], axis=1)
        link_speed_matrix[i] = model(points_list)

    return link_speed_matrix

def calculate_link_speed_center(location_list: np.ndarray, height_list: Optional[np.ndarray], center_location: np.ndarray,\
                                param_B: Optional[float] = None, param_alpha: Optional[float] = None) -> np.ndarray:
    """
    Calculates the link speed for each location in the location_list based on the center_location.

    Args:
        location_list (np.ndarray): Array of locations.
        center_location (np.ndarray): Center location.
        param_B (float): Parameter B. Bandwidth.
        param_alpha (float): Parameter alpha.

    Returns:
        np.ndarray: Array of link speeds.
    """
    if param_B is None:
        param_B = globals()['param_B']

    if param_alpha is None:
        param_alpha = globals()['param_alpha']

    if height_list is not None:
        assert location_list.shape[0] == height_list.shape[0], "The length of the location_list and height_list should be the same"
        distance_list = np.linalg.norm(np.concatenate([location_list - center_location, height_list], axis=1, dtype=np.float32), axis=1)
    else:
        distance_list = np.linalg.norm(location_list - center_location, axis=1)

    distance_list = np.where(distance_list < distance_min, distance_min, distance_list)
    link_speed_list = param_B * np.log2((1 + param_alpha / (distance_list ** 2)).astype(np.float32))
    link_speed_list = np.where(distance_list <= distance_min, rate_reference, link_speed_list)
    link_speed_list = np.where(distance_list >= distance_max, 0, link_speed_list)

    return link_speed_list

def calculate_link_speed_center_with_model(location_list: np.ndarray, height_list: Optional[np.ndarray], center_location: np.ndarray,\
                                           model: RegularGridInterpolator = None) -> np.ndarray:
    """
    Calculate the link speed for each location in the location_list based on the center_location using a given model.

    Args:
        location_list (np.ndarray): Array of locations.
        center_location (np.ndarray): Center location.
        model (RegularGridInterpolator): Model to calculate the link speed.

    Returns:
        np.ndarray: Array of link speeds.
    """
    if model is None:
        model = globals()['model_ground']
    if height_list is not None:
        assert location_list.shape[0] == height_list.shape[0], "The length of the location_list and height_list should be the same"
    else:
        height_list = np.zeros((location_list.shape[0], 1), dtype=np.float32)

    distance_list = np.linalg.norm(location_list - center_location, axis=1)
    distance_list = np.where(distance_list > distance_max_ground, distance_max_ground, distance_list)
    points_list = np.concatenate([distance_list.reshape(-1, 1), height_list], axis=1)
    link_speed_list = model(points_list)

    return link_speed_list

if __name__ == "__main__":
    # location_list = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.float32) * 50
    # height_list = np.array([[0], [10], [20], [30]], dtype=np.float32)
    # print(calculate_link_speed(location_list, height_list))
    # print(calculate_link_speed_center(location_list, height_list, np.array([[0, 0]], dtype=np.float32)))

    location_list = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.float32) * 1000
    height_list = np.array([[0], [1], [2], [3]], dtype=np.float32) * 100
    print(calculate_link_speed_with_model(location_list, height_list, model_air))