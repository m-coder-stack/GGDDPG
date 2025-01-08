import numpy as np
from treelib import Tree
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import List, Optional, Union


def scale_array(arr: np.ndarray, target_range: List[float], original_range: Optional[List[float]] = None) -> np.ndarray:
    """
    Scales the values of the input array to a target range.

    Args:
        arr (np.ndarray): The input array to be scaled.
        target_range (List[float]): The target range to scale the values to.
        original_range (Optional[List[float]], optional): The original range of the input array. If not provided, the minimum and maximum values of the array will be used. Defaults to None.

    Returns:
        np.ndarray: The scaled array.

    """
    if original_range is None:
        original_range = [np.min(arr), np.max(arr)]
    return (arr - original_range[0]) / (original_range[1] - original_range[0]) * (target_range[1] - target_range[0]) + target_range[0]


def draw_map(tree: Tree, center_position: np.ndarray, relay_position: np.ndarray, client_position: np.ndarray, \
             relay_height: Optional[np.ndarray] = None, relay_traffic: Optional[np.ndarray] = None, client_link: np.ndarray = None, \
             size: Optional[List[list[float]]] = None, is_show = True) -> Union[Image.Image, None]:
    """
    Draw a map with the given tree structure and positions of center, relay, and client nodes.

    Args:
        tree (Tree): The tree structure representing the connections between nodes.
        center_position (np.ndarray): The position of the center node.
        relay_position (np.ndarray): The positions of the relay nodes.
        client_position (np.ndarray): The positions of the client nodes.
        relay_height (Optional[np.ndarray], optional): The heights of the relay nodes. Defaults to None.
        relay_traffic (Optional[np.ndarray], optional): The traffic data for the relay nodes. Defaults to None.
        client_link (np.ndarray): The link data between the client and relay nodes. Defaults to None.
        size (Optional[List[list[float]]], optional): The size of the map. Defaults to None.
        is_show (bool, optional): Whether to display the map or save it as an image. Defaults to True.

    Returns:
        Union[Image.Image, None]: The map image if `is_show` is False, otherwise None.
    """
    
    if size:
        plt.xlim(size[0][0], size[0][1])
        x_ticks = np.linspace(size[0][0], size[0][1], 5)
        plt.xticks(x_ticks)
        
        plt.ylim(size[1][0], size[1][1])
        y_ticks = np.linspace(size[1][0], size[1][1], 5)
        plt.yticks(y_ticks)
        plt.axis('equal')

    # draw lines first
    for node_index in tree.expand_tree():
        parent = tree.parent(node_index)
        if parent:
            if parent.identifier == -1:
                # means the parent is the center
                node = tree.get_node(node_index)
                if node.data["link_speed"] > 0:
                    plt.plot([center_position[0,0], relay_position[node_index,0]], [center_position[0,1], relay_position[node_index,1]], linestyle="--", c='grey')
            else:
                # means the parent is a relay
                plt.plot([relay_position[parent.identifier,0], relay_position[node_index,0]], [relay_position[parent.identifier,1], relay_position[node_index,1]], linestyle="--", c='grey')

    # based on the client position draw the link between the client and the relay
    if client_link is not None:
        for i in range(client_position.shape[0]):
            client = client_position[i]
            relay_index = client_link[i]
            if relay_index != -1:
                plt.plot([relay_position[relay_index,0], client[0]], [relay_position[relay_index,1], client[1]], linestyle="--", c='#add8e6')


    # draw the markers now

    plt.scatter(center_position[:, 0], center_position[:, 1], c='r', marker='s', label='center')

    marker_size = None
    if relay_traffic is not None:
        marker_size = scale_array(relay_traffic, [10, 100])
        
    plt.scatter(relay_position[:, 0], relay_position[:, 1], c=relay_height, s=marker_size, marker='P', label='relay', cmap="coolwarm")
    plt.colorbar(label='height')
    # draw label for relay
    for i in range(relay_position.shape[0]):
        plt.text(relay_position[i, 0], relay_position[i, 1], str(i))

    plt.scatter(client_position[:, 0], client_position[:, 1], c='g', marker='o', label='client')
    
    if is_show:
        plt.show()
    else:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.clf()
        # move the cursor to the beginning of the buffer
        buffer.seek(0)
        image = Image.open(buffer)
        return image
    
    