import numpy as np
from treelib import Node, Tree
from visualization import draw_map
from loguru import logger
from typing import Optional, List


class Node:
    def __init__(self, node_id: int, data_amount: float, link_speed: float, parent: Optional[Node]=None):
        self.node_id: int = node_id
        self.parent = parent
        self.children: List[Node] = []
        # the data amout that this node  need to seed to the center
        self.data_amount = data_amount
        # the traffic load that this node need to seed to the center
        self.traffic_load = data_amount
        # the spped that this node can communicate with the parent
        self.link_speed = link_speed

    def add_child(self, child: Node):
        
        self.children.append(child)
        child.parent = self

        # deal with the traffic load
        # for the first child ,the additional traffic load is the same as the child's traffic load
        parent = self
        traffic_load = child.traffic_load
        # so compare the traffic load with the link speed, get the minimum value
        traffic_load = min(traffic_load, child.link_speed)
        # add the traffic load to the parent
        parent.traffic_load += traffic_load
        child = parent
        parent = parent.parent

        # for the rest of the children, the additional traffic load is the increment of the child's traffic load
        while parent:
            traffic_load = min(traffic_load, child.link_speed - child.traffic_load + traffic_load)
            parent.traffic_load += traffic_load
            child = parent
            parent = parent.parent
            

    def available_link_speed(self) -> float:
        """
        Calculates the available link speed based on the traffic load of the node and its ancestors.

        Returns:
            float: The available link speed.
        """
        node = self
        available_speed = float("inf")
        
        while node:
            if node.traffic_load >= node.link_speed:
                return 0.0
            else:
                available_speed = min(available_speed, node.link_speed - node.traffic_load)
                node = node.parent

        return available_speed
    
    def get_traffic_load(self) -> dict[int, float]:
        """
        Returns a dictionary containing the traffic load of the current node and its children.

        Returns:
            dict: A dictionary where the keys are node IDs and the values are the corresponding traffic loads.
        """
        traffic_load = {self.node_id: self.traffic_load}
        for child in self.children:
            traffic_load.update(child.get_traffic_load())
        return traffic_load
    
    def print_tree(self, level: int=0):
        print("---"*level + repr(self.node_id))
        for child in self.children:
            child.print_tree(level+1)

    def build_tree(self) -> Tree:
        tree = Tree()
        tree.create_node(self.node_id, self.node_id, data={"link_speed": self.link_speed, "traffic_load": self.traffic_load})
        for child in self.children:
            tree.paste(self.node_id, child.build_tree())
            
        return tree
    
    def check_tree(self):
        traffic_load = self.data_amount
        for child in self.children:
            traffic_load += child.traffic_load if child.traffic_load < child.link_speed else child.link_speed

        assert traffic_load == self.traffic_load, f"traffic load is not equal to the sum of the children's traffic load, {traffic_load} != {self.traffic_load}" + \
            f"node id: {self.node_id}, children id list: {[child.node_id for child in self.children]}"
        # for debug
        # if traffic_load != self.traffic_load:
        #     logger.warning(f"traffic load is not equal to the sum of the children's traffic load, {traffic_load} == {self.traffic_load} \n" +\
        #                 f"node id: {self.node_id}, children id list: {[child.node_id for child in self.children]}")
        
        for child in self.children:
            child.check_tree()
        


def build_tree(center_node_speed_list: np.ndarray, relay_node_speed_matrix: np.ndarray, \
               relay_node_data_amount: np.ndarray, relay_node_index: Optional[List[int]] = None) -> Node:
    """
    Builds a tree connecting a center node with relay nodes based on their speed and data amount.

    Args:
        center_node_speed_list (np.ndarray): An array of speed for the center node and relay nodes.
        relay_node_speed_matrix (np.ndarray): A matrix representing the speed between relay nodes.
        relay_node_data_amount (np.ndarray): An array of data amount for the relay nodes.
        relay_node_index (Optional[List[int]], None): An optional array of indices for the relay nodes. 
            If not provided, the relay nodes will be sorted based on their speed. Defaults to None.

    Returns:
        Node: The root node of the tree.

    """
    # not given the relay_node_index, sort the relay nodes based on the speed
    # the sort is on the distance, and the relation between the distance and the speed is inverse
    # the relay node with the highest speed will be the first node
    if relay_node_index is None:
        relay_node_index = np.argsort(center_node_speed_list)[::-1]
    elif type(relay_node_index) is np.ndarray:
        relay_node_index = relay_node_index.tolist()
    
    node_num = len(center_node_speed_list)
    # create the center node
    # assume the center node's data amount is 0 and the link speed is infinite
    center_node = Node(-1, 0, float("inf"))

    # create the relay nodes and store them in a dictionary for quick access
    nodes = {-1: center_node}
    # record the index of the nodes that have not been added to the tree
    left_nodes_index = list(range(node_num))

    # find the node that can be directly connected to the center node
    for i in range(node_num):
        if center_node_speed_list[i] >= relay_node_data_amount[i]:
            # create the node
            node = Node(i, relay_node_data_amount[i], center_node_speed_list[i])
            # join the index
            nodes[i] = node
            center_node.add_child(node)
            # remove the index from the left_nodes_index
            left_nodes_index.remove(i)
            relay_node_index.remove(i)
            
    

    # build the tree to connect the center node and the relay nodes
    while left_nodes_index:
        # find the nearest node
        current_node_index = relay_node_index[0]
        left_nodes_index.remove(current_node_index)
        relay_node_index.remove(current_node_index)

        # find the parent node
        # list all nodes that may be the parent node
        relay_nodes_index = list(nodes.keys())
        relay_nodes_index.remove(-1)
        relay_nodes_index = np.array(relay_nodes_index ,dtype=int)

        available_link_speed = center_node_speed_list[current_node_index]
        current_parent_index = -1

        condition = relay_node_speed_matrix[current_node_index, relay_nodes_index] > available_link_speed
        # nodes that may get better link speed
        relay_nodes_index = relay_nodes_index[np.where(condition)]
        if relay_nodes_index.size > 0:
            # find the node that can provide the best link speed
            # first make sure that the link speed is between the current node and the relay node is larger than the current available link speed
            available_link_speed_list = np.array([nodes[i].available_link_speed() for i in relay_nodes_index])
            available_link_speed_list  = np.minimum(available_link_speed_list, relay_node_speed_matrix[current_node_index, relay_nodes_index])
            if available_link_speed < available_link_speed_list.max():
                current_parent_index = relay_nodes_index[np.argmax(available_link_speed_list)] 
                available_link_speed = relay_node_speed_matrix[current_node_index, current_parent_index]
                
        # create the node
        # based on current_node_index, current_parent_index, available_link_speed        
        current_node = Node(current_node_index, relay_node_data_amount[current_node_index], available_link_speed)
        nodes[current_node_index] = current_node
        nodes[current_parent_index].add_child(current_node)

    return center_node


def calculate_reward_from_tree(center_node: Node) -> float:
    return sum_traffic(center_node=center_node)

def sum_traffic(center_node: Node) -> float:
    traffic = 0.0
    stack = [center_node]
    while stack:
        node = stack.pop()
        traffic += node.traffic_load
        # add all the children nodes to the stack
        stack.extend(node.children)
    return traffic




if __name__ == "__main__":
   
    # 示例数据
    center_node_position = np.array([0, 0])
    center_node_speed_list = np.array([3, 2, 1])
    relay_nodes_position = np.array([[1, 1], [2, 2], [3, 3]])
    relay_node_speed_matrix = np.array([
        [0, 3, 3],
        [3, 0, 3],
        [3, 3, 0]
        ])
    relay_node_data_amouts = np.array([1,1,2])

    # 构建树
    center_node = build_tree(center_node_speed_list, relay_node_speed_matrix, [1, 2, 3], [0, 1, 2])
    center_node.print_tree()