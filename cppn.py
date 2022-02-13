from enum import Enum, IntEnum
from node_functions import *
from util import name_to_fn
import uuid
import json
import networkx as nx

class NodeType(IntEnum):
    Input  = 0
    Output = 1
    Hidden = 2


class Node:
    def __init__(self, fn, _type, _id, _layer=2) -> None:
        self.fn = fn
        self.uuid = uuid.uuid1()
        self.id = _id
        self.type = _type
        self.layer = _layer
        self.sum_inputs = np.zeros(1)
        self.outputs = np.zeros(1)
        self.sum_input = 0
        self.output = 0

    def to_json(self):
        self.type = int(self.type)
        self.id = int(self.id)
        self.layer = int(self.id)
        self.uuid = ""
        self.sum_input = float(self.sum_input)
        self.output = float(self.output)
        try:
            self.fn = self.fn.__name__
        except AttributeError:
            pass
        return json.dumps(self.__dict__, sort_keys=True, indent=4)

    def from_json(self, json_dict):
        if(isinstance(json_dict, str)):
            json_dict = json.loads(json_dict)
        self.__dict__ = json_dict 
        self.type = NodeType(self.type)
        self.fn = name_to_fn(self.fn)
        return self
           
    def CreateFromJson(json_dict):
        i = Node(None, None, None, None)
        i = i.from_json(json_dict)
        return i

    def empty():
        return Node(identity, NodeType.Hidden, 0, 0)

    # def __repr__(self) -> str:
    #     if(self.output == None):
    #         return f"{self.id}: ({self.fn.__name__}->{None})"
    #     return f"{self.id}: ({self.fn.__name__}->{self.output:4f})"
    # def __repr__(self) -> str:
    #     return f"{self.id}"



class Connection:
    # connection            e.g.  2->5,  1->4
    # innovation_number            0      1
    # where innovation number is the same for all of same connection
    # i.e. 2->5 and 2->5 have same innovation number, regardless of individual
    innovations = []

    def get_innovation(fromNode, toNode):
        cx = (fromNode.id, toNode.id) # based on id
        # cx = (fromNode.fn.__name__, toNode.fn.__name__) # based on fn
        if(cx in Connection.innovations):
            return Connection.innovations.index(cx)
        else:
            Connection.innovations.append(cx)
            return len(Connection.innovations) - 1
  
    def __init__(self, fromNode, toNode, weight, enabled = True) -> None:
        self.fromNode = fromNode # TODO change to node ids?
        self.toNode = toNode
        self.weight = weight
        self.innovation = Connection.get_innovation(fromNode, toNode)
        self.enabled = enabled
        self.is_recurrent = toNode.layer < fromNode.layer
        
    def to_json(self):
        self.innovation = int(self.innovation)
        self.fromNode = self.fromNode.to_json()
        self.toNode = self.toNode.to_json()
        return json.dumps(self.__dict__, sort_keys=True, indent=4)

    def from_json(self, json_dict):
        if(isinstance(json_dict, str)):
            json_dict = json.loads(json_dict)
        self.__dict__ = json_dict
        self.fromNode = Node.CreateFromJson(self.fromNode)
        self.toNode = Node.CreateFromJson(self.toNode)
        return self
     
    def CreateFromJson(json_dict):
        f_node = Node.empty()
        t_node = Node.empty()
        i = Connection(f_node, t_node, 0)
        i.from_json(json_dict)
        return i
        
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"([{self.fromNode.id}->{self.toNode.id}]I:{self.innovation} W:{self.weight:3f})"
        # return f"{self.fromNode.fn.__name__}->{self.toNode.fn.__name__} (I:{self.innovation}, W:{self.weight:.2f})"



def clustering_coefficient(individual):
    connections = individual.connection_genome
    G = nx.DiGraph()
    for i, node in enumerate(individual.input_nodes()):
        G.add_node(node, layer=(node.layer))
    for node in individual.hidden_nodes():
        G.add_node(node, layer=(node.layer))
    for i, node in enumerate(individual.output_nodes()):
        G.add_node(node, layer=(node.layer))
    for cx in connections:
        if not cx.enabled: continue 
        G.add_edge(cx.fromNode, cx.toNode)
    
    clustering = nx.algorithms.cluster.average_clustering(G)
    return clustering