import numpy as np
    
import math

from node_functions import *
from cppn import *
from image_utils import *
from config import Config
from util import choose_random_function, visualize_network
import copy

def get_disjoint_connections(this_cxs, other_innovation):
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if (t_cx.innovation not in other_innovation and t_cx.innovation < other_innovation[-1])]


def get_excess_connections(this_cxs, other_innovation):
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if (t_cx.innovation not in other_innovation and t_cx.innovation > other_innovation[-1])]


def get_matching_connections(cxs_1, cxs_2):
    # returns connections in cxs_1 that share an innovation number with a connection in cxs_2
    # and     connections in cxs_2 that share an innovation number with a connection in cxs_1
    return sorted([c1 for c1 in cxs_1 if c1.innovation in [c2.innovation for c2 in cxs_2]], key=lambda x: x.innovation),\
        sorted([c2 for c2 in cxs_2 if c2.innovation in [
               c1.innovation for c1 in cxs_1]], key=lambda x: x.innovation)


class Individual:
    pixel_inputs = None
    def initialize(self):
        self.more_fit_parent = None  # for record-keeping
        self.fitness_fn = self.config.fitness_function
        self.n_hidden_nodes = self.config.hidden_nodes_at_start
        self.n_inputs = self.config.num_inputs
        self.n_outputs = self.config.num_outputs
        total_node_count = self.config.num_inputs + \
            self.config.num_outputs + self.config.hidden_nodes_at_start
        self.max_weight = self.config.max_weight
        self.weight_threshold = self.config.weight_threshold
        self.use_input_bias = self.config.use_input_bias
        self.use_radial_distance = self.config.use_radial_distance
        self.allow_recurrent = self.config.allow_recurrent

        for i in range(self.config.num_inputs):
            self.node_genome.append(
                Node(identity, NodeType.Input, self.get_new_node_id(), 0))
        for i in range(self.config.num_inputs, self.config.num_inputs + self.config.num_outputs):
            output_fn = choose_random_function(
                self.config) if self.config.output_activation is None else self.config.output_activation
            self.node_genome.append(
                Node(output_fn, NodeType.Output, self.get_new_node_id(), 2))
        for i in range(self.config.num_inputs + self.config.num_outputs, total_node_count):
            self.node_genome.append(Node(choose_random_function(
                self.config), NodeType.Hidden, self.get_new_node_id(), 1))

        # initialize connection genome
        if self.n_hidden_nodes == 0:
            # connect all input nodes to all output nodes
            for input_node in self.input_nodes():
                for output_node in self.output_nodes():
                    new_cx = Connection(
                            input_node, output_node, self.random_weight())
                    self.connection_genome.append(new_cx)
                    if(np.random.rand() > self.config.init_connection_probability):
                        new_cx.enabled = False
        else:
           # connect all input nodes to all hidden nodes
            for input_node in self.input_nodes():
                for hidden_node in self.hidden_nodes():
                    new_cx = Connection(
                        input_node, hidden_node, self.random_weight())
                    self.connection_genome.append(new_cx)
                    if(np.random.rand() > self.config.init_connection_probability):
                        new_cx.enabled = False
                        
           # connect all hidden nodes to all output nodes
            for hidden_node in self.hidden_nodes():
                for output_node in self.output_nodes():
                    if(np.random.rand() < self.config.init_connection_probability):
                        self.connection_genome.append(Connection(
                            hidden_node, output_node, self.random_weight()))
    
    def __init__(self, config=None) -> None:
        self.fitness = -math.inf
        self.novelty = -math.inf
        self.adjusted_fitness = -math.inf
        self.species_id = -1
        self.image = None
        self.node_genome = []  # inputs first, then outputs, then hidden
        self.connection_genome = []
        
        if config == None:
            # stub
            return 
        else:
            self.config = config
            self.initialize()
       

    def to_json(self):
        return {"node_genome": [n.to_json() for n in self.node_genome], "connection_genome": [c.to_json() for c in self.connection_genome], "species_id": self.species_id, "fitness": self.fitness, "adjusted_fitness": self.adjusted_fitness}

    def from_json(self, json_dict):
        for k, v in json_dict.items():
            self.__dict__[k] = v
        for i, cx in enumerate(self.connection_genome):
            self.connection_genome[i] = Connection.CreateFromJson(cx)
        for i, n in enumerate(self.node_genome):
            self.node_genome[i] = Node.CreateFromJson(n)

        for cx in self.connection_genome:
            cx.fromNode = self.node_genome[cx.fromNode.id]
            cx.toNode = self.node_genome[cx.toNode.id]

        self.update_node_layers()

    def CreateFromJson(json_dict, config):
        i = Individual(config)
        i.from_json(json_dict)
        return i

    def random_weight(self):
        return np.random.uniform(-self.max_weight, self.max_weight)

    def get_new_node_id(self):
        new_id = 0
        while len(self.node_genome) > 0 and new_id in [node.id for node in self.node_genome]:
            new_id += 1
        return new_id

    def update_with_fitness(self, fit, num_in_species):
        self.fitness = fit
        if(num_in_species > 0):
            self.adjusted_fitness = self.fitness / num_in_species  # local competition
            try:
                assert self.adjusted_fitness > - \
                    math.inf, f"adjusted fitness was -inf: fit: {self.fitness} n_in_species: {num_in_species}"
                assert self.adjusted_fitness < math.inf, f"adjusted fitness was -inf: fit: {self.fitness} n_in_species: {num_in_species}"
            except AssertionError as e:
                print(e)

        # weighted average with graph clustering coefficient
        if(self.config.clustering_fitness_ratio>0):
            self.fitness = self.fitness * (1-self.config.clustering_fitness_ratio) + self.config.clustering_fitness_ratio * clustering_coefficient(self)
            
    def eval_fitness(self, config, num_in_species=0):
        self.config = config
        # TODO change to only setting parameters in if statements, making one call at the end
        
        if(isinstance(config.train_image, str)):  # for classification
            self.fitness = self.fitness_fn(self.get_image(
                config.classification_image_size[0], config.classification_image_size[1], self.config.color_mode), self.config.train_image)
        else:
            self.fitness = self.fitness_fn(self.get_image(
                self.config.train_image.shape[0], self.config.train_image.shape[1], self.config.color_mode), self.config.train_image)
        
        self.update_with_fitness(self.fitness)
            
        return self.fitness


    def eval_image_novelty_ae(self, archive, k, ae):
        if(ae.ready): self.novelty = ae.eval_image_novelty(self.image)
        else: self.novelty = 1e-10
        # self.fitness = self.novelty

    def eval_image_novelty(self, archive, k):
        """Find the average Euclidean distance from this individual's image to the nearest k neighbors in the archive."""
        # Note, this could be the minimum distance if k=1
        self.novelty = 0
        self.get_image(
            self.image.shape[0], self.image.shape[1], self.config.color_mode)
        distances = [diff_feature_set(self.image, solution.get_image(
            self.image.shape[0], self.image.shape[1], self.config.color_mode)) for solution in archive]
        # distances = [avg_pixel_distance(self.image, solution.get_image(
        #     self.config.train_image.shape[0], self.config.train_image.shape[1], self.config.color_mode)) for solution in archive]
        distances.sort()  # shortest first
        closest_k = distances[0:k]
        average = np.average(closest_k, axis=0)
        if(average != average):
            average = 0
        self.novelty = average
        return average

    def eval_genetic_novelty(self, archive, k):
        """Find the average distance from this individual's genome to k nearest neighbors."""
        self.novelty = 0
        distances = [self.genetic_difference(solution) for solution in archive]
        distances.sort()  # shortest first
        closest_k = distances[0:k]
        average = np.average(closest_k, axis=0)
        if(average != average):
            average = 0
        self.novelty = average
        return average

    def enabled_connections(self):
        for c in self.connection_genome:
            if c.enabled:
                yield c

    def mutate_activations(self, prob_mutate_activation):
        eligible_nodes = list(self.hidden_nodes())
        if(self.config.output_activation is None):
             eligible_nodes.extend(self.output_nodes())
        if self.config.allow_input_activation_mutation:
             eligible_nodes.extend(self.input_nodes())
        for node in eligible_nodes:
            if(np.random.uniform(0,1) < prob_mutate_activation):
                node.fn = choose_random_function(self.config)
            
    def mutate_weights(self, weight_mutation_max, weight_mutation_probability):
        #         each connection weight is perturbed with a fixed probability by
        # adding a floating point number chosen from a uniform distribution of positive and negative values
        for cx in self.connection_genome:
            if(np.random.uniform(0, 1) < weight_mutation_probability):
                cx.weight += np.random.uniform(-weight_mutation_max,
                                               weight_mutation_max)
            elif(np.random.uniform(0, 1) < self.config.prob_weight_reinit):
                cx.weight = self.random_weight()

        self.clamp_weights()  # TODO NOT SURE

    def mutate_random_weight(self, amount):
        try:
            # TODO only enabled or all connections?
            cx = np.random.choice(self.connection_genome, 1)[
                0]  # choose one random connection
            cx.weight += amount
        except Exception as e:  # TODO no
            print(f"ERROR in mutation: {e}")

    def add_connection(self, chance_to_reenable, allow_recurrent):
        for i in range(20):  # try 20 times
            [fromNode, toNode] = np.random.choice(
                self.node_genome, 2, replace=False)
            existing_cx = None
            for cx in self.connection_genome:
                if cx.fromNode.uuid == fromNode.uuid and cx.toNode.uuid == toNode.uuid:
                    existing_cx = cx
                    break
            if(existing_cx != None):
                if(not existing_cx.enabled and np.random.rand() < chance_to_reenable):
                    existing_cx.enabled = True     # re-enable the connection
                break  # don't allow duplicates

            if(fromNode.layer == toNode.layer):
                continue  # don't allow two nodes on the same layer to connect

            is_recurrent = fromNode.layer > toNode.layer
            if(not allow_recurrent and is_recurrent):
                continue  # invalid
            # if(allow_recurrent and fromNode.layer == self.count_layers()-1): continue # invalid (recurrent from output) TODO unsure
            # valid connection, add
            new_cx = Connection(fromNode, toNode, self.random_weight())
            self.connection_genome.append(new_cx)
            self.update_node_layers()
            break

        # failed to find a valid connection, don't add

    def disable_invalid_connections(self):
        to_remove = []
        for cx in self.connection_genome:
            if(cx.fromNode == cx.toNode):
                raise Exception("Nodes should not be self-recurrent")
            if(cx.toNode.layer == cx.fromNode.layer):
                to_remove.append(cx)
            cx.is_recurrent = cx.fromNode.layer > cx.toNode.layer
            if(not self.allow_recurrent and cx.is_recurrent):
                to_remove.append(cx)  # invalid TODO consider disabling instead

        for cx in to_remove:
            self.connection_genome.remove(cx)

    def add_node(self):
        eligible_cxs = [
            cx for cx in self.connection_genome if not cx.is_recurrent]
        if(len(eligible_cxs) < 1):
            return
        old = np.random.choice(eligible_cxs)
        new_node = Node(choose_random_function(self.config),
                        NodeType.Hidden, self.get_new_node_id())
        self.node_genome.append(new_node)  # add a new node between two nodes
        old.enabled = False  # disable old connection

        # The connection between the first node in the chain and the new node is given a weight of one
        # and the connection between the new node and the last node in the chain is given the same weight as the connection being split

        self.connection_genome.append(Connection(
            self.node_genome[old.fromNode.id], self.node_genome[new_node.id],   self.random_weight()))

        # TODO shouldn't be necessary
        self.connection_genome[-1].fromNode = self.node_genome[old.fromNode.id]
        self.connection_genome[-1].toNode = self.node_genome[new_node.id]
        self.connection_genome.append(Connection(
            self.node_genome[new_node.id],     self.node_genome[old.toNode.id], old.weight))

        self.connection_genome[-1].fromNode = self.node_genome[new_node.id]
        self.connection_genome[-1].toNode = self.node_genome[old.toNode.id]

        self.update_node_layers()
        # self.disable_invalid_connections() # TODO broken af

    def remove_node(self):
        # This is a bit of a buggy mess
        hidden = self.hidden_nodes()
        if(len(hidden) < 1):
            return
        node_id_to_remove = np.random.choice([n.id for n in hidden], 1)[0]
        for cx in self.connection_genome[::-1]:
            if(cx.fromNode.id == node_id_to_remove or cx.toNode.id == node_id_to_remove):
                self.connection_genome.remove(cx)
        for node in self.node_genome[::-1]:
            if node.id == node_id_to_remove:
                self.node_genome.remove(node)
                break

        for i, node in enumerate(self.node_genome):
            node.id = i  # TODO not sure
        for i, cx in enumerate(self.connection_genome):
            cx.innovation = Connection.get_innovation(
                cx.fromNode, cx.toNode)  # TODO FIXME definitely wrong
        self.update_node_layers()
        self.disable_invalid_connections()

    def disable_connection(self):
        eligible_cxs = list(self.enabled_connections())
        if(len(eligible_cxs) < 1):
            return
        cx = np.random.choice(eligible_cxs)
        cx.enabled = False

    def update_node_layers(self) -> int:
        # layer = number of edges in longest path between this node and input
        def get_node_to_input_len(current_node, current_path=0, longest_path=0, attempts=0):
            if(attempts > 1000):
                print("ERROR: infinite recursion while updating node layers")
                return longest_path
            # use recursion to find longest path
            if(current_node.type == NodeType.Input):
                return current_path
            all_inputs = [
                cx for cx in self.connection_genome if not cx.is_recurrent and cx.toNode.id == current_node.id]
            for inp_cx in all_inputs:
                this_len = get_node_to_input_len(
                    inp_cx.fromNode, current_path+1, attempts+1)
                if(this_len >= longest_path):
                    longest_path = this_len
            return longest_path

        highest_hidden_layer = 1
        for node in self.hidden_nodes():
            node.layer = get_node_to_input_len(node)
            highest_hidden_layer = max(node.layer, highest_hidden_layer)

        for node in self.output_nodes():
            node.layer = highest_hidden_layer+1

    def genetic_difference(self, other) -> float:
        # only enabled connections, sorted by innovation id
        this_cxs = sorted(self.enabled_connections(),
                          key=lambda c: c.innovation)
        other_cxs = sorted(other.enabled_connections(),
                           key=lambda c: c.innovation)

        N = max(len(this_cxs), len(other_cxs))
        other_innovation = [c.innovation for c in other_cxs]

        # number of excess connections
        n_excess = len(get_excess_connections(this_cxs, other_innovation))
        # number of disjoint connections
        n_disjoint = len(get_disjoint_connections(this_cxs, other_innovation))

        # matching connections
        this_matching, other_matching = get_matching_connections(
            this_cxs, other_cxs)
        difference_of_matching_weights = [
            abs(o_cx.weight-t_cx.weight) for o_cx, t_cx in zip(other_matching, this_matching)]
        if(len(difference_of_matching_weights) == 0):
            difference_of_matching_weights = 0
        difference_of_matching_weights = np.mean(
            difference_of_matching_weights)

        # Furthermore, the compatibility distance function
        # includes an additional argument that counts how many
        # activation functions differ between the two individuals
        n_different_fns = 0
        for t_node, o_node in zip(self.node_genome, other.node_genome):
            if(t_node.fn.__name__ != o_node.fn.__name__):
                n_different_fns += 1

        # can normalize by size of network (from Ken's paper)
        if(N > 0):
            n_excess /= N
            n_disjoint /= N

        # weight (values from Ken)
        n_excess *= 1
        n_disjoint *= 1
        difference_of_matching_weights *= .4
        n_different_fns *= 1
        difference = n_excess + n_disjoint + \
            difference_of_matching_weights + n_different_fns

        return difference

    def species_comparision(self, other, threshold) -> bool:
        # returns whether other is the same species as self
        return self.genetic_difference(other) <= threshold  # TODO equal to?

    def input_nodes(self) -> list:
        return self.node_genome[0:self.n_inputs]

    def output_nodes(self) -> list:
        return self.node_genome[self.n_inputs:self.n_inputs+self.n_outputs]

    def hidden_nodes(self) -> list:
        return self.node_genome[self.n_inputs+self.n_outputs:]

    # def __repr__(self):
    #     string = "Nodes: "
    #     for n in self.node_genome:
    #         string += f"{n.fn.__name__}, "
    #     string += "connections: "
    #     for c in self.connection_genome:
    #         string += f"({c.fromNode.fn.__name__}->{c.toNode.fn.__name__} | I:{c.innovation} W:{c.weight}), "
    #     return string

    # def __str__(self):
    #     return self.__repr__()

    def set_inputs(self, inputs):
        # inputs = inputs[:2]  # clear any extra inputs TODO
        if(self.use_radial_distance):
            # d = sqrt(x^2 + y^2)
            inputs.append(math.sqrt(inputs[0]**2 + inputs[1]**2))
        if(self.use_input_bias):
            inputs.append(1.0)  # bias = 1.0

        for i, inp in enumerate(inputs):
            # inputs are first N nodes
            self.node_genome[i].sum_input = inp
            self.node_genome[i].output = self.node_genome[i].fn(inp)

    def get_layer(self, layer_index):
        for node in self.node_genome:
            if node.layer == layer_index:
                yield node

    def get_hidden_and_output_layers(self):
        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer
        return [self.get_layer(i) for i in range(1, output_layer+1)]

    def count_layers(self):
        return len(np.unique([node.layer for node in self.node_genome]))

    def clamp_weights(self):
        for cx in self.connection_genome:
            if(cx.weight < self.weight_threshold and cx.weight > -self.weight_threshold):
                cx.weight = 0
            if(cx.weight > self.max_weight):
                cx.weight = self.max_weight
            if(cx.weight < -self.max_weight):
                cx.weight = -self.max_weight

    def eval(self, inputs):
        self.set_inputs(inputs)
        return self.feed_forward()

    def feed_forward(self):
        if self.allow_recurrent:
            for node in self.get_layer(0):  # input nodes (handle recurrent)
                for node_input in list(filter(lambda x: x.toNode.id == node.id, self.enabled_connections())):
                    node.sum_input += node_input.fromNode.output * node_input.weight
                node.output = node.fn(node.sum_input)

        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer

        for layer_index in range(1, output_layer+1):
            # hidden and output layers:
            layer = self.get_layer(layer_index)
            for node in layer:
                node.sum_input = 0
                node.output = 0
                node_inputs = list(
                    filter(lambda x: x.toNode.id == node.id, self.enabled_connections()))  # cxs that end here
                for cx in node_inputs:
                    node.sum_input += cx.fromNode.output * cx.weight

                if(np.isnan(node.sum_input) or np.isinf(abs(node.sum_input))):
                    print(f"input was {node.sum_input}")
                    node.sum_input = 0  # TODO why?

                node.output = node.fn(node.sum_input)  # apply activation
                node.output = np.clip(node.output, -1, 1)  # TODO not sure (SLOW)

        return [node.output for node in self.output_nodes()]

    def get_image_data(self, res_x, res_y, color_mode):
        pixels = []
        # for x in np.linspace(-1, 1.0, res_x):
        #     for y in np.linspace(-1, 1.0, res_y):
        for x in np.linspace(-.5, .5, res_x):
            for y in np.linspace(-.5, .5, res_y):
                outputs = self.eval([x, y])
                pixels.extend(outputs)
        if(color_mode == 'RGB' or color_mode == "HSL"):
            pixels = np.reshape(pixels, (res_x, res_y, self.n_outputs))
        else:
            pixels = np.reshape(pixels, (res_x, res_y))

        self.image = pixels
        return pixels

    def get_image(self, res_x, res_y, color_mode, force_recalculate=False):
        if(not force_recalculate and self.image is not None and res_x == self.image.shape[0] and res_y == self.image.shape[1]):
            return self.image
        if self.allow_recurrent:
            self.image = self.get_image_data(res_x, res_y, color_mode) # pixel by pixel (good for debugging)
        else:
            self.image = self.get_image_data_fast_method(res_x, res_y, color_mode) # whole image at once (100s of times faster)
        return self.image

    def get_image_data_fast_method(self, res_h, res_w, color_mode):
        if self.allow_recurrent:
            raise Exception("Fast method doesn't work with recurrent yet")

        if Individual.pixel_inputs is None or Individual.pixel_inputs.shape[0] != res_h or Individual.pixel_inputs.shape[1]!=res_w:
            # lazy init:
            x_vals = np.linspace(-.5, .5, res_w)
            y_vals = np.linspace(-.5, .5, res_h)
            Individual.pixel_inputs = np.zeros((res_h, res_w, self.config.num_inputs), dtype=np.float32)
            for y in range(res_h):
                for x in range(res_w):
                    this_pixel = [y_vals[y], x_vals[x]] # coordinates
                    if(self.use_radial_distance):
                        # d = sqrt(x^2 + y^2)
                        this_pixel.append(math.sqrt(y_vals[y]**2 + x_vals[x]**2))
                    if(self.use_input_bias):
                        this_pixel.append(1.0)# bias = 1.0
                    Individual.pixel_inputs[y][x] = this_pixel
                    
        for i in range(len(self.node_genome)):
            # initialize outputs to 0:
            self.node_genome[i].outputs = np.zeros((res_h, res_w))
            
        for i in range(self.config.num_inputs):
            # inputs are first N nodes
            self.node_genome[i].sum_inputs = Individual.pixel_inputs[:,:, i]
            self.node_genome[i].outputs = self.node_genome[i].fn( Individual.pixel_inputs[:,:, i])

       
        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer

        for layer_index in range(1, output_layer+1):
            # hidden and output layers:
            layer = self.get_layer(layer_index)
            for node in layer:
                node_inputs = list(
                    filter(lambda x: x.toNode.id == node.id, self.enabled_connections()))  # cxs that end here
                
                node.sum_inputs = np.zeros((res_h, res_w), dtype=np.float32)
                for cx in node_inputs:
                    if(not hasattr(cx.fromNode, "outputs")):
                        print(cx.fromNode.type)
                        print(list(self.enabled_connections()))
                        print(cx.fromNode)
                        print(self.node_genome)
                    inputs = cx.fromNode.outputs * cx.weight
                    node.sum_inputs = node.sum_inputs + inputs

                if(np.isnan(node.sum_inputs).any() or np.isinf(np.abs(node.sum_inputs)).any()):
                    print(f"inputs was {node.sum_inputs}")
                    node.sum_inputs = np.zeros((res_h, res_w), dtype=np.float32)  # TODO why?
                    node.outputs = node.sum_inputs # ignore node
                    
                node.outputs = node.fn(node.sum_inputs)  # apply activation
                node.outputs = node.outputs.reshape((res_h, res_w)) 
                node.outputs = np.clip(node.outputs, -1, 1)  # TODO not sure (SLOW)

        outputs = [node.outputs for node in self.output_nodes()]
        if(color_mode == 'RGB' or color_mode == "HSL"):
            outputs =  np.array(outputs).transpose(1, 2, 0) # move color axis to end
        else:
            outputs = np.reshape(outputs, (res_h, res_w))
        self.image = outputs
        return outputs
    
    def reset_activations(self):
        for node in self.node_genome:
            node.outputs = np.zeros((self.config.train_image.shape[0], self.config.train_image.shape[1]))
            node.sum_inputs = np.zeros((self.config.train_image.shape[0], self.config.train_image.shape[1]))
            
    def save(self, filename):
        json_nodes = [(node.fn.__name__, node.type) for node in self.node_genome]
        json_cxs = [(self.node_genome.index(cx.fromNode), self.node_genome.index(cx.toNode), cx.weight, cx.enabled) for cx in self.connection_genome]
        print(json_cxs)
        json_config = json.loads(self.config.to_json())
        with open(filename, 'w') as f:
            json.dump({'nodes': json_nodes, 'cxs': json_cxs, "config":json_config},f)
            f.close()
            
        self.config.from_json(json_config)
    
    def construct_from_lists(self, nodes, connections):
        self.node_genome = [Node(name_to_fn(n[0]), NodeType(n[1]), i) for i, n in enumerate(nodes)]
        self.connection_genome = [Connection(self.node_genome[c[0]], self.node_genome[c[1]], c[2], c[3]) for c in connections]
        self.update_node_layers()
        # self.disable_invalid_connections()
        
    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            nodes = data['nodes']
            connections = data['cxs']
            self.config = Config.CreateFromJson(data['config'])
            self.config.from_json(data['config'])
            self.initialize()
            f.close()
        self.construct_from_lists(nodes, connections)  
        
    def load(filename):
        individual = Individual(None) 
        individual.load_from_file(filename)        
        return individual
        
def crossover_simple(parent1, parent2):
    [fit_parent, less_fit_parent] = sorted([parent1, parent2], key=lambda x: x.fitness, reverse=True)
    # child = copy.deepcopy(fit_parent) 
    child = Individual(fit_parent.config)
    child.species_id = fit_parent.species_id
    # disjoint/excess genes are inherited from more fit parent
    child.node_genome = copy.deepcopy(fit_parent.node_genome)
    child.connection_genome = copy.deepcopy(fit_parent.connection_genome)

    # child.more_fit_parent = fit_parent # TODO
    
    child.connection_genome.sort(key=lambda x: x.innovation)
    matching1, matching2 = get_matching_connections(fit_parent.connection_genome, less_fit_parent.connection_genome)
    for match_index in range(len(matching1)):
        # Matching genes are inherited randomly
        child_cx =  child.connection_genome[[x.innovation for x in child.connection_genome].index(matching1[match_index].innovation)]
        child_cx.weight = \
            matching1[match_index].weight if np.random.rand()< .5 else matching2[match_index].weight

        new_from =  copy.deepcopy(matching1[match_index].fromNode if np.random.rand() < .5 else matching2[match_index].fromNode)
        child_cx.fromNode = new_from
        child.node_genome[new_from.id] = new_from
        
        new_to =  copy.deepcopy(matching1[match_index].toNode if np.random.rand() < .5 else matching2[match_index].toNode)
        child_cx.toNode = new_to
        child.node_genome[new_to.id] = new_to
            
        if(not matching1[match_index].enabled or not matching2[match_index].enabled): 
            if(np.random.rand()< 0.75): # from Stanley/Miikulainen 2007
                child.connection_genome[match_index].enabled = False   


    
    # for node_index in range(len(child.node_genome)):
    #     child.node_genome[node_index] =\
    #          copy.deepcopy(parent1.node_genome[node_index] if np.random.rand() < .5 else parent2.node_genome[node_index])

    for cx in child.connection_genome:
        # TODO this shouldn't be necessary
        cx.fromNode = child.node_genome[cx.fromNode.id]
        cx.toNode =   child.node_genome[cx.toNode.id]
    child.update_node_layers()
    child.disable_invalid_connections()
    return child

def crossover(parent1, parent2,):
    [fit_parent, less_fit_parent] = sorted([parent1, parent2], key=lambda x: x.fitness, reverse=True)
    child = Individual(fit_parent.config)
    child.species_id = fit_parent.species_id
    # child.more_fit_parent = fit_parent # TODO
    matching1, matching2 = get_matching_connections(fit_parent.connection_genome, less_fit_parent.connection_genome)
    matching1, matching2 = copy.deepcopy(matching1), copy.deepcopy(matching2) # make copies

    # assert len(matching1) == len(matching2), "genome lengths don't match"

    less_fit_innovations = [c.innovation for c in less_fit_parent.connection_genome]
    more_fit_excess = get_excess_connections(fit_parent.connection_genome, less_fit_innovations)
    more_fit_disjoint = get_disjoint_connections(fit_parent.connection_genome, less_fit_innovations)
    more_fit_excess, more_fit_disjoint = copy.deepcopy(more_fit_excess), copy.deepcopy(more_fit_disjoint) # make copies

    len_new_cx_genome = len(matching1) + len(more_fit_disjoint) + len(more_fit_excess)
    child.connection_genome = [None] * len_new_cx_genome
    
    # Matching genes are inherited randomly
    for match_index in range(len(matching1)):
        try:
            to_add = None
            if(np.random.rand()< .5):
                to_add = matching1[match_index]     
            else: 
                to_add = matching2[match_index]
                
            child.connection_genome[match_index] = to_add
            # chance to disable connection if disabled in either parent
            if(not matching1[match_index].enabled or not matching2[match_index].enabled): 
                if(np.random.rand()< 0.75): # from Stanley/Miikulainen 2007
                    child.connection_genome[match_index].enabled = False
        except IndexError:
            # TODO FIXME
            print("indexerror in crossover")
            return copy.deepcopy(fit_parent) 


    # disjoint genes (those that do not match in the middle) 
    # and excess genes (those that do not match in the end) 
    # are inherited from the more fit parent.
    for disjoint_i, disjoint_cx in enumerate(more_fit_disjoint):
        child.connection_genome[len(matching1) + disjoint_i] = disjoint_cx
    for i, excess_cx in enumerate(more_fit_excess):
        child.connection_genome[len(matching1) + len(more_fit_disjoint) + i] = excess_cx

    # child.connection_genome.sort(key=lambda x: x.innovation) # sort new connections by innovation
    # child.node_genome = [None] * (len(set([cx.fromNode.id for cx in child.connection_genome]).union(set([cx.toNode.id for cx in child.connection_genome]))) )
    child.node_genome = []
    
    for i, cx in enumerate(child.connection_genome):
        ids = [n.id for n in child.node_genome]
        if cx.fromNode.id not in ids:
            child.node_genome.append(cx.fromNode)
        if cx.toNode.id not in ids:
            child.node_genome.append(cx.toNode)
    
    assert len(child.node_genome) == len(set([cx.fromNode.id for cx in child.connection_genome]).union(set([cx.toNode.id for cx in child.connection_genome]))), "wrong number of genes"

    child.node_genome.sort(key=lambda x: x.id)    # sort by id
    
    # assert(None not in child.node_genome, "a connection was None during crossover")

    for cx in child.connection_genome:
        try:
            # TODO this shouldn't be necessary
            if cx.fromNode.id >= len(child.node_genome):
                child.node_genome.append(cx.fromNode)
            if cx.toNode.id >= len(child.node_genome):
                child.node_genome.append(cx.toNode)
            cx.fromNode = child.node_genome[cx.fromNode.id]
            cx.toNode =   child.node_genome[cx.toNode.id]
        except IndexError:
            # continue
            return copy.deepcopy(fit_parent) 

    
    child.update_node_layers()
    child.disable_invalid_connections()
    return child



if __name__ == "__main__":
    from experiment import Config
    import time
    c = Config()
    imgs_fast = []
    imgs_slow = []
    inds = []
    num = 10000
    size = 32
    
    
    # c.color_mode = "RGB"
    c.color_mode = "L"
    c.num_outputs = len(c.color_mode)
    
    for i in range(num):
        ind = Individual(c)
        inds.append(ind)
    
    start_slow = time.time()
    for i in range(num):
        img = inds[i].get_image_data(size,size, c.color_mode)
        imgs_slow.append(img)
    end_slow = time.time() 
    
    start_fast = time.time()
    for i in range(num):
        img = inds[i].get_image_data_fast_method(size, size, c.color_mode)
        imgs_fast.append(img)
    end_fast = time.time() 

    
    all_images = imgs_fast + imgs_slow
    print("fast took:", end_fast - start_fast)
    print("slow took:", end_slow - start_slow)
    
    all = num * size * size * len(c.color_mode)
    
    plt.style.use("seaborn")
    print(f"{100*np.count_nonzero(np.isclose(imgs_fast, imgs_slow)) / all:.3f}% correct")
    # show_images(all_images, c.color_mode, [f"True" if np.isclose(imgs_fast[i%num], imgs_slow[i%num]).all() else "False" for i in range(num*2) ])
    plt.show()