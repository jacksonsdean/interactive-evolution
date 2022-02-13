import json

import numpy as np
    
from image_utils import *
from node_functions import *
from util import *
# from .individual import *
from skimage.transform import resize 
from skimage.filters import gaussian, median

class Config:
    def __init__(self) -> None:
        self.fitness_function=None
        self.total_generations=100
        self.color_mode = "L"
        self.num_parents=10
        self.num_children=10
        self.do_crossover=True
        self.use_speciation = True
        self.use_map_elites = False
        self.allow_recurrent = False
        self.tournament_size=4
        self.tournament_winners=2
        self.species_selection_ratio=.5
        self.species_target = 6
        self.species_threshold_delta = .25
        self.init_species_threshold = 2.5
        self.max_weight = 3.0
        self.weight_threshold = 0
        self.weight_mutation_max = 2
        self.prob_random_restart =.001
        self.prob_weight_reinit = 0.0
        self.prob_reenable_connection = 0.1
        self.species_stagnation_threshold = 20
        self.fitness_threshold = 1
        self.init_connection_probability = 1
        self.within_species_elitism = 1
        self.population_elitism = 1
        self.train_image = np.zeros(1)
        self.train_image_path = "train/half/bw_half_10.png"
        self.activations = all_node_functions
        self.novelty_selection_ratio_within_species = 0.0
        self.novelty_adjusted_fitness_proportion = 0.0
        self.novelty_k = 5
        self.novelty_archive_len = 5
        self.curriculum = []
        self.auto_curriculum = 0
        self.num_workers = 4 
        self.resize_train = None
        # from Secretan et al 2008
        # DGNA: probability of adding a node is 0.5 and the probability of adding a connection is 0.4. 
        # SGNA: probability of adding a node is 0.05 and the probability of adding a connection is 0.04. 
        self.prob_mutate_activation = .1 
        self.prob_mutate_weight = .35
        self.prob_add_connection = .1 
        self.prob_add_node = .1
        self.prob_remove_node = 0.0
        self.prob_disable_connection = .3
        
        self.use_dynamic_mutation_rates = False
        self.dynamic_mutation_rate_end_modifier = .1
        self.use_multithreading = False
        self.output_activation = None
        self.save_progress_images = False
        # DGNA/SGMA uses 1 or 2 so that patterns in the initial generation would be nontrivial (Stanley, 2007).
        self.hidden_nodes_at_start=0

        self.allow_input_activation_mutation = False
        
        self.animate = False

        self.use_input_bias = True # SNGA, https://link.springer.com/content/pdf/10.1007/s10710-007-9028-8.pdf page 148
        self.use_radial_distance = True # bias towards radial symmetry
        self.num_inputs = 4 # x,y,bias,d
        self.num_outputs = len(self.color_mode) # one for each color channel

        # only used for classification
        self.classification_image_size = (32, 32) # 32x32 is minimum

        # Autoencoder novelty
        self.autoencoder_frequency = 10
        
        # MAP-ELITES
        self.map_elites_resolution  = (8, 8)
        self.map_elites_max_values  = (1, 7)
        self.map_elites_min_values  = (0, 0)
        self.map_elites_expand_dims = False
        
        # clustering coefficent 
        self.clustering_fitness_ratio = 0
        
    def save_json(self, filename):
        with open(filename, 'w+') as outfile:
            strng = self.to_json()
            outfile.write(strng)
            outfile.close()
     
    def load_saved(self, filename):
        with open(filename, 'r') as infile:
            data = json.load(infile)
            self.from_json(data)
            infile.close()
            
        if self.train_image_path is not None:
            self.train_image = load_image(self.train_image_path, self.color_mode)


    def fns_to_strings(self):
        self.activations= [fn.__name__ for fn in self.activations]
        self.output_activation = self.output_activation.__name__ if self.output_activation is not None else "" 
        self.fitness_function = self.fitness_function.__name__ if self.fitness_function is not None else "" 

    def strings_to_fns(self):
       
        self.activations= [name_to_fn(name) for name in self.activations]
        # self.activations.append(avg_pixel_distance_fitness)
        self.output_activation = name_to_fn(self.output_activation)
        self.fitness_function = name_to_fn(self.fitness_function)

    def to_json(self):
        self.fns_to_strings()
        self.train_image = []
        if len(self.curriculum) > 0 and isinstance(self.curriculum[0], np.ndarray):
            self.curriculum = [c.tolist() for c in self.curriculum] 
        json_string = json.dumps(self.__dict__, sort_keys=True, indent=4)
        self.strings_to_fns()
        return json_string
    
    
    def from_json(self, json_dict):
        if type(json_dict) is str:
            json_dict = json.loads(json_dict)
        self.fns_to_strings()
        self.__dict__ = json_dict
        
        self.strings_to_fns()


    def CreateFromJson(json_str):
        if type(json_str) is str:
            json_str = json.loads(json_str)
        config = Config()
        config.__dict__ = json_str
        config.strings_to_fns()
        if config.train_image_path is not None:
            config.train_image = load_image(config.train_image_path, config.color_mode)
        return config


